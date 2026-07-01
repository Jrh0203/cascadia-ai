//! Row-major SGEMM façade for AZ-v2 hot paths.
//!
//! Three backends, gated by Cargo features:
//!
//! | Feature              | Backend                                     | Note                            |
//! |----------------------|---------------------------------------------|---------------------------------|
//! | `accelerate`         | Apple Accelerate `cblas_sgemm` (cblas-sys)  | Fastest on Apple Silicon.       |
//! | `matmul-portable`    | Pure-Rust `matrixmultiply::sgemm`           | Cross-platform fallback.        |
//! | (neither)            | Naive triple-nested-loop scalar fallback    | Preserves zero-feature parity.  |
//!
//! When both `accelerate` and `matmul-portable` are enabled, `accelerate` wins.
//!
//! Every call must satisfy `a.len() == m * k`, `b.len() == k * n`,
//! `c.len() == m * n`. All buffers are contiguous row-major.

#![allow(dead_code)]

/// `C := alpha * A @ B + beta * C` in row-major layout.
///
/// `A` is `(m, k)`, `B` is `(k, n)`, `C` is `(m, n)`. All inputs are flat
/// slices in row-major order (no strided slicing).
///
/// Hot path: invoked once per HexConv layer in the v2 forward; the
/// `m × n × k` product dominates wall-clock when no SGEMM feature is
/// enabled, so prefer the `accelerate` or `matmul-portable` feature for
/// any real training run.
#[inline]
pub fn sgemm_rm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    debug_assert_eq!(a.len(), m * k, "A has wrong shape (m={m}, k={k})");
    debug_assert_eq!(b.len(), k * n, "B has wrong shape (k={k}, n={n})");
    debug_assert_eq!(c.len(), m * n, "C has wrong shape (m={m}, n={n})");
    if m == 0 || n == 0 {
        return;
    }
    sgemm_dispatch(m, n, k, alpha, a, b, beta, c);
}

/// `C := alpha * A @ B^T + beta * C` in row-major layout.
///
/// `A` is `(m, k)`, `B` is `(n, k)` — note the transpose: B's natural row-major
/// shape is `(n, k)` but the matmul treats it as `(k, n)` via cblas's TransB
/// flag (Accelerate / matrixmultiply both support this with no extra copy).
///
/// Used by SAB and CrossAttn forwards where every weight `W` is stored as
/// `(out, in)` row-major but the math is `X @ W^T`. Lets us skip carrying
/// transposed shadow weights.
#[inline]
pub fn sgemm_rm_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    debug_assert_eq!(a.len(), m * k, "A has wrong shape (m={m}, k={k})");
    debug_assert_eq!(
        b.len(),
        n * k,
        "B has wrong shape (n={n}, k={k}) — B^T is k×n"
    );
    debug_assert_eq!(c.len(), m * n, "C has wrong shape (m={m}, n={n})");
    if m == 0 || n == 0 {
        return;
    }
    sgemm_dispatch_nt(m, n, k, alpha, a, b, beta, c);
}

// ─── Backend dispatch ──────────────────────────────────────────────────

#[cfg(feature = "accelerate")]
#[inline]
fn sgemm_dispatch(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_accelerate(m, n, k, alpha, a, b, beta, c);
}

#[cfg(all(feature = "matmul-portable", not(feature = "accelerate")))]
#[inline]
fn sgemm_dispatch(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_matrixmultiply(m, n, k, alpha, a, b, beta, c);
}

#[cfg(not(any(feature = "accelerate", feature = "matmul-portable")))]
#[inline]
fn sgemm_dispatch(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_scalar(m, n, k, alpha, a, b, beta, c);
}

#[cfg(feature = "accelerate")]
#[inline]
fn sgemm_dispatch_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_accelerate_nt(m, n, k, alpha, a, b, beta, c);
}

#[cfg(all(feature = "matmul-portable", not(feature = "accelerate")))]
#[inline]
fn sgemm_dispatch_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_matrixmultiply_nt(m, n, k, alpha, a, b, beta, c);
}

#[cfg(not(any(feature = "accelerate", feature = "matmul-portable")))]
#[inline]
fn sgemm_dispatch_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_scalar_nt(m, n, k, alpha, a, b, beta, c);
}

// ─── Backend implementations ──────────────────────────────────────────

#[cfg(feature = "accelerate")]
fn sgemm_accelerate(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    // Pull `accelerate-src` into the link line (it only provides a build
    // script that emits `-framework Accelerate`); no items to use here.
    extern crate accelerate_src as _;
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            beta,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(feature = "matmul-portable")]
fn sgemm_matrixmultiply(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    // matrixmultiply::sgemm: row-major when row stride = k, col stride = 1
    // for A; row stride = n, col stride = 1 for B and C.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(feature = "accelerate")]
fn sgemm_accelerate_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    extern crate accelerate_src as _;
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
    // B is stored row-major as (n, k); we want B^T treated as (k, n). cblas
    // handles this with TransB=CblasTrans + ldb=k (leading dim is the row
    // stride of B in its native (n, k) layout).
    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            CBLAS_TRANSPOSE::CblasNoTrans,
            CBLAS_TRANSPOSE::CblasTrans,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            beta,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

#[cfg(feature = "matmul-portable")]
fn sgemm_matrixmultiply_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    // matrixmultiply::sgemm accepts arbitrary strides. To "transpose" B
    // (stored row-major as (n, k)) we swap row and col strides: present it
    // as (k, n) with row stride 1, col stride k.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// Scalar reference for the B-transposed path. C += A @ B^T.
#[inline(always)]
fn sgemm_scalar_nt(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    if beta == 0.0 {
        for row in c.iter_mut() {
            *row = 0.0;
        }
    } else if beta != 1.0 {
        for row in c.iter_mut() {
            *row *= beta;
        }
    }
    // A: (m, k) row-major. B (stored as (n, k) row-major) is treated as B^T.
    // C[i, j] += alpha * sum_l A[i, l] * B[j, l]  (B[j, l] is B^T[l, j]).
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for l in 0..k {
                s += a[i * k + l] * b[j * k + l];
            }
            c[i * n + j] += alpha * s;
        }
    }
}

/// Scalar reference implementation. Slow, but bit-exact across compiles —
/// used to anchor parity tests and as the no-feature fallback.
#[inline(always)]
fn sgemm_scalar(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    if beta == 0.0 {
        for row in c.iter_mut() {
            *row = 0.0;
        }
    } else if beta != 1.0 {
        for row in c.iter_mut() {
            *row *= beta;
        }
    }
    for i in 0..m {
        for l in 0..k {
            let a_il = alpha * a[i * k + l];
            for j in 0..n {
                c[i * n + j] += a_il * b[l * n + j];
            }
        }
    }
}

/// Public scalar reference for parity tests, independent of which feature is
/// active. Tests call this to compute the "expected" tensor and compare to
/// the feature-selected `sgemm_rm` output under tolerance.
pub fn sgemm_scalar_reference(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_scalar(m, n, k, alpha, a, b, beta, c);
}

/// Public scalar reference for the B-transposed variant.
pub fn sgemm_scalar_nt_reference(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    sgemm_scalar_nt(m, n, k, alpha, a, b, beta, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    /// Whatever backend the build selected matches the scalar reference
    /// to within a tight tolerance.
    #[test]
    fn sgemm_rm_matches_scalar_reference_small() {
        let (m, n, k) = (8, 4, 16);
        let a = rand_vec(m * k, 0x100);
        let b = rand_vec(k * n, 0x101);
        let mut c_test = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_rm(m, n, k, 1.0, &a, &b, 0.0, &mut c_test);
        sgemm_scalar_reference(m, n, k, 1.0, &a, &b, 0.0, &mut c_ref);
        for (i, (t, r)) in c_test.iter().zip(c_ref.iter()).enumerate() {
            let diff = (t - r).abs();
            assert!(
                diff < 5e-5,
                "sgemm_rm[{i}] = {t} vs scalar {r}; diff {diff} exceeds 5e-5"
            );
        }
    }

    /// Shape used by HexConv stem (in_c=72 × 7 cols, out_c=96 channels,
    /// over 128 cells) — exercises the actual hot-path matmul size.
    #[test]
    fn sgemm_rm_matches_scalar_reference_hexconv_stem() {
        let (m, n, k) = (96, 128, 72 * 7);
        let a = rand_vec(m * k, 0x200);
        let b = rand_vec(k * n, 0x201);
        let mut c_test = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_rm(m, n, k, 1.0, &a, &b, 0.0, &mut c_test);
        sgemm_scalar_reference(m, n, k, 1.0, &a, &b, 0.0, &mut c_ref);
        let mut max_abs = 0.0f32;
        for (t, r) in c_test.iter().zip(c_ref.iter()) {
            max_abs = max_abs.max((t - r).abs());
        }
        assert!(
            max_abs < 5e-4,
            "stem-shape max-abs-diff = {max_abs} exceeds 5e-4"
        );
    }

    /// `sgemm_rm_nt` matches the scalar reference under tolerance.
    #[test]
    fn sgemm_rm_nt_matches_scalar_reference() {
        // SAB QKV-projection shape: tokens=10, d=64. QKV emits 3d cols, so
        // C is (n_tokens, 3d) = (10, 192) via X @ W^T where W is (3d, d).
        let (m, n, k) = (10, 192, 64);
        let a = rand_vec(m * k, 0x400);
        let b = rand_vec(n * k, 0x401);
        let mut c_test = vec![0.0f32; m * n];
        let mut c_ref = vec![0.0f32; m * n];
        sgemm_rm_nt(m, n, k, 1.0, &a, &b, 0.0, &mut c_test);
        sgemm_scalar_nt_reference(m, n, k, 1.0, &a, &b, 0.0, &mut c_ref);
        let mut max_abs = 0.0f32;
        for (t, r) in c_test.iter().zip(c_ref.iter()) {
            max_abs = max_abs.max((t - r).abs());
        }
        assert!(
            max_abs < 5e-4,
            "sgemm_rm_nt max-abs-diff = {max_abs} exceeds 5e-4"
        );
    }

    /// Confirm beta=1.0 accumulation (used for bias-add patterns).
    #[test]
    fn sgemm_rm_accumulates_with_beta_one() {
        let (m, n, k) = (8, 4, 16);
        let a = rand_vec(m * k, 0x300);
        let b = rand_vec(k * n, 0x301);
        let c_init = rand_vec(m * n, 0x302);
        let mut c_test = c_init.clone();
        let mut c_ref = c_init.clone();
        sgemm_rm(m, n, k, 1.0, &a, &b, 1.0, &mut c_test);
        sgemm_scalar_reference(m, n, k, 1.0, &a, &b, 1.0, &mut c_ref);
        for (t, r) in c_test.iter().zip(c_ref.iter()) {
            assert!((t - r).abs() < 5e-5);
        }
    }
}
