use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-env-changed=F5_SOURCE_GIT_REVISION");
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is required"));
    let revision = env::var("F5_SOURCE_GIT_REVISION")
        .ok()
        .or_else(|| git_output(&manifest_dir, &["rev-parse", "HEAD"]))
        .unwrap_or_else(|| {
            panic!(
                "F5 source Git revision is unavailable; build in a Git checkout or set F5_SOURCE_GIT_REVISION"
            )
        });
    if !(40..=64).contains(&revision.len())
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        panic!("F5 source Git revision must be a 40-64 character hexadecimal object ID");
    }
    if let Some(git_dir) = git_output(&manifest_dir, &["rev-parse", "--absolute-git-dir"]) {
        println!("cargo:rerun-if-changed={git_dir}/HEAD");
    }
    println!(
        "cargo:rustc-env=F5_SOURCE_GIT_REVISION={}",
        revision.to_ascii_lowercase()
    );
}

fn git_output(directory: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .current_dir(directory)
        .args(args)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}
