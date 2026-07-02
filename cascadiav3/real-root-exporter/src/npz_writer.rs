use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Result, bail};
use zip::write::FileOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpzCompression {
    Deflate,
    Stored,
}

impl NpzCompression {
    fn method(self) -> zip::CompressionMethod {
        match self {
            Self::Deflate => zip::CompressionMethod::Deflated,
            Self::Stored => zip::CompressionMethod::Stored,
        }
    }
}

fn shape_tuple(shape: &[usize]) -> String {
    match shape {
        [one] => format!("({},)", one),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn write_npy_header<W: Write>(writer: &mut W, descr: &str, shape: &[usize]) -> Result<()> {
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr,
        shape_tuple(shape)
    );
    let preamble_len = 10usize;
    let newline_len = 1usize;
    let padding = (16 - ((preamble_len + header.len() + newline_len) % 16)) % 16;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    if header.len() > u16::MAX as usize {
        bail!("npy v1 header too large: {} bytes", header.len());
    }
    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])?;
    writer.write_all(&(header.len() as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    Ok(())
}

fn write_u16_npy<W: Write>(writer: &mut W, values: &[u16], shape: &[usize]) -> Result<()> {
    write_npy_header(writer, "<f2", shape)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_i16_npy<W: Write>(writer: &mut W, values: &[i16], shape: &[usize]) -> Result<()> {
    write_npy_header(writer, "<i2", shape)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_i32_npy<W: Write>(writer: &mut W, values: &[i32], shape: &[usize]) -> Result<()> {
    write_npy_header(writer, "<i4", shape)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_i64_npy<W: Write>(writer: &mut W, values: &[i64], shape: &[usize]) -> Result<()> {
    write_npy_header(writer, "<i8", shape)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32_npy<W: Write>(writer: &mut W, values: &[f32], shape: &[usize]) -> Result<()> {
    write_npy_header(writer, "<f4", shape)?;
    for value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_u8_npy<W: Write>(writer: &mut W, values: &[u8], shape: &[usize]) -> Result<()> {
    write_npy_header(writer, "|u1", shape)?;
    writer.write_all(values)?;
    Ok(())
}

fn write_bytes_scalar_npy<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    write_npy_header(writer, &format!("|S{}", value.len()), &[1])?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn start_file<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    name: &str,
    compression: NpzCompression,
) -> zip::result::ZipResult<()> {
    let options = FileOptions::default().compression_method(compression.method());
    zip.start_file(name, options)
}

pub struct GreedyTensorNpz<'a> {
    pub version: &'a str,
    pub metadata_json: &'a str,
    pub tokens_f16_bits: &'a [u16],
    pub token_shape: [usize; 2],
    pub actions_f16_bits: &'a [u16],
    pub action_shape: [usize; 2],
    pub token_offsets: &'a [i64],
    pub action_offsets: &'a [i64],
    pub selected_action_index: &'a [i16],
    pub compression: NpzCompression,
}

pub fn write_greedy_tensor_npz(path: &Path, shard: GreedyTensorNpz<'_>) -> Result<()> {
    let file = BufWriter::new(File::create(path)?);
    let mut zip = zip::ZipWriter::new(file);

    start_file(&mut zip, "version.npy", shard.compression)?;
    write_bytes_scalar_npy(&mut zip, shard.version)?;

    start_file(&mut zip, "metadata_json.npy", shard.compression)?;
    write_bytes_scalar_npy(&mut zip, shard.metadata_json)?;

    start_file(&mut zip, "tokens.npy", shard.compression)?;
    write_u16_npy(&mut zip, shard.tokens_f16_bits, &shard.token_shape)?;

    start_file(&mut zip, "actions.npy", shard.compression)?;
    write_u16_npy(&mut zip, shard.actions_f16_bits, &shard.action_shape)?;

    start_file(&mut zip, "token_offsets.npy", shard.compression)?;
    write_i64_npy(&mut zip, shard.token_offsets, &[shard.token_offsets.len()])?;

    start_file(&mut zip, "action_offsets.npy", shard.compression)?;
    write_i64_npy(
        &mut zip,
        shard.action_offsets,
        &[shard.action_offsets.len()],
    )?;

    start_file(&mut zip, "selected_action_index.npy", shard.compression)?;
    write_i16_npy(
        &mut zip,
        shard.selected_action_index,
        &[shard.selected_action_index.len()],
    )?;

    zip.finish()?;
    Ok(())
}

pub struct ExpertTensorNpz<'a> {
    pub version: &'a str,
    pub metadata_json: &'a str,
    pub tokens_f16_bits: &'a [u16],
    pub token_shape: [usize; 2],
    pub actions_f16_bits: &'a [u16],
    pub action_shape: [usize; 2],
    pub token_offsets: &'a [i64],
    pub action_offsets: &'a [i64],
    pub relation_edges_i32: &'a [i32],
    pub relation_edge_shape: [usize; 2],
    pub relation_offsets: &'a [i64],
    pub selected_action_index: &'a [i16],
    pub target_q: &'a [f32],
    pub target_score_to_go: &'a [f32],
    pub q_valid: &'a [u8],
    pub priors: &'a [f32],
    pub visits: &'a [f32],
    pub q_variance: &'a [f32],
    pub q_count: &'a [f32],
    pub truncated_count: &'a [f32],
    pub exact_afterstate_score_active: &'a [f32],
    pub final_score_vector: &'a [f32],
    pub rank_vector: &'a [i16],
    pub score_decomposition: &'a [f32],
    /// v2-only arrays; omitted from the archive when `None`.
    pub improved_policy: Option<&'a [f32]>,
    pub search_root_value: Option<&'a [f32]>,
    pub record_count: usize,
    pub compression: NpzCompression,
}

pub fn write_expert_tensor_npz(path: &Path, shard: ExpertTensorNpz<'_>) -> Result<()> {
    let file = BufWriter::new(File::create(path)?);
    let mut zip = zip::ZipWriter::new(file);

    start_file(&mut zip, "version.npy", shard.compression)?;
    write_bytes_scalar_npy(&mut zip, shard.version)?;

    start_file(&mut zip, "metadata_json.npy", shard.compression)?;
    write_bytes_scalar_npy(&mut zip, shard.metadata_json)?;

    start_file(&mut zip, "tokens.npy", shard.compression)?;
    write_u16_npy(&mut zip, shard.tokens_f16_bits, &shard.token_shape)?;

    start_file(&mut zip, "actions.npy", shard.compression)?;
    write_u16_npy(&mut zip, shard.actions_f16_bits, &shard.action_shape)?;

    start_file(&mut zip, "token_offsets.npy", shard.compression)?;
    write_i64_npy(&mut zip, shard.token_offsets, &[shard.token_offsets.len()])?;

    start_file(&mut zip, "action_offsets.npy", shard.compression)?;
    write_i64_npy(
        &mut zip,
        shard.action_offsets,
        &[shard.action_offsets.len()],
    )?;

    start_file(&mut zip, "relation_edges.npy", shard.compression)?;
    write_i32_npy(
        &mut zip,
        shard.relation_edges_i32,
        &shard.relation_edge_shape,
    )?;

    start_file(&mut zip, "relation_offsets.npy", shard.compression)?;
    write_i64_npy(
        &mut zip,
        shard.relation_offsets,
        &[shard.relation_offsets.len()],
    )?;

    start_file(&mut zip, "selected_action_index.npy", shard.compression)?;
    write_i16_npy(
        &mut zip,
        shard.selected_action_index,
        &[shard.selected_action_index.len()],
    )?;

    for (name, values) in [
        ("target_q.npy", shard.target_q),
        ("target_score_to_go.npy", shard.target_score_to_go),
        ("priors.npy", shard.priors),
        ("visits.npy", shard.visits),
        ("q_variance.npy", shard.q_variance),
        ("q_count.npy", shard.q_count),
        ("truncated_count.npy", shard.truncated_count),
        (
            "exact_afterstate_score_active.npy",
            shard.exact_afterstate_score_active,
        ),
    ] {
        start_file(&mut zip, name, shard.compression)?;
        write_f32_npy(&mut zip, values, &[values.len()])?;
    }

    start_file(&mut zip, "q_valid.npy", shard.compression)?;
    write_u8_npy(&mut zip, shard.q_valid, &[shard.q_valid.len()])?;

    start_file(&mut zip, "final_score_vector.npy", shard.compression)?;
    write_f32_npy(&mut zip, shard.final_score_vector, &[shard.record_count, 4])?;

    start_file(&mut zip, "rank_vector.npy", shard.compression)?;
    write_i16_npy(&mut zip, shard.rank_vector, &[shard.record_count, 4])?;

    start_file(&mut zip, "score_decomposition.npy", shard.compression)?;
    write_f32_npy(
        &mut zip,
        shard.score_decomposition,
        &[shard.record_count, 3, 4],
    )?;

    if let Some(improved_policy) = shard.improved_policy {
        start_file(&mut zip, "improved_policy.npy", shard.compression)?;
        write_f32_npy(&mut zip, improved_policy, &[improved_policy.len()])?;
    }
    if let Some(search_root_value) = shard.search_root_value {
        start_file(&mut zip, "search_root_value.npy", shard.compression)?;
        write_f32_npy(&mut zip, search_root_value, &[search_root_value.len()])?;
    }

    zip.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::shape_tuple;

    #[test]
    fn shape_tuple_matches_numpy_singleton_convention() {
        assert_eq!(shape_tuple(&[1]), "(1,)");
        assert_eq!(shape_tuple(&[2, 3]), "(2, 3)");
    }
}
