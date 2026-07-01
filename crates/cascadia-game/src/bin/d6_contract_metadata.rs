use std::{
    env,
    error::Error,
    fmt::Write as _,
    fs,
    io::{self, Write as _},
    path::{Path, PathBuf},
};

use cascadia_game::{D6ContractMetadata, D6TransformMetadata, d6_contract_metadata};

const EXPECTED_SCIENTIFIC_BLAKE3: &str =
    "db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f";

enum Command {
    Stdout,
    Output(PathBuf),
    Check(PathBuf),
}

fn main() -> Result<(), Box<dyn Error>> {
    let command = parse_command()?;
    let metadata = d6_contract_metadata();
    if metadata.scientific_blake3 != EXPECTED_SCIENTIFIC_BLAKE3 {
        return Err(format!(
            "Rust D6 scientific hash changed: expected {EXPECTED_SCIENTIFIC_BLAKE3}, got {}",
            metadata.scientific_blake3
        )
        .into());
    }
    let encoded = encode_metadata_json(&metadata);

    match command {
        Command::Stdout => io::stdout().lock().write_all(encoded.as_bytes())?,
        Command::Output(path) => write_atomically(&path, encoded.as_bytes())?,
        Command::Check(path) => {
            let bundled = fs::read(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            if bundled != encoded.as_bytes() {
                return Err(format!(
                    "D6 metadata drift at {}; regenerate it with `cargo run -p cascadia-game \
                     --bin d6_contract_metadata -- --output {}`",
                    path.display(),
                    path.display()
                )
                .into());
            }
            eprintln!(
                "D6 metadata matches Rust contract at {} ({EXPECTED_SCIENTIFIC_BLAKE3})",
                path.display()
            );
        }
    }
    Ok(())
}

fn parse_command() -> Result<Command, Box<dyn Error>> {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [] => Ok(Command::Stdout),
        [flag] if flag == "--stdout" => Ok(Command::Stdout),
        [flag, path] if flag == "--output" => Ok(Command::Output(path.into())),
        [flag, path] if flag == "--check" => Ok(Command::Check(path.into())),
        _ => Err(
            "usage: d6_contract_metadata [--stdout | --output PATH | --check PATH]"
                .to_owned()
                .into(),
        ),
    }
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("output path has no UTF-8 file name: {}", path.display()))?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", std::process::id()));
    fs::write(&temporary, bytes)?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(error.into());
    }
    Ok(())
}

fn encode_metadata_json(metadata: &D6ContractMetadata) -> String {
    let mut output = String::with_capacity(8_192);
    output.push_str("{\n");
    writeln!(output, "  \"schema_version\": {},", metadata.schema_version).unwrap();
    output.push_str("  \"contract_id\": ");
    write_json_string(&mut output, &metadata.contract_id);
    output.push_str(",\n  \"edge_order\": ");
    write_string_array(&mut output, &metadata.edge_order);
    output.push_str(",\n  \"coordinate_matrices\": ");
    write_i8_matrices(&mut output, &metadata.coordinate_matrices);
    output.push_str(",\n  \"direction_tables\": ");
    write_u8_table(&mut output, &metadata.direction_tables);
    output.push_str(",\n  \"dual_tile_rotation_tables\": ");
    write_u8_table(&mut output, &metadata.dual_tile_rotation_tables);
    output.push_str(",\n  \"single_tile_rotation_tables\": ");
    write_u8_table(&mut output, &metadata.single_tile_rotation_tables);
    output.push_str(",\n  \"inverse_table\": ");
    write_u8_array(&mut output, &metadata.inverse_table);
    output.push_str(",\n  \"composition_table\": ");
    write_u8_table(&mut output, &metadata.composition_table);
    output.push_str(",\n  \"transforms\": ");
    write_transforms(&mut output, &metadata.transforms);
    output.push_str(",\n  \"scientific_blake3\": ");
    write_json_string(&mut output, &metadata.scientific_blake3);
    output.push_str("\n}\n");
    output
}

fn write_json_string(output: &mut String, value: &str) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                write!(output, "\\u{:04x}", character as u32).unwrap();
            }
            character => output.push(character),
        }
    }
    output.push('"');
}

fn write_string_array<const N: usize>(output: &mut String, values: &[String; N]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write_json_string(output, value);
    }
    output.push(']');
}

fn write_i8_matrices<const N: usize>(output: &mut String, matrices: &[[[i8; 2]; 2]; N]) {
    output.push_str("[\n");
    for (index, matrix) in matrices.iter().enumerate() {
        write!(
            output,
            "    [[{}, {}], [{}, {}]]{}",
            matrix[0][0],
            matrix[0][1],
            matrix[1][0],
            matrix[1][1],
            if index + 1 == N { "" } else { "," }
        )
        .unwrap();
        output.push('\n');
    }
    output.push_str("  ]");
}

fn write_u8_array<const N: usize>(output: &mut String, values: &[u8; N]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push_str(", ");
        }
        write!(output, "{value}").unwrap();
    }
    output.push(']');
}

fn write_u8_table<const ROWS: usize, const COLUMNS: usize>(
    output: &mut String,
    table: &[[u8; COLUMNS]; ROWS],
) {
    output.push_str("[\n");
    for (index, row) in table.iter().enumerate() {
        output.push_str("    ");
        write_u8_array(output, row);
        if index + 1 != ROWS {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ]");
}

fn write_transforms<const N: usize>(output: &mut String, transforms: &[D6TransformMetadata; N]) {
    output.push_str("[\n");
    for (index, transform) in transforms.iter().enumerate() {
        output.push_str("    {\"id\": ");
        write!(output, "{}", transform.id).unwrap();
        output.push_str(", \"rotation_steps\": ");
        write!(output, "{}", transform.rotation_steps).unwrap();
        output.push_str(", \"reflected\": ");
        output.push_str(if transform.reflected { "true" } else { "false" });
        output.push_str(", \"name\": ");
        write_json_string(output, &transform.name);
        output.push('}');
        if index + 1 != N {
            output.push(',');
        }
        output.push('\n');
    }
    output.push_str("  ]");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_json_round_trips_through_serde() {
        let metadata = d6_contract_metadata();
        let first = encode_metadata_json(&metadata);
        let second = encode_metadata_json(&metadata);
        assert_eq!(first, second);

        let decoded: D6ContractMetadata = serde_json::from_str(&first).unwrap();
        assert_eq!(decoded, metadata);
        assert_eq!(decoded.scientific_blake3, EXPECTED_SCIENTIFIC_BLAKE3);
    }
}
