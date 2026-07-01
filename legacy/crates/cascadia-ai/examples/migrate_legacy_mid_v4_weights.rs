use cascadia_ai::nnue::NNUENetwork;
use std::path::Path;

fn main() {
    let mut args = std::env::args_os();
    let program = args
        .next()
        .and_then(|value| value.into_string().ok())
        .unwrap_or_else(|| "migrate_legacy_mid_v4_weights".to_string());
    let input = args.next();
    let output = args.next();
    if input.is_none() || output.is_none() || args.next().is_some() {
        eprintln!("usage: {program} <historical-nnue.bin> <corrected-nnue.bin>");
        std::process::exit(2);
    }

    let input = input.expect("checked above");
    let output = output.expect("checked above");
    if let Err(error) =
        NNUENetwork::migrate_legacy_mid_v4_weights(Path::new(&input), Path::new(&output))
    {
        eprintln!("migration failed: {error}");
        std::process::exit(1);
    }
}
