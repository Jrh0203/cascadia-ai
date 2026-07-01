use std::{env, error::Error, io, path::PathBuf};

use cascadia_differential::full_legal_audit::read_full_legal_audit_shard;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    if args.next().as_deref() != Some("validate".as_ref())
        || args.next().as_deref() != Some("--input".as_ref())
    {
        return Err("usage: full-legal-roundtrip-validator validate --input PATH".into());
    }
    let input = PathBuf::from(args.next().ok_or("missing input path")?);
    if args.next().is_some() {
        return Err("unexpected trailing arguments".into());
    }

    let shard = read_full_legal_audit_shard(&input)?;
    serde_json::to_writer_pretty(io::stdout().lock(), &shard.summary)?;
    println!();
    Ok(())
}
