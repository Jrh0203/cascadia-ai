use clap::Parser;
use t1_horizon_search::{Args, run};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    run(Args::parse())
}
