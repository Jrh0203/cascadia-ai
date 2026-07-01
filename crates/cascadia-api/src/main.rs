use std::{net::SocketAddr, path::PathBuf};

use cascadia_api::router_with_cluster_history;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "cascadia-web-v2",
    version,
    about = "Local Cascadia v2 web server"
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8787")]
    listen: SocketAddr,
    #[arg(long, default_value = "apps/web/dist")]
    static_dir: PathBuf,
    #[arg(long)]
    api_only: bool,
    #[arg(
        long,
        default_value = "artifacts/cluster/telemetry-v1.jsonl",
        help = "Persistent seven-day cluster telemetry journal"
    )]
    cluster_history_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let static_dir = (!args.api_only).then_some(args.static_dir);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!("Cascadia v2 is listening on http://{}", args.listen);
    axum::serve(
        listener,
        router_with_cluster_history(static_dir, args.cluster_history_path)?,
    )
    .await?;
    Ok(())
}
