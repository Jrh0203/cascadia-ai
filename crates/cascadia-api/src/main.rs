use std::{net::SocketAddr, path::PathBuf};

use cascadia_api::router_with_cluster_paths_and_experiments_and_r2_map;
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
    #[arg(
        long,
        default_value = "artifacts/cluster/research-queue-v1.json",
        help = "Manifest-backed cluster research queue"
    )]
    cluster_queue_path: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/cluster/research-experiments-v1.json",
        help = "Durable research experiment ledger"
    )]
    cluster_experiments_path: PathBuf,
    #[arg(
        long,
        default_value = "artifacts/cluster/r2-map-dashboard-serving-projection-v2.json",
        help = "Compact hash-bound R2-MAP dashboard serving projection"
    )]
    r2_map_status_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let static_dir = (!args.api_only).then_some(args.static_dir);
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    println!("Cascadia v2 is listening on http://{}", args.listen);
    axum::serve(
        listener,
        router_with_cluster_paths_and_experiments_and_r2_map(
            static_dir,
            args.cluster_history_path,
            args.cluster_queue_path,
            args.cluster_experiments_path,
            args.r2_map_status_path,
        )?,
    )
    .await?;
    Ok(())
}
