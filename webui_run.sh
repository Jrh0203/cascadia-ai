set -euo pipefail
cd ~/cascadia
export CASCADIA_CHAMPION_CMD="bash $HOME/cascadia/champion_server.sh"
exec ./target/release/cascadia-api --listen 0.0.0.0:8787 --static-dir apps/web/dist
