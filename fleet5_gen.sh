set -euo pipefail
cd ~/cascadia
M=cascadiav3/checkpoints/full_v3_distq_k8/best_locked_val.manifest.json
./cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --gumbel-selfplay-tensor-corpus \
  --model-service "env CASCADIA_CGAB_FUSED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src venv/bin/python -m cascadiav3.torch_inference_bridge --manifest $M --device mps" \
  --model-manifest $M --model-timeout-ms 120000 \
  --gumbel-n-simulations 256 --gumbel-top-m 16 --gumbel-depth-rounds 1 \
  --gumbel-determinizations 4 --gumbel-blend-weight 0.75 --k-interior 16 \
  --model-sessions 3 --shared-model-session \
  --first-seed 2026815000 --seed-count 150 --plies-per-seed 80 \
  --max-actions 8 --rollouts-per-action 1 --rollout-top-k 4 \
  --tensor-compression stored --rayon-threads 8 \
  --out ~/cascadia/fleet5_shard_john1.npz \
  --manifest ~/cascadia/fleet5_shard_john1_manifest.json
echo "FLEET5_DONE_john1 $(date -Is)"
