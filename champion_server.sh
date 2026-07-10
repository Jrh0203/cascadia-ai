#!/usr/bin/env bash
# Champion suggest server: distq_k8 + Gumbel n256/d4 on MPS.
cd ~/cascadia
M=cascadiav3/checkpoints/full_v3_distq_k8/best_locked_val.manifest.json
exec ./cascadiav3/real-root-exporter/target/release/cascadiav3-real-root-exporter \
  --gumbel-suggest-server \
  --model-service "env CASCADIA_CGAB_FUSED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src venv/bin/python -m cascadiav3.torch_inference_bridge --manifest $M --device mps" \
  --model-manifest $M --model-timeout-ms 120000 \
  --gumbel-n-simulations 256 --gumbel-top-m 16 --gumbel-depth-rounds 1 \
  --gumbel-determinizations 4 --gumbel-blend-weight 0.5 --k-interior 16
