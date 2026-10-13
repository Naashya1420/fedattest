#!/usr/bin/env bash
# Runs every scenario concurrently. Each client container is capped by CUDA MPS
# at <=28% of the GPU's SMs and uses ~1 CPU / <1 GB RAM, so the scenarios can
# share the GPU. Learning results are unaffected by the sharing; emulated
# wall-clock timings are taken from an isolated run instead (runs/smoke).
#   usage: scripts/run_all.sh [scenario ...]
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
python3 -c "from bqfl import emulation; emulation.ensure_mps()"
SCEN=("$@")
[ ${#SCEN[@]} -eq 0 ] && SCEN=(attack_nodefense attack_attestation clean remove_client3_only remove_client7_only)
for s in "${SCEN[@]}"; do
  ( python3 -m bqfl.server "$s" --resume >> "logs/$s.log" 2>&1 || echo "FAILED $s" >> logs/failures.txt ) &
  sleep 5
done
wait
echo "DONE ${SCEN[*]} $(date)" >> logs/failures.txt
