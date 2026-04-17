#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/liambarkley/Documents/Projects/Mach Cutoff/Mach Cutoff Simulation"
PYTHON_BIN="$ROOT/.venv/bin/python3"
LOG_DIR="$ROOT/results/logs"

MODE="${1:-fast}"
MAX_PARALLEL="${2:-4}"
case "$MODE" in
  fast)
    MANIFEST="$ROOT/results/configs/benchmark_xb1_paper_screening_limit100k_20260413.json"
    PREFIX="xb1_paper_screening_limit100k_20260413"
    ;;
  visual)
    MANIFEST="$ROOT/results/configs/benchmark_xb1_paper_screening_limit100k_visual_20260413.json"
    PREFIX="xb1_paper_screening_limit100k_visual_20260413"
    ;;
  *)
    echo "usage: $0 [fast|visual] [max_parallel]" >&2
    exit 2
    ;;
esac

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]] || [[ "$MAX_PARALLEL" -lt 1 ]]; then
  echo "max_parallel must be a positive integer" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"

corridors=("iah_mia" "den_slc" "dfw_atl" "iad_bos")
timestamps=("20250115T0000Z" "20250115T1200Z" "20250415T0000Z" "20250415T1200Z" "20250715T0000Z" "20250715T1200Z")
pids=()

count_live_pids() {
  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "0"
    return
  fi
  local live=0
  local pid
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      live=$((live + 1))
    fi
  done
  echo "$live"
}

for corridor in "${corridors[@]}"; do
  while [[ "$(count_live_pids)" -ge "$MAX_PARALLEL" ]]; do
    sleep 5
  done

  shard_log="$LOG_DIR/${PREFIX}__${corridor}.log"
  args=("$PYTHON_BIN" -u -m mach_cutoff.benchmark.cli --benchmark-config "$MANIFEST" --resume)
  for stamp in "${timestamps[@]}"; do
    args+=(--scenario-id "${corridor}__${stamp}")
  done
  (
    cd "$ROOT"
    "${args[@]}"
  ) >"$shard_log" 2>&1 &
  pids+=("$!")
  echo "[screening] launched corridor=$corridor pid=${pids[${#pids[@]}-1]} log=$shard_log max_parallel=$MAX_PARALLEL"
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[screening] one or more corridor shards failed" >&2
  exit 1
fi

aggregate_log="$LOG_DIR/${PREFIX}__aggregate.log"
(
  cd "$ROOT"
  "$PYTHON_BIN" -u -m mach_cutoff.benchmark.cli --benchmark-config "$MANIFEST" --resume
) >"$aggregate_log" 2>&1

echo "[screening] complete manifest=$MANIFEST aggregate_log=$aggregate_log"
