#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/liambarkley/Documents/Projects/Mach Cutoff/Mach Cutoff Simulation"
PYTHON_BIN="$ROOT/.venv/bin/python3"
CLI_MODULE="mach_cutoff.benchmark.cli"
LOG_DIR="$ROOT/results/logs"

mkdir -p "$LOG_DIR"

run_manifest_parallel() {
  local manifest_rel="$1"
  local manifest_name
  manifest_name="$(basename "$manifest_rel" .json)"
  local manifest_path="$ROOT/$manifest_rel"

  local -a corridors=(
    "iah_mia"
    "lax_sfo"
    "msp_dtw"
    "den_slc"
    "dfw_atl"
    "iad_bos"
  )

  local -a timestamps=(
    "20250115T0000Z"
    "20250115T1200Z"
    "20250415T0000Z"
    "20250415T1200Z"
    "20250715T0000Z"
    "20250715T1200Z"
  )

  echo "[batch] starting manifest $manifest_name"

  local -a pids=()
  local -a shard_logs=()

  for corridor in "${corridors[@]}"; do
    local shard_log="$LOG_DIR/${manifest_name}__${corridor}.log"
    shard_logs+=("$shard_log")

    local -a args=(
      "$PYTHON_BIN" -u -m "$CLI_MODULE"
      --benchmark-config "$manifest_path"
      --resume
    )

    for stamp in "${timestamps[@]}"; do
      args+=(--scenario-id "${corridor}__${stamp}")
    done

    (
      cd "$ROOT"
      "${args[@]}"
    ) >"$shard_log" 2>&1 &

    local pid="$!"
    pids+=("$pid")
    echo "[batch] launched $manifest_name corridor=$corridor pid=$pid log=$shard_log"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "[batch] shard failure for $manifest_name"
    return 1
  fi

  local aggregate_log="$LOG_DIR/${manifest_name}__aggregate.log"
  echo "[batch] rebuilding aggregate outputs for $manifest_name"
  (
    cd "$ROOT"
    "$PYTHON_BIN" -u -m "$CLI_MODULE" --benchmark-config "$manifest_path" --resume
  ) >"$aggregate_log" 2>&1

  echo "[batch] completed manifest $manifest_name"
}

main() {
  run_manifest_parallel "results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json"
  run_manifest_parallel "results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json"
  run_manifest_parallel "results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json"
}

main "$@"
