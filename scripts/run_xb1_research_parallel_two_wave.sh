#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/liambarkley/Documents/Projects/Mach Cutoff/Mach Cutoff Simulation"
LOG_DIR="$ROOT/results/logs"

mkdir -p "$LOG_DIR"

run_manifest_parallel() {
  "$ROOT/scripts/run_xb1_research_parallel.sh" "$1"
}

# Wrapper around the existing per-manifest launcher. This script keeps at most
# two full manifests active at once (12 corridor shards total), which fits the
# current 10-core machine much better than serial execution without jumping
# straight to 18 concurrent shards.
main() {
  local log50="$LOG_DIR/twowave_limit50k_20260413.log"
  local log100="$LOG_DIR/twowave_limit100k_20260413.log"
  local log300="$LOG_DIR/twowave_limit300k_20260413.log"

  (
    cd "$ROOT"
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume \
      --scenario-id iah_mia__20250115T0000Z --scenario-id iah_mia__20250115T1200Z --scenario-id iah_mia__20250415T0000Z --scenario-id iah_mia__20250415T1200Z --scenario-id iah_mia__20250715T0000Z --scenario-id iah_mia__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__iah_mia.log" &
    p1=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume \
      --scenario-id lax_sfo__20250115T0000Z --scenario-id lax_sfo__20250115T1200Z --scenario-id lax_sfo__20250415T0000Z --scenario-id lax_sfo__20250415T1200Z --scenario-id lax_sfo__20250715T0000Z --scenario-id lax_sfo__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__lax_sfo.log" &
    p2=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume \
      --scenario-id msp_dtw__20250115T0000Z --scenario-id msp_dtw__20250115T1200Z --scenario-id msp_dtw__20250415T0000Z --scenario-id msp_dtw__20250415T1200Z --scenario-id msp_dtw__20250715T0000Z --scenario-id msp_dtw__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__msp_dtw.log" &
    p3=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume \
      --scenario-id den_slc__20250115T0000Z --scenario-id den_slc__20250115T1200Z --scenario-id den_slc__20250415T0000Z --scenario-id den_slc__20250415T1200Z --scenario-id den_slc__20250715T0000Z --scenario-id den_slc__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__den_slc.log" &
    p4=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume \
      --scenario-id dfw_atl__20250115T0000Z --scenario-id dfw_atl__20250115T1200Z --scenario-id dfw_atl__20250415T0000Z --scenario-id dfw_atl__20250415T1200Z --scenario-id dfw_atl__20250715T0000Z --scenario-id dfw_atl__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__dfw_atl.log" &
    p5=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume \
      --scenario-id iad_bos__20250115T0000Z --scenario-id iad_bos__20250115T1200Z --scenario-id iad_bos__20250415T0000Z --scenario-id iad_bos__20250415T1200Z --scenario-id iad_bos__20250715T0000Z --scenario-id iad_bos__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__iad_bos.log" &
    p6=$!
    wait "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit50k_20260413.json" \
      --resume > "$LOG_DIR/benchmark_xb1_custom_visual_limit50k_20260413__aggregate.log" 2>&1
  ) >"$log50" 2>&1 &
  local pid50=$!

  (
    cd "$ROOT"
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume \
      --scenario-id iah_mia__20250115T0000Z --scenario-id iah_mia__20250115T1200Z --scenario-id iah_mia__20250415T0000Z --scenario-id iah_mia__20250415T1200Z --scenario-id iah_mia__20250715T0000Z --scenario-id iah_mia__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__iah_mia.log" &
    p1=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume \
      --scenario-id lax_sfo__20250115T0000Z --scenario-id lax_sfo__20250115T1200Z --scenario-id lax_sfo__20250415T0000Z --scenario-id lax_sfo__20250415T1200Z --scenario-id lax_sfo__20250715T0000Z --scenario-id lax_sfo__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__lax_sfo.log" &
    p2=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume \
      --scenario-id msp_dtw__20250115T0000Z --scenario-id msp_dtw__20250115T1200Z --scenario-id msp_dtw__20250415T0000Z --scenario-id msp_dtw__20250415T1200Z --scenario-id msp_dtw__20250715T0000Z --scenario-id msp_dtw__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__msp_dtw.log" &
    p3=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume \
      --scenario-id den_slc__20250115T0000Z --scenario-id den_slc__20250115T1200Z --scenario-id den_slc__20250415T0000Z --scenario-id den_slc__20250415T1200Z --scenario-id den_slc__20250715T0000Z --scenario-id den_slc__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__den_slc.log" &
    p4=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume \
      --scenario-id dfw_atl__20250115T0000Z --scenario-id dfw_atl__20250115T1200Z --scenario-id dfw_atl__20250415T0000Z --scenario-id dfw_atl__20250415T1200Z --scenario-id dfw_atl__20250715T0000Z --scenario-id dfw_atl__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__dfw_atl.log" &
    p5=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume \
      --scenario-id iad_bos__20250115T0000Z --scenario-id iad_bos__20250115T1200Z --scenario-id iad_bos__20250415T0000Z --scenario-id iad_bos__20250415T1200Z --scenario-id iad_bos__20250715T0000Z --scenario-id iad_bos__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__iad_bos.log" &
    p6=$!
    wait "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit100k_20260413.json" \
      --resume > "$LOG_DIR/benchmark_xb1_custom_visual_limit100k_20260413__aggregate.log" 2>&1
  ) >"$log100" 2>&1 &
  local pid100=$!

  wait "$pid50" "$pid100"

  (
    cd "$ROOT"
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume \
      --scenario-id iah_mia__20250115T0000Z --scenario-id iah_mia__20250115T1200Z --scenario-id iah_mia__20250415T0000Z --scenario-id iah_mia__20250415T1200Z --scenario-id iah_mia__20250715T0000Z --scenario-id iah_mia__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__iah_mia.log" &
    p1=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume \
      --scenario-id lax_sfo__20250115T0000Z --scenario-id lax_sfo__20250115T1200Z --scenario-id lax_sfo__20250415T0000Z --scenario-id lax_sfo__20250415T1200Z --scenario-id lax_sfo__20250715T0000Z --scenario-id lax_sfo__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__lax_sfo.log" &
    p2=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume \
      --scenario-id msp_dtw__20250115T0000Z --scenario-id msp_dtw__20250115T1200Z --scenario-id msp_dtw__20250415T0000Z --scenario-id msp_dtw__20250415T1200Z --scenario-id msp_dtw__20250715T0000Z --scenario-id msp_dtw__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__msp_dtw.log" &
    p3=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume \
      --scenario-id den_slc__20250115T0000Z --scenario-id den_slc__20250115T1200Z --scenario-id den_slc__20250415T0000Z --scenario-id den_slc__20250415T1200Z --scenario-id den_slc__20250715T0000Z --scenario-id den_slc__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__den_slc.log" &
    p4=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume \
      --scenario-id dfw_atl__20250115T0000Z --scenario-id dfw_atl__20250115T1200Z --scenario-id dfw_atl__20250415T0000Z --scenario-id dfw_atl__20250415T1200Z --scenario-id dfw_atl__20250715T0000Z --scenario-id dfw_atl__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__dfw_atl.log" &
    p5=$!
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume \
      --scenario-id iad_bos__20250115T0000Z --scenario-id iad_bos__20250115T1200Z --scenario-id iad_bos__20250415T0000Z --scenario-id iad_bos__20250415T1200Z --scenario-id iad_bos__20250715T0000Z --scenario-id iad_bos__20250715T1200Z \
      2>&1 | tee "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__iad_bos.log" &
    p6=$!
    wait "$p1" "$p2" "$p3" "$p4" "$p5" "$p6"
    "$ROOT/.venv/bin/python3" -u -m mach_cutoff.benchmark.cli \
      --benchmark-config "$ROOT/results/configs/benchmark_xb1_custom_visual_limit300k_20260413.json" \
      --resume > "$LOG_DIR/benchmark_xb1_custom_visual_limit300k_20260413__aggregate.log" 2>&1
  ) >"$log300" 2>&1 &
  wait "$pid300"
}

main "$@"
