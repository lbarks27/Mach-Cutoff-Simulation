# Mach Cutoff Simulation (Experimental)

This project provides a modular, research-oriented Python simulation for supersonic sonic-boom ray propagation using time-varying HRRR atmospheric data.

## What it does

1. Loads a flight path from waypoint JSON (`lat/lon/alt/time`).
2. Pulls HRRR pressure-level data along the path and timeline from NOAA AWS open data.
3. Converts atmospheric state (`pressure`, `temperature`, `humidity`, `wind`) to an **acoustic effective refractive index** field.
4. Simulates a supersonic point-mass aircraft and emits shock rays (wavefront-normal approximation).
5. Ray-traces propagation in a WGS84-aware setup and records ground-intersection footprints.
6. Optionally overlays population data to estimate:
   - local population around each ground-hit point
   - total population in traced ground-impact envelopes between hits
7. Produces multiple visualization options:
   - Matplotlib (terrain-aware static + GIF animation)
   - Plotly (terrain-aware interactive HTML)
     - Includes 3D atmospheric overlay (`plotly_atmosphere_3d.html`) with measure toggles and wind vectors
   - PyVista (terrain-aware 3D screenshots + GIF animation)
   - Atmospheric diagnostics (time-series + vertical profile) in Matplotlib and Plotly

## Package layout

- `mach_cutoff/config.py`: configurable experiment model
- `mach_cutoff/guidance_config.py`: dedicated guidance config model/loader
- `mach_cutoff/core/constants.py`: physical constants
- `mach_cutoff/core/geodesy.py`: WGS84 transforms
- `mach_cutoff/core/raytrace.py`: adaptive ray integrator
- `mach_cutoff/flight/waypoints.py`: waypoint ingestion and interpolation
- `mach_cutoff/flight/aircraft.py`: point-mass aircraft + shock ray generation
- `mach_cutoff/flight/guidance.py`: waypoint guidance + wind-aware dynamics
- `mach_cutoff/atmosphere/hrrr.py`: HRRR file selection/download/load
- `mach_cutoff/atmosphere/interpolation.py`: HRRR interpolation
- `mach_cutoff/atmosphere/acoustics.py`: sound-speed + effective-index field
- `mach_cutoff/simulation/engine.py`: end-to-end pipeline
- `mach_cutoff/simulation/population.py`: population heatmap loading + exposure analysis
- `mach_cutoff/simulation/route_optimizer.py`: iterative rerun optimizer (low-fidelity search + full-fidelity finalists)
- `mach_cutoff/visualization/backends/*`: plotting backends
- `mach_cutoff/cli.py`: command line entrypoint

## Install

From this repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e '.[all]'
```

`cfgrib` requires ECMWF ecCodes installed on your system.

If you see `error: externally-managed-environment`, you are using system Python without a venv. Use the commands above (venv + activate) and retry.

If you use `zsh`, keep extras quoted (for example `'.[all]'`), otherwise zsh may treat brackets as globs.

If you want minimal install first:

```bash
python3 -m pip install -e '.[core]'
```

Then add optional groups as needed:

```bash
python3 -m pip install -e '.[hrrr]'
python3 -m pip install -e '.[viz]'
```

## Waypoint JSON format

Use a JSON array or object with `waypoints` key.

```json
{
  "waypoints": [
    {"lat": 34.1, "lon": -118.4, "alt_m": 12000, "time_utc": "2026-02-10T18:00:00Z"},
    {"lat": 35.6, "lon": -114.5, "alt_m": 12000, "time_utc": "2026-02-10T18:40:00Z"}
  ]
}
```

## Run

```bash
mach-cutoff \
  --waypoints examples/waypoints_example.json \
  --config examples/config_example.json \
  --guidance-config examples/guidance_example.json \
  --output-dir outputs
```

First-run note: full config can take time due HRRR download and 3 visualization backends.

Or without installing script:

```bash
python3 -m mach_cutoff.cli \
  --waypoints examples/waypoints_example.json \
  --config examples/config_example.json \
  --guidance-config examples/guidance_example.json \
  --output-dir outputs
```

Quick smoke run (validated locally on this machine):

```bash
python3 -m mach_cutoff.cli \
  --waypoints examples/waypoints_example.json \
  --config examples/config_smoke.json \
  --guidance-config examples/guidance_example.json \
  --output-dir outputs_smoke \
  --no-animation
```

### Useful flags

- `--skip-matplotlib`
- `--skip-plotly`
- `--skip-pyvista`
- `--no-animation`
- `--interactive` (opens live windows for matplotlib/pyvista and opens plotly 3D HTML in browser)
- `--map-style {topographic,road,satellite}` (switch map/basemap style across visualization outputs)
- `--guidance-config <path>` (load dedicated guidance dynamics/controller settings)
- `--optimize-routing` / `--no-optimize-routing` (override `route_optimization.enabled` from config)
- `--optimizer-*` flags now act as overrides for `route_optimization.*` config values

### Iterative route optimization

```bash
python3 -m mach_cutoff.cli \
  --waypoints examples/waypoints_msp_dtw.json \
  --config examples/config_population_example.json \
  --guidance-config examples/guidance_example.json \
  --output-dir outputs_opt
```

Behavior:

- Runs repeated low-fidelity sims with deterministic candidate generation.
- Candidate mutations are phase-aware (`takeoff_climb`, `enroute`, `terminal`) and keep baseline as a failsafe candidate.
- Objective combines population exposure, ground-reaching boom penalties, speed/time proxy, and optional fuel proxy.
- Can apply a soft boom-exposure limit (`boom_exposure_limit_people`) so cutoff-aware routes optimize for time while staying below a declared study threshold.
- Can apply a cutoff target (`min_cutoff_emission_fraction`) so cutoff-aware routes are penalized when too much of the route still produces ground-reaching boom.
- Promotes top low-fidelity candidates to full-fidelity sims and selects the best final route.
- Subsonic emissions are suppressed from ray generation (`effective_mach <= 1.0` or aircraft `mach <= 1.0`).

Optimization settings live in main config under `route_optimization`:

```json
"route_optimization": {
  "enabled": true,
  "max_wall_time_s": 420.0,
  "seed": 17,
  "batch_size": 4,
  "finalists": 3,
  "enable_fuel_objective": true,
  "weight_population": 1.0,
  "boom_exposure_limit_people": 100000.0,
  "weight_boom_exposure_limit": 8.0,
  "min_cutoff_emission_fraction": 0.75,
  "weight_cutoff_shortfall": 5.0,
  "weight_speed": 0.35,
  "weight_fuel": 0.2
}
```

Optimization artifacts are written under `outputs_opt/optimization/`:

- `optimized_waypoints.json`
- `optimization_report.json`
- `optimization_iteration_metrics.csv`
- `low_fidelity_candidates/*`
- `full_fidelity_finalists/*`

## Guidance config (phase 1)

Guidance is configured separately from the main experiment config to keep control-law iteration isolated.

- Example file: `examples/guidance_example.json`
- If `--guidance-config` is omitted, built-in defaults are used (guidance still enabled).
- Loader accepts either a direct object or `{ "guidance": { ... } }`
- Current mode set includes: `takeoff_climb`, `enroute`, `terminal`, `abort_failsafe`
- `boom_avoidance.optimizer` enables phase-2 predictive candidate search (Mach/altitude) with configurable risk/cost weights.
- Guidance feeds back:
  - effective Mach (`source_mach_cutoff`)
  - sonic-boom ground-hit fraction from the previous emission

## Outputs

- `outputs/simulation_summary.json`
- `outputs/simulation_hits.npz`
- `outputs/google_earth_overlay.kml` (open in Google Earth for true globe-map overlay)
- If optimization is enabled: `outputs/optimization/*` with candidate-by-candidate objective traces and route files
- Summary/NPZ include guidance diagnostics (mode counts, cross-track statistics, command channels, optimizer cost/risk predictions) when guidance is enabled.
- If `population.enabled=true`, summary/NPZ include population impact metrics and ranked hit exposure values.
  - Includes separate direct-overflight corridor metrics (`total_overflight_population`, `total_overflight_area_km2`)
- Visualization files depending on enabled backends
  - Plotly may include `plotly_atmosphere_3d.html` for gridded atmosphere overlays
  - Plotly includes `plotly_globe.html` for interactive globe-based inspection of flight, rays, and ground hits
  - Ground-footprint figures include population heatmap + estimated exposed area overlays when enabled
  - Includes atmospheric diagnostics when `visualization.include_atmosphere=true`

## Population impact analysis

Population analysis is controlled by `population.*` in config:

```json
"population": {
  "enabled": true,
  "dataset_path": null,
  "heatmap_cell_deg": 0.12,
  "hit_radius_km": 25.0,
  "trace_half_width_km": 12.0,
  "overflight_half_width_km": 20.0
}
```

- `dataset_path = null` uses bundled sample data: `mach_cutoff/data/us_metro_population_sample.csv`
- Supported dataset formats:
  - CSV with columns like `lat_deg`, `lon_deg`, `population`
  - NPZ point arrays `lat_deg`/`lon_deg`/`population` (or `lat`/`lon`/`pop`)
  - NPZ gridded arrays `lat_edges_deg`/`lon_edges_deg`/`population_grid`
- Traced footprints connect/close ground-hit sets and estimate exposed area/population over that envelope.
- Direct-overflight metrics use aircraft sampled track and `population.overflight_half_width_km`.

Example config: `examples/config_population_example.json`

### Google Earth overlay

Each run writes `google_earth_overlay.kml` in the output directory. This KML contains:

- Aircraft flight track (absolute altitude)
- Sampled shock-ray trajectories (absolute altitude)
- Ground-hit markers (if any)

Recommended viewer: Google Earth Pro (desktop) or Google Earth Web.  
Apple Maps does not provide a general KML import workflow for this type of 3D overlay.

## Research/experiment knobs

Primary tunables are in `config_example.json`:

- HRRR retrieval/caching: `hrrr.*`
- Field resolution and extents: `grid.*`
- Aircraft assumptions: `aircraft.*`
- Shock source density: `shock.*`
- Integrator controls: `raytrace.*`
- Runtime trial slicing: `runtime.*`
- Population impact analysis: `population.*`
- Route optimization controls: `route_optimization.*`
- Visualization outputs/toggles: `visualization.*` (including `visualization.include_atmosphere`, `visualization.map_style`)

## Benchmark pipeline

Run the research benchmark with:

```bash
mach-cutoff-benchmark --benchmark-config examples/benchmark_us_archetypes.json
```

The bundled benchmark manifest models the study as three route classes:

- `fastest`: time-first supersonic baseline
- `population_avoidant`: penalizes direct overflight of dense population
- `cutoff_optimized`: penalizes predicted boom exposure and rewards higher route-level cutoff fraction

The benchmark manifest also declares research-objective thresholds used in the aggregate feasibility assessment:

- `boom_exposure_limit_people`
- `max_time_penalty_pct`
- `min_cutoff_emission_fraction`
- `max_abort_samples`
- `max_distance_to_destination_m`

Benchmark manifests may also define `"extends": "<base-manifest>.json"` to inherit and override a shared JSON manifest. Relative paths are resolved from the child manifest location.

Optional flags:

- `--output-root <dir>`
- `--jobs <int>`

Aggregate benchmark outputs include:

- `aggregate/run_metrics.csv`
- `aggregate/route_efficiency_summary.csv`
- `aggregate/feasibility_summary.json`
- `aggregate/benchmark_report.md`
- `aggregate/exposure_maps/*.png` for retained anchor scenarios
- `--resume`
- `--force`
- `--skip-sensitivity`
- `--scenario-id <id>` (repeatable)
- `--route-class <name>` (repeatable)

Benchmark outputs:

- `benchmarks/<name>/core/<scenario_id>/<route_class>/...`
- `benchmarks/<name>/sensitivity/<scenario_id>/<route_class>/<profile_id>/...`
- `benchmarks/<name>/aggregate/run_metrics.csv`
- `benchmarks/<name>/aggregate/paired_statistics.{csv,json}`
- `benchmarks/<name>/aggregate/dominance_matrix.csv`
- `benchmarks/<name>/aggregate/dominance_summary.json`
- `benchmarks/<name>/aggregate/robustness_summary.json`
- `benchmarks/<name>/aggregate/benchmark_report.md`
- `benchmarks/<name>/aggregate/plots/*`

### GPW full population data

The benchmark pipeline can auto-prepare GPWv4 r11 data into a cached NPZ grid:

- checks prepared cache first
- then checks raw cache
- downloads raw data when missing (if `gpw.auto_download=true`)
- preprocesses to `lat_edges_deg/lon_edges_deg/population_grid`

Auth lookup order for download:

1. `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD`
2. `~/.netrc` (`urs.earthdata.nasa.gov`)

Shock direction reference toggle:

- `shock.direction_reference = "earth_down"`: earth-relative cone (axis aligned with local vertical down)
- `shock.direction_reference = "aircraft_forward"`: aircraft-relative forward cone (ahead of aircraft, recommended default)
- `shock.direction_reference = "aircraft_aft"`: aircraft-relative aft cone (behind aircraft)

For quick trials:

- Lower `grid.nx`, `grid.ny`, `grid.nz`
- Increase `shock.emission_interval_s`
- Lower `shock.rays_per_emission`
- Lower `raytrace.max_steps`

For higher fidelity:

- Increase grid resolution
- Decrease `raytrace.ds_m`
- Increase ray count and reduce emission interval

## HRRR notes

This code targets pressure-level files:

- Product: `wrfprsf`
- Bucket key template:
  `hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfprsf00.grib2`

Default behavior uses hourly analysis snapshots nearest to each emission time.

## Modeling assumptions (current)

- Aircraft is a point-mass with 3D waypoint guidance and simple wind-aware kinematics.
- Shock rays are generated as cone-distributed wavefront normals.
- Effective-index field is scalar and uses a configurable wind projection approximation.
- Atmosphere sampling is nearest-horizontal + linear-vertical interpolation.

These are modular and designed for extension to future guidance/control and repeated multi-trial sweeps.

## Parameter sweeps

For repeated trials with changed parameters, use `mach_cutoff.simulation.run_parameter_sweep`.

Input:

- waypoint JSON path
- base `ExperimentConfig`
- list of override dictionaries

Output:

- `trial_###/simulation_summary.json`
- `trial_###/simulation_hits.npz`
- `sweep_manifest.json`
