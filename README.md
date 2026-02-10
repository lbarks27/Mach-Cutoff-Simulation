# Mach Cutoff Simulation (Experimental)

This project provides a modular, research-oriented Python simulation for supersonic sonic-boom ray propagation using time-varying HRRR atmospheric data.

## What it does

1. Loads a flight path from waypoint JSON (`lat/lon/alt/time`).
2. Pulls HRRR pressure-level data along the path and timeline from NOAA AWS open data.
3. Converts atmospheric state (`pressure`, `temperature`, `humidity`, `wind`) to an **acoustic effective refractive index** field.
4. Simulates a supersonic point-mass aircraft and emits shock rays (wavefront-normal approximation).
5. Ray-traces propagation in a WGS84-aware setup and records ground-intersection footprints.
6. Produces multiple visualization options:
   - Matplotlib (static + GIF animation)
   - Plotly (interactive HTML)
   - PyVista (3D screenshots + GIF animation)

## Package layout

- `mach_cutoff/config.py`: configurable experiment model
- `mach_cutoff/waypoints.py`: waypoint ingestion and interpolation
- `mach_cutoff/geodesy.py`: WGS84 transforms
- `mach_cutoff/aircraft.py`: point-mass aircraft + shock ray generation
- `mach_cutoff/atmosphere/hrrr.py`: HRRR file selection/download/load
- `mach_cutoff/atmosphere/interpolation.py`: HRRR interpolation
- `mach_cutoff/atmosphere/acoustics.py`: sound-speed + effective-index field
- `mach_cutoff/raytrace.py`: adaptive ray integrator
- `mach_cutoff/simulation/engine.py`: end-to-end pipeline
- `mach_cutoff/visualization/*`: plotting backends
- `mach_cutoff/cli.py`: command line entrypoint

## Install

From this repo:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[all]
```

`cfgrib` requires ECMWF ecCodes installed on your system.

If you want minimal install first:

```bash
pip install -e .[core]
```

Then add optional groups as needed:

```bash
pip install -e .[hrrr]
pip install -e .[viz]
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
  --output-dir outputs
```

Or without installing script:

```bash
python -m mach_cutoff.cli \
  --waypoints examples/waypoints_example.json \
  --config examples/config_example.json \
  --output-dir outputs
```

### Useful flags

- `--skip-matplotlib`
- `--skip-plotly`
- `--skip-pyvista`
- `--no-animation`

## Outputs

- `outputs/simulation_summary.json`
- `outputs/simulation_hits.npz`
- Visualization files depending on enabled backends

## Research/experiment knobs

Primary tunables are in `config_example.json`:

- HRRR retrieval/caching: `hrrr.*`
- Field resolution and extents: `grid.*`
- Aircraft assumptions: `aircraft.*`
- Shock source density: `shock.*`
- Integrator controls: `raytrace.*`
- Runtime trial slicing: `runtime.*`

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

- Aircraft is point-mass with constant Mach and constant altitude.
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
