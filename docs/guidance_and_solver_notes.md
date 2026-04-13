# Guidance And Solver Notes

This note records the current guidance and optimization setup used for the Gulf-focused `IAH -> MIA` experiments in [tmp/gulf_route_configs/benchmark_iah_mia.json](/Users/liambarkley/Documents/Projects/Mach%20Cutoff/Mach%20Cutoff%20Simulation/tmp/gulf_route_configs/benchmark_iah_mia.json).

Reusable saved presets for future work:

- [examples/config_xb1_future_trials.json](/Users/liambarkley/Documents/Projects/Mach%20Cutoff/Mach%20Cutoff%20Simulation/examples/config_xb1_future_trials.json)
- [examples/guidance_xb1_population_trials.json](/Users/liambarkley/Documents/Projects/Mach%20Cutoff/Mach%20Cutoff%20Simulation/examples/guidance_xb1_population_trials.json)

XB-1 aircraft baseline used in the saved preset:

- `Mach 1.18`
- `36,514 ft` = `11,129.6672 m`

Source attribution:

- Boom official XB-1 page, which states XB-1 reached `Mach 1.18` and `36,514 feet` on February 10, 2025: [Boom Supersonic XB-1](https://boomsupersonic.com/xb-1)
- Boom fact sheet for airframe reference data such as `3 GE J85-15 engines`, `62.6 ft` length, and `21 ft` wingspan: [Boom Fact Sheet PDF](https://boomsupersonic.com/wp-content/uploads/2024/12/Boom-Fact-Sheet-1.pdf)

## Current Gulf-Tuned Parameters

### Route Optimization Objective

These are the active `population_avoidant` route objective weights from the Gulf manifest:

- `weight_population`: `4.0`
- `weight_populated_ground_hits`: `12.0`
- `weight_total_ground_hits`: `0.02`
- `weight_overflight_population`: `0.0`
- `weight_overflight_area`: `0.0`
- `weight_route_stretch`: `2.8`
- `weight_route_heading_change`: `1.6`
- `boom_exposure_limit_people`: `300000.0`
- `weight_boom_exposure_limit`: `8.0`
- `min_cutoff_emission_fraction`: `0.05`
- `weight_cutoff_shortfall`: `0.15`
- `weight_speed`: `1.4`
- `weight_fuel`: `0.0`
- `unpopulated_speed_weight_bonus`: `4.5`

### Guidance Speed Targets

- `target_mach`: `1.28`
- `effective_mach_target`: `1.20`

### Exposure Corridor

- `enabled`: `true`
- `min_lookahead_m`: `25000`
- `max_lookahead_m`: `180000`
- `dense_brake_lookahead_m`: `45000`
- `sample_spacing_m`: `10000`
- `corridor_half_width_km`: `12`
- `sparse_density_people_per_km2`: `0.75`
- `dense_density_people_per_km2`: `20`
- `sparse_mach_bias`: `0.28`
- `dense_mach_bias`: `0.12`
- `sparse_ground_risk_scale`: `0.01`
- `dense_ground_risk_scale`: `2.0`
- `bias_smoothing`: `0.65`
- `terminal_scale`: `0.5`
- `takeoff_scale`: `0.3`

### Mode Switching

- `terminal_distance_m`: `100000`

### Boom Feedback

- `enable_ground_hit_feedback`: `true`
- `ground_hit_fraction_threshold`: `0.4`
- `altitude_gain_m_per_hit_fraction`: `1500`
- `altitude_bias_decay_m_per_step`: `90`
- `max_altitude_bias_m`: `1800`
- `mach_reduction_per_hit_fraction`: `0.008`
- `source_cutoff_recovery_mach_step`: `0.015`

### Boom-Aware Online Optimizer

- `candidate_altitude_offsets_m`: `[-1200, -600, -300, 0, 500, 1000, 1600, 2200]`
- `candidate_mach_offsets`: `[-0.18, -0.12, -0.08, -0.04, 0.0, 0.06, 0.12, 0.18, 0.24]`
- `effective_mach_margin`: `1.0`
- `ground_risk_mach_gain`: `0.6`
- `ground_risk_altitude_relief_per_km`: `0.14`
- `weight_ground_risk`: `2.4`
- `weight_cutoff_risk`: `0.2`
- `weight_tracking`: `0.7`
- `weight_altitude_deviation`: `0.08`
- `weight_mach_deviation`: `0.008`
- `weight_terminal_altitude_bias`: `0.08`
- `max_altitude_adjustment_m`: `2600`
- `max_altitude_step_m_per_cycle`: `700`
- `max_mach_adjustment`: `0.3`
- `max_mach_step_per_cycle`: `0.12`

## How Guidance Works

The live controller is in [mach_cutoff/flight/guidance.py](/Users/liambarkley/Documents/Projects/Mach%20Cutoff/Mach%20Cutoff%20Simulation/mach_cutoff/flight/guidance.py).

At each guidance step it does four things:

1. It projects the aircraft onto the current flight path and decides which mode it is in: `takeoff_climb`, `enroute`, `terminal`, or `abort_failsafe`.
2. It computes a baseline desired Mach from the configured speed target, plus a correction that tries to keep effective Mach near the desired effective Mach.
3. It applies a forward-looking population corridor term:
   - the controller samples population density ahead of the aircraft along the route
   - low-density near-field corridor increases Mach
   - dense near-term shoreline corridor decreases Mach
   - low-density corridor also reduces the online boom-risk cost scale
4. It runs a small online optimizer over candidate altitude and Mach offsets and chooses the lowest-cost pair using predicted ground-risk, cutoff-risk, tracking cost, altitude deviation, and Mach deviation.

### Why The Corridor Term Matters

The corridor sampler is deliberately asymmetric:

- `near_density_people_per_km2` drives the sparse-water speed-up
- `brake_density_people_per_km2` is computed only over the shorter `dense_brake_lookahead_m` window

That means dense population far ahead should not suppress speed in the middle of open water. The aircraft can accelerate over the Gulf, then start braking only when the nearer shoreline window becomes dense.

## How The Route Solver Works

The route solver is in [mach_cutoff/simulation/route_optimizer.py](/Users/liambarkley/Documents/Projects/Mach%20Cutoff/Mach%20Cutoff%20Simulation/mach_cutoff/simulation/route_optimizer.py).

It is now a three-stage search:

1. Low fidelity:
   - broad candidate search
   - coarse atmosphere/shock/grid/ray settings
   - cheap enough to explore waypoint mutations
2. Mid fidelity:
   - evaluates the best low-fidelity semifinalists
   - reduces surrogate mismatch before final promotion
3. Full fidelity:
   - evaluates the best mid-fidelity finalists
   - chooses the final route

### What Gets Mutated

For each candidate, the solver mutates:

- interior waypoint lateral position
- interior waypoint altitude
- target Mach
- effective Mach target

The optimizer then scores each candidate using:

- exposed population
- populated hit count
- total hit count
- speed proxy
- cutoff shortfall
- exposure-limit violation
- route stretch
- accumulated heading change

The added `weight_route_stretch` and `weight_route_heading_change` terms are what stop the solver from making unnecessary offshore zig-zags that do not materially improve the objective.

## Current Default Multi-Fidelity Budget

The benchmark defaults in [examples/config_benchmark_base.json](/Users/liambarkley/Documents/Projects/Mach%20Cutoff/Mach%20Cutoff%20Simulation/examples/config_benchmark_base.json) are now:

- `max_wall_time_s`: `300`
- `reserve_time_for_mid_fidelity_s`: `90`
- `reserve_time_for_full_fidelity_s`: `100`
- `batch_size`: `4`
- `semifinalists`: `6`
- `finalists`: `4`

Low-fidelity scales:

- `low_fidelity_emission_interval_scale`: `2.0`
- `low_fidelity_rays_scale`: `0.35`
- `low_fidelity_grid_scale`: `0.55`
- `low_fidelity_step_scale`: `1.8`
- `low_fidelity_max_steps_scale`: `0.45`
- `low_fidelity_max_emissions`: `96`

Mid-fidelity scales:

- `mid_fidelity_emission_interval_scale`: `1.35`
- `mid_fidelity_rays_scale`: `0.65`
- `mid_fidelity_grid_scale`: `0.8`
- `mid_fidelity_step_scale`: `1.25`
- `mid_fidelity_max_steps_scale`: `0.75`
- `mid_fidelity_max_emissions`: `160`

## Practical Interpretation

The present system is not a convex guidance law. It is a hybrid:

- local online feedback controller for heading, altitude, and Mach
- forward-looking population corridor bias for proactive speed management
- iterative stochastic waypoint optimizer for route generation

The route optimizer chooses the path. The guidance controller then flies that path while still adapting Mach and altitude in real time based on predicted boom risk and future corridor density.
