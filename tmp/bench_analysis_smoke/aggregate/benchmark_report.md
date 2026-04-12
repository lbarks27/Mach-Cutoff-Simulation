# Benchmark Report

## Dataset and benchmark manifest
- Total runs: 3
- Core runs: 3

## Core scenario inventory
- Route classes: cutoff_optimized, fastest, population_avoidant
- Unique scenarios: 1

## Per-class metric medians and ranges
- `cutoff_optimized`: boom median=10.0, overflight median=50.0, elapsed median=140.0s
- `fastest`: boom median=100.0, overflight median=50.0, elapsed median=100.0s
- `population_avoidant`: boom median=100.0, overflight median=20.0, elapsed median=120.0s

## Pairwise findings
- population_avoidant_vs_fastest / elapsed_time_s: median_delta=20.000, p=0.0455, wins=0, losses=1
- population_avoidant_vs_fastest / time_proxy_s: median_delta=20.000, p=0.0455, wins=0, losses=1
- population_avoidant_vs_fastest / route_distance_km: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / fuel_proxy: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / ground_hit_count: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / populated_ground_hit_count: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / boom_exposed_population: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / boom_exposed_area_km2: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / overflight_population: median_delta=-30.000, p=0.0455, wins=1, losses=0
- population_avoidant_vs_fastest / overflight_area_km2: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / cutoff_emission_fraction: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / distance_to_destination_m: median_delta=0.000, p=1, wins=0, losses=0
- population_avoidant_vs_fastest / abort_samples: median_delta=0.000, p=1, wins=0, losses=0
- cutoff_optimized_vs_fastest / elapsed_time_s: median_delta=40.000, p=0.0455, wins=0, losses=1
- cutoff_optimized_vs_fastest / time_proxy_s: median_delta=40.000, p=0.0455, wins=0, losses=1
- cutoff_optimized_vs_fastest / route_distance_km: median_delta=0.000, p=1, wins=0, losses=0
- cutoff_optimized_vs_fastest / fuel_proxy: median_delta=0.000, p=1, wins=0, losses=0
- cutoff_optimized_vs_fastest / ground_hit_count: median_delta=0.000, p=1, wins=0, losses=0

## Dominance findings

## Sensitivity robustness findings
- Dominance retention rate: 0.000
- Primary-metric best rate `cutoff_optimized`: 0.000
- Primary-metric best rate `fastest`: 0.000
- Primary-metric best rate `population_avoidant`: 0.000

## Failures / incomplete runs
- See run directories for `run_complete.json` status and any captured error traces.

## Threats to validity
- Optimization stochasticity may influence route ranking; compare across seeds for stronger confidence.
- Atmospheric sampling at selected timestamps may not capture full climatological variability.
- Population metrics are model-based proxies, not direct measured exposure outcomes.
