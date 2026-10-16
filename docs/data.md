# Data contract

Keep data you are authorized to use in `data/`, which is ignored by Git. Generated run outputs go in the ignored `artifacts/` folder. No patient recordings are bundled with this repository.

## Waveform CSV

Each row is one uniformly sampled observation. Segmentation and the inspiration boundary are supplied by an upstream adapter. Rows for each breath must already be ordered by increasing time.

| Column | Meaning |
|---|---|
| `patient` | Stable patient/group identifier, read as a string |
| `breath_id` | Breath identifier, unique within a patient |
| `time_s` | Time in seconds; evenly spaced, strictly increasing within a breath |
| `flow_l_min` | Flow in L/min, positive or zero during inspiration, negative or zero during expiration |
| `pressure_cm_h2o` | Airway pressure in cm H2O |
| `inspiration` | 1 during inspiration, then 0 during expiration |

Example with a complete small breath:

```csv
patient,breath_id,time_s,flow_l_min,pressure_cm_h2o,inspiration
synthetic-001,1,0.00,30,10,1
synthetic-001,1,0.02,30,12,1
synthetic-001,1,0.04,30,14,1
synthetic-001,1,0.06,-20,7,0
synthetic-001,1,0.08,-20,6,0
synthetic-001,1,0.10,-20,5,0
```

The extractor requires at least two samples in each phase, a single inspiration-to-expiration transition, finite measurements, and a positive inhaled volume. It rejects irregular timestamps and incompatible flow signs so the adapter can correct them explicitly. More complex waveforms need a suitable adapter before using this feature contract.

The demo uses the original experiment's 0.02-second interval (50 Hz). The public extractor infers the interval from the supplied timestamps and validates it for each breath.

## Sixteen numeric features

| Feature | Definition / unit |
|---|---|
| `i_time` | Inspiration sample count × interval, s |
| `e_time` | Expiration sample count × interval, s |
| `i_e_ratio` | Inspiration duration / expiration duration |
| `rr` | 60 / total breath duration, breaths/min |
| `tvi` | Simpson integral of inspiratory flow × 1000/60, mL |
| `tve` | Negative Simpson integral of expiratory flow × 1000/60, mL |
| `tve_tvi_ratio` | Exhaled / inhaled volume |
| `max_flow`, `min_flow` | Whole-breath extrema, L/min |
| `max_pressure`, `min_pressure` | Whole-breath extrema, cm H2O |
| `pip` | Maximum inspiratory pressure, cm H2O |
| `maw` | Mean inspiratory pressure, cm H2O; retained experiment convention |
| `peep` | Median of the last up to five expiratory pressure samples, cm H2O |
| `ip_auc`, `ep_auc` | Phase-specific pressure integrals, cm H2O·s |

Durations cover the sample bins, while the Simpson integrals span the first to last sample timestamp of each phase. For example, three constant 30 L/min inspiratory samples at 0.02-second spacing give a 0.06-second phase duration and a 20 mL sampled-span integral.

In the original notebook, eight existing features came from a supplied table, including exhaled volume, PEEP, and pressure areas. The public raw-waveform path computes all 16 using the definitions above. Its PEEP estimate and phase-area conventions may differ from the supplied table. For an existing feature table, use `evaluate` directly after confirming its units and definitions.

## Label CSV

One row per `(patient, breath_id)`:

```csv
patient,breath_id,bsa,dta
synthetic-001,1,0,0
synthetic-001,2,1,0
synthetic-001,3,0,1
```

The mapping is `target = bsa + 2*dta`. Both flags must be binary. Dual-positive rows are counted and excluded from this three-class task. Nonfinite feature rows are also counted and excluded. The retained keys and targets stay aligned through the same filtering mask.

Duplicate keys, missing identifiers, or unequal feature/label key coverage stop the run. Labels are never aligned by file row order. Patient and breath identifiers are excluded from the model feature matrix.
