# Air-quality constrained CQP calculation

This folder contains the calculation code for the air-quality example in the
manuscript.  It aggregates hourly records into GMT-day city-level fuzzy
observations, solves the nonnegative-spread constrained CQP rows for models
`M2` and `M3`, and writes the Table 6 output.

## Main script

```bash
python run_air_quality_constrained_cqp.py
```

The default input file is:

```text
data/air_quality_2024.csv
```

Another CSV file can be supplied with:

```bash
python run_air_quality_constrained_cqp.py --csv path/to/air_quality.csv
```

## Outputs

The script writes to `outputs/`:

- `daily_fuzzy_observations.csv`
- `air_quality_constrained_coefficients.csv`
- `air_quality_constrained_metrics.csv`
- `air_quality_table6_revised_constrained.csv`

The CSV contains 52,704 hourly records and is aggregated into 2,196 GMT-day
city-level fuzzy observations.  The table output contains the four
fuzzy-distance diagnostics used in the manuscript: `S_DYK2`, `S_DDK2`,
`R_DYK2`, and `R_DDK2`.  The two normal-fuzzy-number model `M3` rows are
combined in Table 6 because their displayed diagnostic values coincide at the
reported precision.
