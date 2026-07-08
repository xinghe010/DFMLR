# Financial-distress example constrained CQP calculation

This folder contains the calculation code for the financial-distress example in
the manuscript.  The code implements the nonnegative-spread constrained
estimators used for Table 2.

## Main script

```bash
python run_example1_constrained_cqp.py
```

## What the script does

- Converts the displayed triangular vertex inputs
  `<lower endpoint, modal value, upper endpoint>_T` to modal-spread form.
- Uses the fuzzy distress-score coding stated in the manuscript:
  category `0` is `(0.98, 0.08, 0.01)_T` and category `1` is
  `(0.02, 0.01, 0.08)_T`.
- Solves model `M2` by fixed slope-sign branch enumeration.
- Solves model `M3` using the signed-input constrained CQP, so negative crisp
  financial ratios exchange left and right spread contributions correctly.
- Writes the manuscript Table 2 values with the two visually identical model
  `M3` diagnostic rows combined into one row.

## Outputs

The script writes to `outputs/`:

- `example1_constrained_m2_coefficients.csv`
- `example1_constrained_m2_metrics.csv`
- `example1_constrained_m3_coefficients.csv`
- `example1_constrained_m3_metrics.csv`
- `example1_table2_revised_constrained.csv`

The table output contains the four distance-based diagnostics used in the
manuscript: `S_DYK2`, `S_DDK2`, `R_DYK2`, and `R_DDK2`.
