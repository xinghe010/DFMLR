# Calculation code for the examples in the manuscript

This package contains the constrained quadratic-programming (CQP) calculation
code used for the numerical examples in the manuscript.  The scripts reproduce
the reported distance-based diagnostic tables.

## Folders

- `example1/`: financial-distress example, including the constrained model M2
  branch enumeration, the signed-input constrained model M3 calculation, and the
  Table 2 output.
- `air_quality/`: air-quality data aggregation, constrained model M2/M3
  calculations, and the Table 6 output.

## Run

```bash
cd example1
python run_example1_constrained_cqp.py

cd ../air_quality
python run_air_quality_constrained_cqp.py
```

Dependencies are listed in each folder's `requirements.txt`.
