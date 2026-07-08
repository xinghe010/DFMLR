Datasets for all_examples_constrained_cqp_code

1) financial_distress_example_dataset.csv
   Dataset used by the financial-distress example. It contains the crisp class label,
   triangular fuzzy inputs in vertex form, modal-spread form used in the code, and
   the fuzzy distress response used in the CQP calculations.

2) air_quality_2024_hourly_dataset.csv
   Raw GMT hourly air-quality dataset read by the air_quality script. The code uses
   CO, NO2, SO2, O3, PM2.5, PM10 as predictors and AQI as the response; CO2 is
   retained in the raw file but not used in the regression calculations.

3) air_quality_daily_fuzzy_dataset.csv
   Processed GMT-day city-level fuzzy dataset derived from the raw hourly file.
   For each retained variable it includes daily mean, left spread (mean-min),
   right spread (max-mean), normal spread (sample standard deviation), min, max,
   and std. These columns correspond to the triangular and normal fuzzy quantities
   used in the model calculations.

4) air_quality_daily_fuzzy_dataset_code_columns.csv
   Compact processed daily dataset with only the mean, left-spread, right-spread,
   and normal-spread columns used by the CQP arrays.
