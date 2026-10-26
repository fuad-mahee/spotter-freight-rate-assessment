# Freight-rate prediction | Spotter assessment

CatBoost regression predicts total load rates for 12,000 November-December 2025 loads. **September–October holdout MAE: $121.83, a 50.5% reduction versus the distance × quote baseline.** This is a local retrospective result, not Spotter's hidden final score. Systematic underprediction and untested unseen-city performance are the main limitations.

## Results and artifacts

- [Assessment report with the required December chart](output/assessment_report.pdf)
- [Final predictions: 12,000 loads](output/validation_predictions.csv)
- [Completed December predictions](output/december_predictions.csv)
- [Public GitHub repository](https://github.com/fuad-mahee/spotter-freight-rate-assessment)
- [Original challenge instructions](provided_readme.md) and [assessment brief](freight-rate-ml-assessment-1-.pdf)

## Approach and validation

The labeled data contains 48,000 January-October 2025 loads. The unlabeled final set covers the next two months, so chronological validation matches the task.

| Stage | Fit | Evaluate |
| --- | --- | --- |
| Candidate selection | January-June: 28,806 rows | July-August: 9,671 rows |
| Local holdout | January-August: 38,477 rows | September-October: 9,523 rows |
| Final model | January-October: 48,000 rows | November-December: 12,000 hidden targets |

CatBoost suits mixed numeric and categorical data and learns nonlinear relationships and interactions among distance, equipment, location and market inputs without manually one-hot encoding all lanes. It supports numeric missing values and unseen categories; that support does **not** establish accuracy on new cities. All candidates use 650 trees, depth 6, learning rate 0.055, four CPU threads and seed 42. MAE was the selection criterion; no official accuracy metric was specified. No holdout-based early stopping or calibration was used.

Non-positive weights become missing with explicit flags; CatBoost handles numerical NaNs. All positive target outliers remain in training and headline evaluation. IDs and targets are excluded from features. Feature transformations are stateless, and categorical statistics are learned within each fit. The final-set features are audited but are not used to train or select models.

The main model excludes `quote_signal`: that candidate performed better on the designated selection period. The ablation does not prove leakage or establish the signal's upstream provenance.

## Results and decisions

| Model | July-August MAE | September-October MAE |
| --- | ---: | ---: |
| Distance × quote baseline | $367.29 | $246.02 |
| Selected main model: no quote, total-dollar target | $140.48 | $121.83 |
| Selected December model: six inputs, rate-per-mile target | $159.47 | $106.34 |

Main-model holdout RMSE is **$638.85**, WAPE **5.09%**, and R² **0.8247**. The core model's holdout MAE is 12.7% lower. This reversal shows that rankings vary across time. We preserved the main model chosen on July-August rather than reselecting after seeing the holdout. The models differ in both inputs and target formulation, so this comparison does not isolate the effect of removing features. Earlier rolling chronological windows and a feature/target factorial comparison are future work, not completed experiments.

### Residuals: tail risk and broad underprediction

Signed error means prediction minus actual. These are post-evaluation diagnostics from saved holdout predictions, not additional tuning data.

| Diagnostic | Main-model holdout |
| --- | ---: |
| Median absolute error | $42.49 |
| 95th-percentile absolute error | $238.01 |
| Mean signed error | -$90.93 |
| Median signed error | -$37.83 |
| Loads underpredicted | 80.6% |
| Squared error from worst 96 rows (about 1%) | 96.5% |

Rare extreme misses dominate RMSE, while negative median error and widespread underprediction also show bias among ordinary loads. September/October mean signed errors are -$88.02/-$93.72. Across distance bands below 500, 500-<1,000, 1,000-<2,000 and at least 2,000 miles, mean signed errors are -$26.80, -$51.88, -$130.17 and -$179.05; underprediction rates are 67.0%, 81.5%, 85.2% and 88.8%. These intervals include their lower bounds and exclude their upper bounds, accounting for fractional distances. See the [full month/distance table](output/residual_slices.csv).

If used for quotes or budgets, these estimates could understate transport costs, especially on longer trips. These diagnostics do not establish the cause. We did **not** add $90.93 or otherwise adjust predictions using this holdout. Any correction must be estimated and validated on development-only chronological folds and assessed on a separate future evaluation period.

### Geographic coverage gap

The final set has **1,447 unseen-city loads (12.1%)**, relative to all labeled training data. The September-October holdout has **zero**, relative to January-August training. This holdout does not measure new-city deployment accuracy. A proposed stress test would select cities before inspecting errors, exclude all loads touching them from an earlier fit window, then evaluate future loads involving them separately from seen-city loads. This test has not been run.

### December model and loss formulation

The chart supplies only pickup, delivery, distance, equipment, weight and date. A separately evaluated core model uses those six inputs and derived features; no market forecasts or quote values are fabricated.

The core rate-per-mile candidate fits `posted_rate / distance` with MAE and multiplies predictions by distance for total-dollar evaluation. Its training loss is proportional to `abs(total_rate_error) / distance`: a dollar of error on a shorter trip receives more weight. It is not merely a change of units. We tried this to model distance-normalized pricing; it beat the core total-rate candidate on July-August dollar MAE ($159.47 versus $163.01). Distance-weighted per-mile training would align its loss with total-dollar MAE; this comparison is future work. The December curve is a conditional prediction, with no demonstrated December seasonal accuracy from January-October history.

## Reproduce

Use Python 3.12 from this directory. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-solution.txt -c requirements-lock.txt
python solution.py
python score.py --predictions output/validation_predictions.csv --december-predictions output/december_predictions.csv --output-dir output
python -m unittest -v test_solution.py
python diagnostics.py
python build_report.py
```

On macOS/Linux activate with `source .venv/bin/activate`; other commands are the same. Tested package versions are pinned; minor cross-platform floating-point differences are possible. No external datasets or network services are used for training. `score.py` validates format and creates the chart; it does not score hidden-set accuracy. Input files retain their root-level hyphenated names; the original brief's `data/` paths are adapted above.

To use saved models without retraining:

```powershell
python predict.py --input validation.csv --output output/reproduced_predictions.csv
python predict.py --role core --input december-chart-inputs.csv --output output/reproduced_december.csv
```

## Code and evidence

- [solution.py](solution.py): input checks, features, model selection, evaluation and refit.
- [predict.py](predict.py): saved-model inference.
- [diagnostics.py](diagnostics.py): residual slices and geographic coverage; no fitting.
- [test_solution.py](test_solution.py): six checks for feature isolation, invalid weights, schemas and saved-model parity.
- [metrics.json](output/metrics.json): original selection, holdout results and environment.
- [diagnostics.json](output/diagnostics.json): post-evaluation audit and model/prediction hashes.
- [verification.txt](output/verification.txt): scorer and regression-check results.
