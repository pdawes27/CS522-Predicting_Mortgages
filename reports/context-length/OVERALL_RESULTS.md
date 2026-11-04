# Overall effect of extending context from 12 to 24 months

This reexamines the completed runs using overall state-classification metrics; no models were refitted. The original experiment protocol is preserved as a historical record.

Three paired seeds, identical training/validation target samples, and 809,831 identical test loan-months. Both arms require 24 consecutive months of available history. The target remains next month.

| Model | Context | Macro-F1 (higher better) | Multiclass log loss (lower better) | Accuracy |
|---|---:|---:|---:|---:|
| transformer | 12 | 0.3399 | 0.08504 | 98.072% |
| transformer | 24 | 0.3371 | 0.08524 | 98.059% |
| transformer_conditioned | 12 | 0.3326 | 0.08530 | 98.056% |
| transformer_conditioned | 24 | 0.3372 | 0.08528 | 98.059% |
| xgboost | 12 | 0.3491 | 0.08991 | 98.146% |
| xgboost | 24 | 0.3498 | 0.08950 | 98.148% |
| persistence | 12 | 0.3706 | — | 98.011% |
| persistence | 24 | 0.3706 | — | 98.011% |

## Interpretation

- The ordinary transformer gets slightly worse on average in macro-F1, log loss, and accuracy. Macro-F1 improves in two seeds but a larger decline in the third reverses the mean gain.
- The conditioned transformer has higher mean macro-F1 and almost unchanged log loss, but macro-F1 improves in only one of three seeds. Its changes are unstable: +0.05754, -0.03110, and -0.01253. This does not establish a dependable benefit.
- XGBoost has a small mean macro-F1 increase and improves log loss in all three seeds. Accuracy is essentially unchanged. The absolute gains are modest.
- Persistence is identical because it uses only the last observed state and the evaluated targets are held fixed.
- Both transformer variants have lower multiclass log loss than XGBoost at either length, but worse macro-F1. Probability quality and hard state classification therefore give different rankings; neither model family is uniformly best. Persistence has no probabilistic log-loss result.

## Per-class pattern

For the ordinary transformer, mean F1 falls for current, 30–59 days, 60–89 days, 90+ days, and payoff when increasing context. The 30–59-day class falls from 0.31809 to 0.30504. Adverse-exit and other-exit F1 remain zero. The conditioned model improves 60–89-day F1 from 0.18331 to 0.21728, but the overall seed inconsistency remains.

## Conclusion

Keep 12 months as the default. These runs do not establish an overall transformer improvement from doubling context. Accuracy stays around 98% because unchanged/current loans dominate, so it should not be used alone to judge forecasting improvements. The full per-seed overall and per-class metrics are saved in overall_metrics.csv and per_class_metrics.csv. These are exploratory results from inspected test data, not independent confirmation.
