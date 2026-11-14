# Bank Term Deposit Subscription Prediction

Predicts whether a bank client will subscribe to a term deposit after a marketing call
(binary classification), built as part of my MSc AI coursework at Seneca College.

## Dataset
Bank Marketing dataset (UCI Machine Learning Repository): 4,521 clients, 16 features,
11.5% positive class. The data is not included in this repo; download it from the UCI source.

## Approach
- **EDA and cleaning:** found that "unknown" in `poutcome` means "not contacted before"
  (verified against `pdays` and `previous`) and handled it with a dedicated feature.
- **Leakage prevention:** removed `duration` (only known after the call); all preprocessing
  (capping, log transform, scaling, encoding) runs inside a scikit-learn Pipeline so it is
  fitted on training data only.
- **Validation:** stratified 80/20 train/test split, with stratified 5-fold CV on the training set.
- **Models:** logistic regression, random forest, gradient boosting, tuned with GridSearchCV
  (scored on PR-AUC).
- **Imbalance:** compared class weights and SMOTE against a threshold-tuning approach.

## Key results
| | ROC-AUC | PR-AUC |
|---|---|---|
| Final model (random forest), test set | 0.75 | 0.34 (no-skill baseline: 0.115) |

- At the chosen threshold, the hit rate of the call list rises from 11.5% to 33.6% (2.9x).
- SMOTE and class weights did not improve ranking quality and distorted predicted
  probabilities, so the final model uses a cross-validated threshold instead.
- The advantage over logistic regression is small and not statistically confirmed
  (bootstrap interval includes zero).

## Limitations and next steps
- Most missed subscribers are clients never contacted before, where features carry little signal.
- The model flags most clients over 60, a fairness risk for deployment.
- Future work: cost-based threshold, time-aware validation, SHAP explanations, drift monitoring.

## Tools
Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn
