# Experiments and evaluation

## Original coursework, January 2026

The private EEC 174 notebook contains the feature implementation and saved experiment outputs. The supplied split retained 5,972 training breaths and 1,247 validation breaths after cleaning. The validation labels were 842 Non-PVA, 301 BSA, and 104 DTA.

The comparison used all 16 numeric features with Random Forest, L1 Logistic Regression (`saga`), and KNN (`k=5`). StandardScaler was fitted on training rows and applied to validation rows for the recorded scaled comparison. A separate Random Forest experiment used only exhaled volume and expiratory pressure area; its saved accuracy was 0.63 and macro F1 was 0.41.

The rule baseline applied the supplied feature thresholds:

- DTA pattern: expiration time below 0.3 seconds, with a volume ratio below 0.25; or a ratio below 0.5 and exhaled volume below 100 mL.
- BSA pattern: volume ratio below 0.9 when the DTA rule did not apply.
- Otherwise: Non-PVA.

These are the implemented coursework rules. They encode a useful domain baseline for this experiment.

### Prediction adjustment in the retained reports

The learned-model cells relabelled the preceding array element as DTA when the current prediction was DTA. The rule cell could also relabel a preceding Non-PVA prediction. The published historical scores therefore describe the complete reported pipelines, including that adjustment.

The array-based implementation did not check patient boundaries or adjacency of breath identifiers; at the first element, a negative index could address the final element. The public implementation uses independent per-breath predictions. It does not apply this sequence heuristic. This methodological change is one reason a new run should be reported separately from the retained scores.

### Repeated runs and cross-validation

Ten Random Forest seeds yielded mean DTA precision 0.9250, recall 0.0365, and F1 0.0680. The five-fold Logistic Regression experiment used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and fitted a scaler on each fold's training data. Mean macro precision was 0.7415, recall 0.6920, and F1 0.7079.

That cross-validation split individual breaths. The saved notebook does not establish unseen-patient performance. The fixed source snapshot, rounded class reports, and fold summaries are recorded in [the aggregate results](../results/coursework_metrics.json).

## Public package, October 2026

The standalone implementation uses the following evaluation sequence:

1. Validate feature/label identities, join one-to-one, and record exclusion counts.
2. Reserve 25% of patient groups by default with a fixed-seed `GroupShuffleSplit`.
3. Require all three labels in training and validation. Exclude identifiers from features.
4. Fit each learned model on the same training rows. Scale Logistic Regression and KNN inside their pipelines; Random Forest uses the numeric features directly.
5. Evaluate all four baselines on the same held-out patients, without sequence adjustment.
6. Run grouped Logistic Regression CV on training patients only, refitting preprocessing for each fold.
7. Save package versions, seed, split counts, per-class metrics, confusion matrices, and predictions.

The held-out set is used for evaluation. The package does not tune hyperparameters against it. `--cv-folds 0` disables training-set CV for small datasets. A fold lacking a class stops the run with a coverage error.

See the scikit-learn documentation for [group-based splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) and [preprocessing without data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## How to read the metrics

Accuracy weights each example equally; a frequent class can dominate it. Macro F1 weights each class equally. DTA recall is the fraction of labelled DTA examples detected. The confusion matrix shows what the missed examples became.

The historical comparison is valuable because these views disagree: Random Forest achieved 0.89 accuracy and 0.02 DTA recall. The feature-rule baseline achieved 0.90 accuracy, 0.81 macro F1, and 0.54 DTA recall. The conclusion is specific to the recorded dataset, split, and pipeline.

Synthetic-demo labels are generated from mathematical waveform patterns with class-dependent volume ratios and timings. High scores on that demo are expected and exercise the code path; they provide no estimate of performance on patient recordings.
