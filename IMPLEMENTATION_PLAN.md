# Kaggle Titanic Implementation Plan

## Objective
Build an end-to-end local pipeline for the Titanic Kaggle competition that:

- trains competitive models on `data/train.csv`
- evaluates them with reproducible local validation
- records intermediate parameters, findings, and evaluation metrics
- generates a submission-ready `predictions.csv` for `data/test.csv`
- runs in one command after dependencies are installed

## Local Inputs

- `data/train.csv`
- `data/test.csv`
- `data/gender_submission.csv`
- `competition_overview.md`
- `dataset_description.md`

## Known Constraints

- The competition metric is classification accuracy.
- Test labels are not available locally, so leaderboard feedback is only available after Kaggle submission.
- The current environment has `python3` available, but the standard Python data stack is not yet installed locally.

## Execution Plan

### 1. Project scaffolding

Create a clean runnable structure under `code/` with:

- a main training and inference script
- utility functions for feature engineering and experiment logging
- a simple output directory layout for models, metrics, and submissions

### 2. Reproducible environment

Add a dependency definition so the workflow can be rerun consistently, likely including:

- `pandas`
- `numpy`
- `scikit-learn`
- optional gradient boosting package if it is available and worth using

### 3. Data loading and profiling

Load the training and test datasets and record:

- shapes
- columns
- target balance
- missingness by feature
- candidate categorical and numeric feature groups

These findings should be saved in the run artifacts for later review.

### 4. Feature engineering

Implement strong Titanic-specific features that commonly help leaderboard performance:

- title extracted from passenger name
- family size and family grouping features
- is-alone indicator
- ticket group size
- cabin-known flag
- cabin deck extracted from cabin
- fare bands or transformed fare
- age imputation features
- interaction-aware use of sex, class, age, fare, and embarkation

The feature pipeline should be consistent across train and test.

### 5. Model benchmarking

Benchmark a compact but strong set of models suitable for small tabular data:

- logistic regression
- random forest
- extra trees
- gradient boosting or histogram gradient boosting

If validation results support it, also test:

- soft voting ensemble
- simple stacked blend

### 6. Model selection strategy

Use stratified cross-validation and compare:

- mean validation accuracy
- fold-by-fold stability
- parameter choices used in each run

Select the final model based on the best tradeoff between accuracy and robustness rather than a single favorable split.

### 7. Experiment tracking

Persist intermediate outputs for each run, including:

- run configuration
- engineered feature list
- imputation decisions
- model hyperparameters
- fold metrics
- aggregate ranking of candidate models
- final selected model summary

### 8. Submission generation

Train the final selected model on all training rows and generate:

- `predictions.csv`

The file must contain exactly:

- `PassengerId`
- `Survived`

### 9. End-to-end verification

Before finishing, run the pipeline locally and verify:

- the script completes without manual intervention
- logs and metrics are written
- `predictions.csv` is created successfully
- the row count and header format match Kaggle requirements

## Deliverables

- runnable pipeline code under `code/`
- dependency file
- logged metrics and findings
- final `predictions.csv`

## Expected One-Run Workflow

After implementation, the intended usage should look like:

1. install dependencies
2. run the main training and prediction script
3. obtain `predictions.csv` and supporting experiment artifacts
