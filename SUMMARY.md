# Titanic Competition Summary

## Goal

Iterate on local models for the Titanic Kaggle competition and use leaderboard accuracy to judge real generalization.

## Attempt 1

- Model: `soft_vote_core`
- Approach: soft-voting ensemble of logistic regression, extra trees, and histogram gradient boosting
- Features: broad engineered set including title, family size, fare per person, ticket group size, surname group size, cabin deck, ticket prefix, age/fare bands, and interaction features such as sex-class and title-class
- Local CV: `0.84393`
- Kaggle score: `0.765`
- Result: strongest local CV, weakest leaderboard score; likely overfit

## Attempt 2

- Model: `blend_simple`
- Approach: weighted blend of logistic regression, histogram gradient boosting, and CatBoost
- Features: simpler engineered set with class, sex, embarked, title, cabin deck, family size, fare, log-fare, fare per person, family label, and child/mother/missingness indicators
- Local CV: `0.83456`
- Kaggle score: `0.770`
- Result: improved leaderboard performance after simplifying features and model selection

## Attempt 3

- Model: `xgboost_only`
- Approach: XGBoost only with simple preprocessing and focused CV parameter search
- Features: compact tabular set with class, sex, embarked, title, deck, age, fare, log-fare, family size, is-alone, child flag, and missingness indicators
- Best params:
  - `max_depth=3`
  - `learning_rate=0.03`
  - `n_estimators=120`
  - `min_child_weight=1`
  - `subsample=0.8`
  - `colsample_bytree=0.8`
  - `reg_lambda=1.0`
  - `reg_alpha=0.0`
  - `gamma=0.0`
- Local CV: `0.83238`
- Kaggle score: `0.773`
- Result: best leaderboard score so far

## Key Takeaways

- Higher local CV did not translate directly to better Kaggle accuracy.
- Simpler feature engineering generalized better than the first broader ensemble.
- A tuned single XGBoost model outperformed the blended round-2 model on Kaggle.

## Comparison

| Attempt | Model | Algorithm | Local CV | Kaggle Score | Notes |
|---|---|---|---:|---:|---|
| 1 | `soft_vote_core` | Logistic + Extra Trees + HistGB soft vote | 0.84393 | 0.765 | Best local CV, weakest leaderboard |
| 2 | `blend_simple` | Logistic + HistGB + CatBoost weighted blend | 0.83456 | 0.770 | Simpler blend, better generalization |
| 3 | `xgboost_only` | Tuned XGBoost | 0.83238 | 0.773 | Best Kaggle score |
