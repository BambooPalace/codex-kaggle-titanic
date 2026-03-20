# Round 2 Search

## Validation Ranking
- catboost_simple: mean=0.83187, std=0.02400, robust=0.82347
- histgb_simple: mean=0.83254, std=0.02823, robust=0.82266
- logistic_simple: mean=0.82961, std=0.02539, robust=0.82073

## Submission Candidates
- candidate_logistic.csv: survival_rate=0.40670, female_survival_rate=0.94737, male_survival_rate=0.09774
- candidate_histgb.csv: survival_rate=0.36842, female_survival_rate=0.82895, male_survival_rate=0.10526
- candidate_catboost.csv: survival_rate=0.39713, female_survival_rate=0.94079, male_survival_rate=0.08647
- candidate_blend.csv: survival_rate=0.39713, female_survival_rate=0.94079, male_survival_rate=0.08647
- candidate_blend_adjusted.csv: survival_rate=0.40431, female_survival_rate=0.94079, male_survival_rate=0.09774

Default exported submission: candidate_blend.csv
Artifacts: /Users/claire_gong/ocbc/codex/kaggle/artifacts/round2/20260320_105603