# Titanic Run Report

## Data Profile
- Train shape: (891, 12)
- Test shape: (418, 11)
- Target distribution: {0: 0.6161616161616161, 1: 0.3838383838383838}
- Highest missingness in train: {'Cabin': 687, 'Age': 177, 'Embarked': 2}

## Engineered Features
- Numeric features: Age, Fare, FarePerPerson, FamilySize, SibSp, Parch, NameLength, TicketGroupSize, SurnameGroupSize, PclassNum
- Categorical features: Sex, Embarked, Pclass, Title, CabinDeck, TicketPrefix, FamilyLabel, AgeBand, FareBand, SexPclass, TitlePclass
- Binary features: IsAlone, CabinKnown, AgeMissing, FareMissing, EmbarkedMissing, IsChild, IsMother

## Cross-Validation Ranking
- soft_vote_core: mean=0.84393, std=0.03013, min=0.79775, max=0.90000
- soft_vote_full: mean=0.84281, std=0.02865, min=0.79775, max=0.90000
- svc_rbf: mean=0.83944, std=0.02997, min=0.79775, max=0.90000
- extra_trees: mean=0.83834, std=0.02282, min=0.80899, max=0.87778
- random_forest: mean=0.83609, std=0.02495, min=0.80899, max=0.87778

## Final Selection
- Selected model: soft_vote_core
- Selection note: Soft vote of logistic, extra trees, and gradient boosting.
- Training-set holdout accuracy on last fold split: 0.83146
- Submission file: /Users/claire_gong/ocbc/codex/kaggle/predictions.csv
- Artifact directory: artifacts/runs/20260320_104239