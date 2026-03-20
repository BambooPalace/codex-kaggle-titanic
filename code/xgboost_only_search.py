#!/usr/bin/env python3
import json
import os
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

TARGET = "Survived"
RANDOM_SEEDS = [17, 42, 88]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FamilySize",
    "SibSp",
    "Parch",
    "LogFare",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "Deck",
]

BINARY_FEATURES = [
    "IsAlone",
    "CabinKnown",
    "AgeMissing",
    "FareMissing",
    "IsChild",
]

TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Royalty",
    "the Countess": "Royalty",
    "Countess": "Royalty",
    "Dona": "Royalty",
    "Sir": "Royalty",
    "Don": "Royalty",
    "Jonkheer": "Royalty",
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Dr": "Officer",
    "Rev": "Officer",
}


def normalize_title(raw_title: str) -> str:
    mapped = TITLE_MAP.get(raw_title, raw_title)
    return mapped if mapped in {"Mr", "Mrs", "Miss", "Master", "Royalty", "Officer"} else "Rare"


def extract_title(series: pd.Series) -> pd.Series:
    title = series.fillna("Unknown, Unknown.").str.extract(r",\s*([^\.]+)\.", expand=False)
    return title.fillna("Unknown").map(normalize_title)


class SimpleTitanicFeatures:
    def fit(self, df: pd.DataFrame) -> "SimpleTitanicFeatures":
        temp_title = extract_title(df["Name"])
        age_frame = pd.DataFrame(
            {
                "Age": df["Age"],
                "Sex": df["Sex"].fillna("missing"),
                "Pclass": df["Pclass"],
                "Title": temp_title,
            }
        )
        self.embarked_mode_ = df["Embarked"].mode(dropna=True).iloc[0]
        self.fare_median_ = float(df["Fare"].median())
        self.fare_by_pclass_ = df.groupby("Pclass")["Fare"].median().dropna().to_dict()
        self.age_by_title_sex_class_ = (
            age_frame.groupby(["Title", "Sex", "Pclass"])["Age"].median().dropna().to_dict()
        )
        self.age_by_sex_class_ = age_frame.groupby(["Sex", "Pclass"])["Age"].median().dropna().to_dict()
        self.age_global_ = float(age_frame["Age"].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        title = extract_title(df["Name"]).astype(str)
        embarked = df["Embarked"].fillna(self.embarked_mode_).astype(str)
        fare_missing = df["Fare"].isna().astype(int)
        fare = df["Fare"].fillna(df["Pclass"].map(self.fare_by_pclass_)).fillna(self.fare_median_).astype(float)

        ages = []
        for raw_age, row_title, sex, pclass in zip(
            df["Age"],
            title,
            df["Sex"].fillna("missing"),
            df["Pclass"],
            strict=False,
        ):
            if pd.notna(raw_age):
                ages.append(float(raw_age))
            else:
                ages.append(
                    float(
                        self.age_by_title_sex_class_.get(
                            (row_title, sex, pclass),
                            self.age_by_sex_class_.get((sex, pclass), self.age_global_),
                        )
                    )
                )
        age = pd.Series(ages, index=df.index, dtype="float64")
        family_size = (df["SibSp"] + df["Parch"] + 1).astype(float)

        return pd.DataFrame(
            {
                "Age": age,
                "Fare": fare,
                "FamilySize": family_size,
                "SibSp": df["SibSp"].astype(float),
                "Parch": df["Parch"].astype(float),
                "LogFare": np.log1p(fare),
                "Pclass": df["Pclass"].astype(str),
                "Sex": df["Sex"].fillna("missing").astype(str),
                "Embarked": embarked,
                "Title": title,
                "Deck": df["Cabin"].fillna("U").astype(str).str[0].str.upper(),
                "IsAlone": (family_size == 1).astype(int),
                "CabinKnown": df["Cabin"].notna().astype(int),
                "AgeMissing": df["Age"].isna().astype(int),
                "FareMissing": fare_missing.astype(int),
                "IsChild": (age < 16).astype(int),
            },
            index=df.index,
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def build_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        gamma=params["gamma"],
        n_jobs=1,
    )


def parameter_grid() -> list[dict]:
    grid = {
        "n_estimators": [120, 220],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.06],
        "min_child_weight": [1, 3],
        "subsample": [0.8, 0.95],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 4.0],
        "reg_alpha": [0.0, 0.2],
        "gamma": [0.0],
    }
    combos = []
    for values in product(*grid.values()):
        combos.append(dict(zip(grid.keys(), values, strict=False)))
    return combos


def score_params(train_df: pd.DataFrame, params: dict) -> dict:
    y = train_df[TARGET].astype(int)
    X_raw = train_df.drop(columns=[TARGET])
    fold_scores: list[float] = []

    for seed in RANDOM_SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, valid_idx in cv.split(X_raw, y):
            fb = SimpleTitanicFeatures()
            x_train = fb.fit_transform(X_raw.iloc[train_idx])
            x_valid = fb.transform(X_raw.iloc[valid_idx])

            preprocess = make_preprocessor()
            x_train_enc = preprocess.fit_transform(x_train)
            x_valid_enc = preprocess.transform(x_valid)

            model = build_model(params)
            model.fit(x_train_enc, y.iloc[train_idx])
            preds = (model.predict_proba(x_valid_enc)[:, 1] >= 0.5).astype(int)
            fold_scores.append(float(accuracy_score(y.iloc[valid_idx], preds)))

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))
    robust_score = float(mean_score - 0.35 * std_score)
    return {
        "params": params,
        "mean_accuracy": mean_score,
        "std_accuracy": std_score,
        "robust_score": robust_score,
        "fold_scores": fold_scores,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_df = pd.read_csv(root / "data" / "train.csv")
    test_df = pd.read_csv(root / "data" / "test.csv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "artifacts" / "xgboost_only" / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    all_params = parameter_grid()
    coarse_candidates = []
    for idx, params in enumerate(all_params, start=1):
        result = score_params(train_df, params)
        coarse_candidates.append(result)
        if idx % 50 == 0:
            print(f"Scored {idx}/{len(all_params)} parameter sets")

    results_df = pd.DataFrame(
        [
            {
                "mean_accuracy": r["mean_accuracy"],
                "std_accuracy": r["std_accuracy"],
                "robust_score": r["robust_score"],
                "params_json": json.dumps(r["params"], sort_keys=True),
                "fold_scores_json": json.dumps(r["fold_scores"]),
            }
            for r in coarse_candidates
        ]
    ).sort_values(by=["robust_score", "mean_accuracy"], ascending=[False, False])
    results_df.to_csv(out_dir / "cv_results.csv", index=False)

    best_params = json.loads(results_df.iloc[0]["params_json"])
    fb = SimpleTitanicFeatures()
    x_train = fb.fit_transform(train_df.drop(columns=[TARGET]))
    x_test = fb.transform(test_df)
    preprocess = make_preprocessor()
    x_train_enc = preprocess.fit_transform(x_train)
    x_test_enc = preprocess.transform(x_test)

    final_model = build_model(best_params)
    final_model.fit(x_train_enc, train_df[TARGET].astype(int))
    test_probs = final_model.predict_proba(x_test_enc)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    submission = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"].astype(int),
            "Survived": test_preds,
        }
    )
    submission.to_csv(root / "predictions.csv", index=False)
    submission.to_csv(out_dir / "xgboost_submission.csv", index=False)

    report = [
        "# XGBoost Only Search",
        "",
        f"- Parameter sets tested: {len(all_params)}",
        f"- Best mean CV accuracy: {results_df.iloc[0]['mean_accuracy']:.5f}",
        f"- Best std CV accuracy: {results_df.iloc[0]['std_accuracy']:.5f}",
        f"- Best robust score: {results_df.iloc[0]['robust_score']:.5f}",
        f"- Best params: `{results_df.iloc[0]['params_json']}`",
        f"- Submission path: `{(out_dir / 'xgboost_submission.csv').resolve()}`",
        "",
        "Top 10 parameter sets are saved in `cv_results.csv`.",
    ]
    (out_dir / "report.md").write_text("\n".join(report))

    print(f"XGBoost artifacts: {out_dir}")
    print(f"Best params: {best_params}")
    print(f"Best mean CV accuracy: {results_df.iloc[0]['mean_accuracy']:.5f}")
    print(f"Submission written to: {(root / 'predictions.csv').resolve()}")


if __name__ == "__main__":
    main()
