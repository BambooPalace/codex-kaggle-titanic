#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "Survived"
BASE_RANDOM_SEEDS = [11, 42, 77, 123, 202]
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "LogFare",
    "FamilySize",
    "SibSp",
    "Parch",
    "FarePerPerson",
]

CATEGORICAL_FEATURES = [
    "Pclass",
    "Sex",
    "Embarked",
    "Title",
    "CabinDeck",
    "FamilyLabel",
]

BINARY_FEATURES = [
    "IsAlone",
    "CabinKnown",
    "AgeMissing",
    "FareMissing",
    "IsChild",
    "IsMother",
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


def label_family_size(size: int) -> str:
    if size == 1:
        return "Solo"
    if size <= 4:
        return "Small"
    return "Large"


class FeatureBuilder:
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        title = extract_title(df["Name"])
        temp = pd.DataFrame(
            {
                "Age": df["Age"],
                "Title": title,
                "Sex": df["Sex"].fillna("missing"),
                "Pclass": df["Pclass"],
            }
        )
        self.embarked_mode = df["Embarked"].mode(dropna=True).iloc[0]
        self.global_fare = float(df["Fare"].median())
        self.fare_by_pclass = df.groupby("Pclass")["Fare"].median().dropna().to_dict()
        self.age_by_title_sex_class = (
            temp.groupby(["Title", "Sex", "Pclass"])["Age"].median().dropna().to_dict()
        )
        self.age_by_sex_class = temp.groupby(["Sex", "Pclass"])["Age"].median().dropna().to_dict()
        self.age_by_sex = temp.groupby("Sex")["Age"].median().dropna().to_dict()
        self.global_age = float(temp["Age"].median())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        title = extract_title(df["Name"])
        embarked = df["Embarked"].fillna(self.embarked_mode).astype(str)
        fare_missing = df["Fare"].isna().astype(int)
        fare = df["Fare"].fillna(df["Pclass"].map(self.fare_by_pclass)).fillna(self.global_fare).astype(float)

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
                continue
            ages.append(
                float(
                    self.age_by_title_sex_class.get(
                        (row_title, sex, pclass),
                        self.age_by_sex_class.get((sex, pclass), self.age_by_sex.get(sex, self.global_age)),
                    )
                )
            )
        age = pd.Series(ages, index=df.index, dtype="float64")
        family_size = (df["SibSp"] + df["Parch"] + 1).astype(int)
        sex = df["Sex"].fillna("missing").astype(str)
        cabin_known = df["Cabin"].notna().astype(int)
        cabin_deck = df["Cabin"].fillna("U").astype(str).str[0].str.upper()
        is_child = (age < 16).astype(int)
        is_mother = ((sex == "female") & (df["Parch"] > 0) & (age > 18) & (title != "Miss")).astype(int)

        out = pd.DataFrame(
            {
                "Age": age,
                "Fare": fare,
                "LogFare": np.log1p(fare),
                "FamilySize": family_size.astype(float),
                "SibSp": df["SibSp"].astype(float),
                "Parch": df["Parch"].astype(float),
                "FarePerPerson": (fare / family_size.clip(lower=1)).astype(float),
                "Pclass": df["Pclass"].astype(str),
                "Sex": sex,
                "Embarked": embarked,
                "Title": title.astype(str),
                "CabinDeck": cabin_deck,
                "FamilyLabel": family_size.map(label_family_size).astype(str),
                "IsAlone": (family_size == 1).astype(int),
                "CabinKnown": cabin_known.astype(int),
                "AgeMissing": df["Age"].isna().astype(int),
                "FareMissing": fare_missing.astype(int),
                "IsChild": is_child.astype(int),
                "IsMother": is_mother.astype(int),
            },
            index=df.index,
        )
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def hist_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


@dataclass
class ModelResult:
    name: str
    fold_scores: list[float]
    mean_accuracy: float
    std_accuracy: float
    robust_score: float


def fit_predict_logistic(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline(
        [
            ("prep", linear_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=0.4,
                    solver="liblinear",
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1], model.predict(test_x).astype(int)


def fit_predict_histgb(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = Pipeline(
        [
            ("prep", hist_preprocessor()),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.045,
                    max_depth=3,
                    max_iter=250,
                    min_samples_leaf=10,
                    l2_regularization=0.3,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    probs = model.predict_proba(test_x)[:, 1]
    return probs, (probs >= 0.5).astype(int)


def fit_predict_catboost(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cat_cols = CATEGORICAL_FEATURES
    model = CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.035,
        loss_function="Logloss",
        eval_metric="Accuracy",
        l2_leaf_reg=5.0,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_x, train_y, cat_features=cat_cols)
    probs = model.predict_proba(test_x)[:, 1]
    return probs, (probs >= 0.5).astype(int)


def fit_predict_blend(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lr_probs, _ = fit_predict_logistic(train_x, train_y, test_x)
    hgb_probs, _ = fit_predict_histgb(train_x, train_y, test_x)
    cb_probs, _ = fit_predict_catboost(train_x, train_y, test_x)
    probs = 0.25 * lr_probs + 0.2 * hgb_probs + 0.55 * cb_probs
    return probs, (probs >= 0.5).astype(int)


def fit_predict_blend_adjusted(
    train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    probs, preds = fit_predict_blend(train_x, train_y, test_x)
    adjusted = preds.copy()
    female_pclass = test_x["Sex"].eq("female") & test_x["Pclass"].isin(["1", "2"])
    adjusted[female_pclass.to_numpy()] = 1
    master_rescue = (test_x["Title"] == "Master") & (test_x["Age"] <= 12)
    adjusted[master_rescue.to_numpy()] = 1
    return probs, adjusted.astype(int)


def cross_validate_model(name: str, train_df: pd.DataFrame, fit_predict_fn) -> ModelResult:
    y = train_df[TARGET].astype(int)
    X = train_df.drop(columns=[TARGET])
    scores = []
    for seed in BASE_RANDOM_SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, valid_idx in cv.split(X, y):
            fold_train = X.iloc[train_idx]
            fold_valid = X.iloc[valid_idx]
            fb = FeatureBuilder()
            x_train = fb.fit_transform(fold_train)
            x_valid = fb.transform(fold_valid)
            _, preds = fit_predict_fn(x_train, y.iloc[train_idx], x_valid)
            scores.append(accuracy_score(y.iloc[valid_idx], preds))
    scores = [float(s) for s in scores]
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    robust = float(mean - 0.35 * std)
    return ModelResult(name=name, fold_scores=scores, mean_accuracy=mean, std_accuracy=std, robust_score=robust)


def build_submission_candidates(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    y = train_df[TARGET].astype(int)
    X_train_raw = train_df.drop(columns=[TARGET])
    X_test_raw = test_df.copy()

    fb = FeatureBuilder()
    X_train = fb.fit_transform(X_train_raw)
    X_test = fb.transform(X_test_raw)

    lr_probs, lr_preds = fit_predict_logistic(X_train, y, X_test)
    hgb_probs, hgb_preds = fit_predict_histgb(X_train, y, X_test)
    cb_probs, cb_preds = fit_predict_catboost(X_train, y, X_test)

    blend_probs = 0.25 * lr_probs + 0.2 * hgb_probs + 0.55 * cb_probs
    blend_preds = (blend_probs >= 0.5).astype(int)

    adjusted_preds = blend_preds.copy()
    female_pclass = X_test["Sex"].eq("female") & X_test["Pclass"].isin(["1", "2"])
    adjusted_preds[female_pclass.to_numpy()] = 1
    master_rescue = (X_test["Title"] == "Master") & (X_test["Age"] <= 12)
    adjusted_preds[master_rescue.to_numpy()] = 1

    candidates = {
        "candidate_logistic.csv": lr_preds,
        "candidate_histgb.csv": hgb_preds,
        "candidate_catboost.csv": cb_preds,
        "candidate_blend.csv": blend_preds,
        "candidate_blend_adjusted.csv": adjusted_preds,
    }

    summary_rows = []
    for filename, preds in candidates.items():
        sub = pd.DataFrame({"PassengerId": test_df["PassengerId"].astype(int), "Survived": preds.astype(int)})
        sub.to_csv(output_dir / filename, index=False)
        summary_rows.append(
            {
                "submission_file": filename,
                "survival_rate": float(sub["Survived"].mean()),
                "female_survival_rate": float(
                    sub.merge(test_df[["PassengerId", "Sex"]], on="PassengerId").query("Sex == 'female'")["Survived"].mean()
                ),
                "male_survival_rate": float(
                    sub.merge(test_df[["PassengerId", "Sex"]], on="PassengerId").query("Sex == 'male'")["Survived"].mean()
                ),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    train_df = pd.read_csv(root / "data" / "train.csv")
    test_df = pd.read_csv(root / "data" / "test.csv")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "artifacts" / "round2" / timestamp
    submissions_dir = out_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=False)

    model_fns = {
        "logistic_simple": fit_predict_logistic,
        "histgb_simple": fit_predict_histgb,
        "catboost_simple": fit_predict_catboost,
        "blend_simple": fit_predict_blend,
        "blend_adjusted": fit_predict_blend_adjusted,
    }

    results = []
    for name, fn in model_fns.items():
        results.append(cross_validate_model(name, train_df, fn))

    results_df = pd.DataFrame(
        [
            {
                "model_name": r.name,
                "mean_accuracy": r.mean_accuracy,
                "std_accuracy": r.std_accuracy,
                "robust_score": r.robust_score,
                "fold_scores": json.dumps(r.fold_scores),
            }
            for r in results
        ]
    ).sort_values(by=["robust_score", "mean_accuracy"], ascending=[False, False])
    results_df.to_csv(out_dir / "cv_summary.csv", index=False)

    submission_summary = build_submission_candidates(train_df, test_df, submissions_dir)
    submission_summary.to_csv(out_dir / "submission_candidates.csv", index=False)

    ranked_names = results_df["model_name"].tolist()
    if ranked_names[0] == "blend_adjusted":
        chosen_name = "candidate_blend_adjusted.csv"
    elif ranked_names[0] == "blend_simple":
        chosen_name = "candidate_blend.csv"
    elif ranked_names[0] == "catboost_simple":
        chosen_name = "candidate_catboost.csv"
    elif ranked_names[0] == "histgb_simple":
        chosen_name = "candidate_histgb.csv"
    else:
        chosen_name = "candidate_logistic.csv"
    chosen_path = submissions_dir / chosen_name
    final_path = root / "predictions.csv"
    pd.read_csv(chosen_path).to_csv(final_path, index=False)

    report_lines = [
        "# Round 2 Search",
        "",
        "## Validation Ranking",
    ]
    for row in results_df.itertuples(index=False):
        report_lines.append(
            f"- {row.model_name}: mean={row.mean_accuracy:.5f}, std={row.std_accuracy:.5f}, robust={row.robust_score:.5f}"
        )
    report_lines.extend(
        [
            "",
            "## Submission Candidates",
        ]
    )
    for row in submission_summary.itertuples(index=False):
        report_lines.append(
            f"- {row.submission_file}: survival_rate={row.survival_rate:.5f}, "
            f"female_survival_rate={row.female_survival_rate:.5f}, male_survival_rate={row.male_survival_rate:.5f}"
        )
    report_lines.extend(
        [
            "",
            f"Default exported submission: {chosen_name}",
            f"Artifacts: {out_dir}",
        ]
    )
    (out_dir / "round2_report.md").write_text("\n".join(report_lines))

    print(f"Round 2 artifacts: {out_dir}")
    print("Validation ranking:")
    print(results_df[["model_name", "mean_accuracy", "std_accuracy", "robust_score"]].to_string(index=False))
    print(f"Default submission exported to: {final_path}")


if __name__ == "__main__":
    main()
