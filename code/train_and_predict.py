#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
TARGET = "Survived"

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

NUMERIC_FEATURES = [
    "Age",
    "Fare",
    "FarePerPerson",
    "FamilySize",
    "SibSp",
    "Parch",
    "NameLength",
    "TicketGroupSize",
    "SurnameGroupSize",
    "PclassNum",
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked",
    "Pclass",
    "Title",
    "CabinDeck",
    "TicketPrefix",
    "FamilyLabel",
    "AgeBand",
    "FareBand",
    "SexPclass",
    "TitlePclass",
]

BINARY_FEATURES = [
    "IsAlone",
    "CabinKnown",
    "AgeMissing",
    "FareMissing",
    "EmbarkedMissing",
    "IsChild",
    "IsMother",
]

ALL_ENGINEERED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

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
    title = TITLE_MAP.get(raw_title, raw_title)
    if title not in {"Mr", "Mrs", "Miss", "Master", "Royalty", "Officer"}:
        return "Rare"
    return title


def extract_title(name_series: pd.Series) -> pd.Series:
    titles = name_series.fillna("Unknown, Unknown.").str.extract(r",\s*([^\.]+)\.", expand=False)
    return titles.fillna("Unknown").map(normalize_title)


def extract_surname(name_series: pd.Series) -> pd.Series:
    surnames = name_series.fillna("Unknown").str.split(",", n=1).str[0].str.strip().str.upper()
    return surnames.replace("", "UNKNOWN")


def clean_ticket(ticket_series: pd.Series) -> pd.Series:
    cleaned = (
        ticket_series.fillna("MISSING")
        .astype(str)
        .str.upper()
        .str.replace(r"[\./]", "", regex=True)
        .str.replace(r"\s+", "", regex=True)
    )
    return cleaned.replace("", "MISSING")


def extract_ticket_prefix(ticket_series: pd.Series) -> pd.Series:
    cleaned = clean_ticket(ticket_series)
    prefixes = cleaned.str.extract(r"^([A-Z]+)", expand=False).fillna("NUM")
    return prefixes.replace("", "NUM")


def label_family_size(size: int) -> str:
    if size <= 1:
        return "Solo"
    if size <= 4:
        return "Small"
    return "Large"


def to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


class TitanicFeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "TitanicFeatureBuilder":
        frame = X.copy()
        titles = extract_title(frame["Name"])
        surnames = extract_surname(frame["Name"])

        age_frame = pd.DataFrame(
            {
                "Age": frame["Age"],
                "Title": titles,
                "Sex": frame["Sex"].fillna("missing"),
                "Pclass": frame["Pclass"],
            }
        )
        self.embarked_mode_ = frame["Embarked"].mode(dropna=True).iloc[0]
        self.fare_global_ = float(frame["Fare"].median())
        self.fare_by_pclass_ = frame.groupby("Pclass")["Fare"].median().dropna().to_dict()
        self.age_by_title_sex_class_ = (
            age_frame.groupby(["Title", "Sex", "Pclass"])["Age"].median().dropna().to_dict()
        )
        self.age_by_sex_class_ = age_frame.groupby(["Sex", "Pclass"])["Age"].median().dropna().to_dict()
        self.age_by_sex_ = age_frame.groupby("Sex")["Age"].median().dropna().to_dict()
        self.age_global_ = float(age_frame["Age"].median())
        self.ticket_counts_ = clean_ticket(frame["Ticket"]).value_counts().to_dict()
        self.surname_counts_ = surnames.value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        titles = extract_title(frame["Name"])
        surnames = extract_surname(frame["Name"])

        embarked_missing = frame["Embarked"].isna().astype(int)
        embarked = frame["Embarked"].fillna(self.embarked_mode_)

        fare_missing = frame["Fare"].isna().astype(int)
        fare = frame["Fare"].copy()
        fare = fare.fillna(frame["Pclass"].map(self.fare_by_pclass_)).fillna(self.fare_global_)

        age_missing = frame["Age"].isna().astype(int)
        age = []
        for raw_age, title, sex, pclass in zip(
            frame["Age"],
            titles,
            frame["Sex"].fillna("missing"),
            frame["Pclass"],
            strict=False,
        ):
            if pd.notna(raw_age):
                age.append(float(raw_age))
                continue
            age.append(
                float(
                    self.age_by_title_sex_class_.get(
                        (title, sex, pclass),
                        self.age_by_sex_class_.get(
                            (sex, pclass),
                            self.age_by_sex_.get(sex, self.age_global_),
                        ),
                    )
                )
            )
        age = pd.Series(age, index=frame.index, dtype="float64")

        family_size = frame["SibSp"].fillna(0) + frame["Parch"].fillna(0) + 1
        fare_per_person = fare / family_size.replace(0, 1)
        ticket_prefix = extract_ticket_prefix(frame["Ticket"])
        ticket_group_size = clean_ticket(frame["Ticket"]).map(self.ticket_counts_).fillna(1).astype(int)
        surname_group_size = surnames.map(self.surname_counts_).fillna(1).astype(int)
        cabin_known = frame["Cabin"].notna().astype(int)
        cabin_deck = frame["Cabin"].fillna("U").astype(str).str[0].str.upper()
        name_length = frame["Name"].fillna("").str.len().astype(float)
        family_label = family_size.map(label_family_size)

        age_band = pd.cut(
            age,
            bins=[-np.inf, 5, 12, 18, 30, 45, 60, np.inf],
            labels=["Infant", "Child", "Teen", "YoungAdult", "Adult", "Senior", "Elder"],
        ).astype(str)
        fare_band = pd.cut(
            fare,
            bins=[-np.inf, 8, 15, 32, 80, np.inf],
            labels=["Low", "Budget", "Mid", "High", "Luxury"],
        ).astype(str)
        sex = frame["Sex"].fillna("missing").astype(str)
        pclass = frame["Pclass"].astype(int).astype(str)
        is_child = (age < 16).astype(int)
        is_mother = ((sex == "female") & (frame["Parch"] > 0) & (age > 18) & (titles != "Miss")).astype(int)

        engineered = pd.DataFrame(
            {
                "Age": age,
                "Fare": fare.astype(float),
                "FarePerPerson": fare_per_person.astype(float),
                "FamilySize": family_size.astype(float),
                "SibSp": frame["SibSp"].astype(float),
                "Parch": frame["Parch"].astype(float),
                "NameLength": name_length,
                "TicketGroupSize": ticket_group_size.astype(float),
                "SurnameGroupSize": surname_group_size.astype(float),
                "PclassNum": frame["Pclass"].astype(float),
                "Sex": sex,
                "Embarked": embarked.astype(str),
                "Pclass": pclass,
                "Title": titles.astype(str),
                "CabinDeck": cabin_deck,
                "TicketPrefix": ticket_prefix.astype(str),
                "FamilyLabel": family_label.astype(str),
                "AgeBand": age_band,
                "FareBand": fare_band,
                "SexPclass": (sex + "_" + pclass).astype(str),
                "TitlePclass": (titles.astype(str) + "_" + pclass).astype(str),
                "IsAlone": (family_size == 1).astype(int),
                "CabinKnown": cabin_known.astype(int),
                "AgeMissing": age_missing.astype(int),
                "FareMissing": fare_missing.astype(int),
                "EmbarkedMissing": embarked_missing.astype(int),
                "IsChild": is_child.astype(int),
                "IsMother": is_mother.astype(int),
            },
            index=frame.index,
        )
        return engineered[ALL_ENGINEERED_FEATURES]

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.array(ALL_ENGINEERED_FEATURES, dtype=object)


def build_linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
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


def build_tree_preprocessor() -> ColumnTransformer:
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


def build_hist_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
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


def make_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", TitanicFeatureBuilder()),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


@dataclass
class CandidateModel:
    name: str
    estimator: BaseEstimator
    notes: str


def build_candidate_models() -> list[CandidateModel]:
    logistic = make_pipeline(
        build_linear_preprocessor(),
        LogisticRegression(
            C=0.35,
            max_iter=4000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )
    logistic_balanced = make_pipeline(
        build_linear_preprocessor(),
        LogisticRegression(
            C=0.25,
            class_weight="balanced",
            max_iter=4000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
    )
    extra_trees = make_pipeline(
        build_tree_preprocessor(),
        ExtraTreesClassifier(
            n_estimators=1200,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    )
    random_forest = make_pipeline(
        build_tree_preprocessor(),
        RandomForestClassifier(
            n_estimators=1000,
            max_depth=7,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    )
    hist_gb = make_pipeline(
        build_hist_preprocessor(),
        HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=3,
            max_iter=300,
            min_samples_leaf=9,
            l2_regularization=0.2,
            random_state=RANDOM_STATE,
        ),
    )
    svc = make_pipeline(
        build_linear_preprocessor(),
        SVC(
            C=2.0,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    )
    vote_core = VotingClassifier(
        estimators=[
            ("lr", logistic),
            ("et", extra_trees),
            ("hgb", hist_gb),
        ],
        voting="soft",
        weights=[2, 3, 3],
        n_jobs=1,
    )
    vote_full = VotingClassifier(
        estimators=[
            ("lr", logistic),
            ("et", extra_trees),
            ("hgb", hist_gb),
            ("svc", svc),
        ],
        voting="soft",
        weights=[2, 3, 3, 2],
        n_jobs=1,
    )

    return [
        CandidateModel("logistic", logistic, "Scaled linear baseline with engineered features."),
        CandidateModel(
            "logistic_balanced",
            logistic_balanced,
            "Class-balanced logistic variant for recall on the minority class.",
        ),
        CandidateModel("extra_trees", extra_trees, "High-variance tree ensemble on one-hot engineered features."),
        CandidateModel("random_forest", random_forest, "Constrained random forest for smoother fold behavior."),
        CandidateModel("hist_gradient_boosting", hist_gb, "Histogram gradient boosting with ordinal-encoded categories."),
        CandidateModel("svc_rbf", svc, "Kernel SVM on scaled engineered features."),
        CandidateModel("soft_vote_core", vote_core, "Soft vote of logistic, extra trees, and gradient boosting."),
        CandidateModel("soft_vote_full", vote_full, "Soft vote of logistic, extra trees, gradient boosting, and SVM."),
    ]


def profile_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    title_counts = extract_title(train_df["Name"]).value_counts().to_dict()
    grouped = {}
    for column in ["Pclass", "Sex", "Embarked", "SibSp", "Parch"]:
        grouped[column] = (
            train_df.groupby(column, dropna=False)[TARGET]
            .agg(["count", "mean"])
            .reset_index()
            .to_dict(orient="records")
        )
    return {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_nulls": train_df.isna().sum().sort_values(ascending=False).to_dict(),
        "test_nulls": test_df.isna().sum().sort_values(ascending=False).to_dict(),
        "target_distribution": train_df[TARGET].value_counts(normalize=True).sort_index().to_dict(),
        "numeric_summary": train_df[["Age", "Fare"]].describe().round(4).to_dict(),
        "title_counts": title_counts,
        "group_survival_summary": grouped,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_serializable(payload), indent=2, sort_keys=True))


def extract_feature_importance(final_estimator: BaseEstimator) -> pd.DataFrame | None:
    if not isinstance(final_estimator, Pipeline):
        return None
    preprocess = final_estimator.named_steps["preprocess"]
    model = final_estimator.named_steps["model"]
    feature_names = preprocess.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(np.ravel(model.coef_))
    else:
        return None

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    return frame


def build_run_report(
    run_dir: Path,
    profile: dict[str, Any],
    cv_results: pd.DataFrame,
    selected_model: CandidateModel,
    holdout_accuracy: float,
    submission_path: Path,
) -> None:
    top_rows = cv_results.head(5).copy()
    lines = [
        "# Titanic Run Report",
        "",
        "## Data Profile",
        f"- Train shape: {tuple(profile['train_shape'])}",
        f"- Test shape: {tuple(profile['test_shape'])}",
        f"- Target distribution: {profile['target_distribution']}",
        f"- Highest missingness in train: {dict(list(profile['train_nulls'].items())[:3])}",
        "",
        "## Engineered Features",
        f"- Numeric features: {', '.join(NUMERIC_FEATURES)}",
        f"- Categorical features: {', '.join(CATEGORICAL_FEATURES)}",
        f"- Binary features: {', '.join(BINARY_FEATURES)}",
        "",
        "## Cross-Validation Ranking",
    ]
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"- {row.model_name}: mean={row.mean_accuracy:.5f}, std={row.std_accuracy:.5f}, "
            f"min={row.min_accuracy:.5f}, max={row.max_accuracy:.5f}"
        )
    lines.extend(
        [
            "",
            "## Final Selection",
            f"- Selected model: {selected_model.name}",
            f"- Selection note: {selected_model.notes}",
            f"- Training-set holdout accuracy on last fold split: {holdout_accuracy:.5f}",
            f"- Submission file: {submission_path}",
            f"- Artifact directory: {run_dir}",
        ]
    )
    (run_dir / "run_report.md").write_text("\n".join(lines))


def evaluate_models(
    candidates: list[CandidateModel],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for candidate in candidates:
        scores = cross_validate(
            candidate.estimator,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
            return_train_score=False,
        )["test_score"]
        records.append(
            {
                "model_name": candidate.name,
                "notes": candidate.notes,
                "mean_accuracy": float(scores.mean()),
                "std_accuracy": float(scores.std(ddof=0)),
                "min_accuracy": float(scores.min()),
                "max_accuracy": float(scores.max()),
                "fold_scores": [float(score) for score in scores],
            }
        )
    return pd.DataFrame(records).sort_values(
        by=["mean_accuracy", "std_accuracy", "max_accuracy"],
        ascending=[False, True, False],
    )


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Titanic models and generate Kaggle submission.")
    parser.add_argument("--train-path", default="data/train.csv")
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--submission-path", default="predictions.csv")
    parser.add_argument("--output-root", default="artifacts/runs")
    parser.add_argument("--n-splits", type=int, default=10)
    return parser


def main() -> None:
    args = make_argument_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].astype(int)

    profile = profile_dataset(train_df, test_df)
    write_json(run_dir / "data_profile.json", profile)

    feature_builder = TitanicFeatureBuilder().fit(X)
    engineered_profile = feature_builder.transform(X)
    engineered_summary = {
        "engineered_feature_count": len(engineered_profile.columns),
        "engineered_features": list(engineered_profile.columns),
        "engineered_nulls": engineered_profile.isna().sum().to_dict(),
        "preview_rows": engineered_profile.head(5).to_dict(orient="records"),
    }
    write_json(run_dir / "engineered_feature_summary.json", engineered_summary)

    candidates = build_candidate_models()
    candidate_configs = {
        candidate.name: {
            "notes": candidate.notes,
            "estimator_repr": repr(candidate.estimator),
        }
        for candidate in candidates
    }
    write_json(run_dir / "candidate_models.json", candidate_configs)

    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_results = evaluate_models(candidates, X, y, cv)
    cv_results_to_save = cv_results.copy()
    cv_results_to_save["fold_scores"] = cv_results_to_save["fold_scores"].map(json.dumps)
    cv_results_to_save.to_csv(run_dir / "cv_results.csv", index=False)

    best_model_name = cv_results.iloc[0]["model_name"]
    selected_model = next(candidate for candidate in candidates if candidate.name == best_model_name)

    splits = list(cv.split(X, y))
    train_idx, valid_idx = splits[-1]
    holdout_estimator = selected_model.estimator
    holdout_estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
    holdout_predictions = holdout_estimator.predict(X.iloc[valid_idx]).astype(int)
    holdout_accuracy = accuracy_score(y.iloc[valid_idx], holdout_predictions)

    final_estimator = selected_model.estimator
    final_estimator.fit(X, y)
    test_predictions = final_estimator.predict(test_df).astype(int)
    submission_df = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"].astype(int),
            "Survived": test_predictions,
        }
    )
    submission_path = Path(args.submission_path)
    submission_df.to_csv(submission_path, index=False)
    submission_df.to_csv(run_dir / "predictions.csv", index=False)

    joblib.dump(final_estimator, run_dir / "final_model.joblib")

    importance_df = extract_feature_importance(final_estimator)
    if importance_df is not None:
        importance_df.to_csv(run_dir / "feature_importance.csv", index=False)

    final_summary = {
        "selected_model": selected_model.name,
        "selection_note": selected_model.notes,
        "cross_validation": cv_results.iloc[0].to_dict(),
        "holdout_accuracy_last_fold": holdout_accuracy,
        "submission_path": str(submission_path.resolve()),
        "artifact_submission_path": str((run_dir / "predictions.csv").resolve()),
        "artifact_model_path": str((run_dir / "final_model.joblib").resolve()),
    }
    write_json(run_dir / "final_selection.json", final_summary)

    build_run_report(
        run_dir=run_dir,
        profile=profile,
        cv_results=cv_results,
        selected_model=selected_model,
        holdout_accuracy=holdout_accuracy,
        submission_path=submission_path.resolve(),
    )

    print(f"Run directory: {run_dir}")
    print(f"Selected model: {selected_model.name}")
    print(f"Mean CV accuracy: {cv_results.iloc[0]['mean_accuracy']:.5f}")
    print(f"Holdout accuracy on final fold: {holdout_accuracy:.5f}")
    print(f"Submission written to: {submission_path.resolve()}")


if __name__ == "__main__":
    main()
