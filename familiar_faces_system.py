"""Familiar Faces: governed care-transition outreach prioritization prototype.

This educational workflow uses the public, deidentified UCI Diabetes 130-US
Hospitals dataset. It predicts 30-day readmission as an advisory signal, audits
model behavior across protected groups, validates the audit with controlled
simulations, and can use an OpenAI model to draft nonclinical review material.

Nothing produced by this project is a diagnosis, treatment recommendation,
eligibility decision, or authorization to contact a patient.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEED = 20260728
RAW_DATA_PATH = Path("data/raw/diabetic_data.csv")
PROCESSED_DATA_PATH = Path("data/processed/familiar_faces_uci_encounters.csv")
DICTIONARY_PATH = Path("data/processed/data_dictionary.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

PATIENT_ID = "patient_nbr"
ENCOUNTER_ID = "encounter_id"
TARGET = "readmitted_within_30_days"

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult",
    "insulin",
    "change",
    "diabetesMed",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FAIRNESS_AUDIT_FIELDS = ["age", "gender", "race"]
PROHIBITED_MODEL_FIELDS = [
    PATIENT_ID,
    ENCOUNTER_ID,
    TARGET,
    *FAIRNESS_AUDIT_FIELDS,
]

SYSTEM_BOUNDARIES = {
    "deidentified_public_research_data_only": True,
    "human_review_required": True,
    "autonomous_contact_allowed": False,
    "diagnosis_allowed": False,
    "treatment_recommendation_allowed": False,
    "autonomous_eligibility_decision_allowed": False,
}

# Records discharged to hospice or recorded as expired are not meaningful
# candidates for a 30-day readmission/outreach workflow.
EXCLUDED_DISCHARGE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

MIN_GROUP_TOTAL = 500
MIN_CONDITION_DENOMINATOR = 200
PRACTICAL_DIFFERENCE = 0.10
FAMILYWISE_ALPHA = 0.05
AUDIT_SIMULATIONS = 200
INJECTED_DIFFERENCE = 0.15
LLM_CASE_COUNT = 5
DEFAULT_LLM_MODEL = "gpt-5.6"


def prepare_output_directories() -> None:
    """Create project output directories."""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def validate_system_boundaries() -> None:
    """Fail immediately if a required safety boundary is weakened."""
    required = {
        "deidentified_public_research_data_only": True,
        "human_review_required": True,
        "autonomous_contact_allowed": False,
        "diagnosis_allowed": False,
        "treatment_recommendation_allowed": False,
        "autonomous_eligibility_decision_allowed": False,
    }
    if SYSTEM_BOUNDARIES != required:
        raise ValueError("Required system safety boundaries have changed.")

    overlap = set(MODEL_FEATURES) & set(PROHIBITED_MODEL_FIELDS)
    if overlap:
        raise ValueError(f"Prohibited model fields detected: {sorted(overlap)}")


def _normalize_race(value: str) -> str:
    value = str(value).strip()
    return "Unknown/Not recorded" if value in {"?", "", "nan"} else value


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UCI encounters and create a reproducible analytic cohort."""
    validate_system_boundaries()
    prepare_output_directories()

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")

    raw = pd.read_csv(RAW_DATA_PATH)
    required = {
        PATIENT_ID,
        ENCOUNTER_ID,
        "readmitted",
        *MODEL_FEATURES,
        *FAIRNESS_AUDIT_FIELDS,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Required UCI fields are missing: {missing}")

    data = raw.loc[
        ~raw["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_DISPOSITIONS)
    ].copy()
    data = data.sort_values(ENCOUNTER_ID).drop_duplicates(
        PATIENT_ID, keep="first"
    )
    data[TARGET] = (data["readmitted"] == "<30").astype(int)
    data["race"] = data["race"].map(_normalize_race)
    data["gender"] = data["gender"].replace(
        {"Unknown/Invalid": "Unknown/Not recorded"}
    )

    selected_columns = [
        ENCOUNTER_ID,
        PATIENT_ID,
        *FAIRNESS_AUDIT_FIELDS,
        *MODEL_FEATURES,
        TARGET,
    ]
    data = data[selected_columns].reset_index(drop=True)

    if not data[PATIENT_ID].is_unique:
        raise ValueError("Cohort must contain one encounter per patient.")
    if set(data[TARGET].unique()) - {0, 1}:
        raise ValueError("Readmission target must be binary.")
    if data[NUMERIC_FEATURES].isna().any().any():
        raise ValueError("Selected numeric model features contain nulls.")

    dictionary_rows = [
        {
            "field_name": column,
            "role": (
                "identifier"
                if column in {PATIENT_ID, ENCOUNTER_ID}
                else "fairness_audit_only"
                if column in FAIRNESS_AUDIT_FIELDS
                else "outcome"
                if column == TARGET
                else "model_feature"
            ),
            "description": {
                ENCOUNTER_ID: "Deidentified hospital encounter identifier.",
                PATIENT_ID: "Deidentified patient identifier; used only to prevent repeat-patient leakage.",
                "age": "Age interval retained only for post-model fairness auditing.",
                "gender": "Recorded gender retained only for post-model fairness auditing.",
                "race": "Recorded race retained only for post-model fairness auditing.",
                TARGET: "One when the recorded readmission category is less than 30 days.",
            }.get(column, f"UCI encounter field used as a {column.replace('_', ' ')} predictor."),
        }
        for column in selected_columns
    ]
    dictionary = pd.DataFrame(dictionary_rows)

    data.to_csv(PROCESSED_DATA_PATH, index=False)
    dictionary.to_csv(DICTIONARY_PATH, index=False)
    return data, dictionary


def run_statistical_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize the real analytic cohort and prior utilization."""
    prior_emergency_band = pd.cut(
        df["number_emergency"],
        bins=[-1, 0, 1, 2, float("inf")],
        labels=["0", "1", "2", "3+"],
    )
    band_results = (
        df.assign(prior_emergency_band=prior_emergency_band)
        .groupby("prior_emergency_band", observed=True)
        .agg(
            encounters=(ENCOUNTER_ID, "count"),
            readmission_rate=(TARGET, "mean"),
            mean_prior_inpatient=("number_inpatient", "mean"),
        )
        .reset_index()
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "raw_encounters",
                "eligible_unique_patient_encounters",
                "excluded_or_repeat_encounters",
                "30_day_readmission_rate",
                "mean_prior_emergency_visits",
                "mean_prior_inpatient_visits",
                "mean_prior_outpatient_visits",
                "median_time_in_hospital_days",
            ],
            "value": [
                len(pd.read_csv(RAW_DATA_PATH, usecols=[ENCOUNTER_ID])),
                len(df),
                len(pd.read_csv(RAW_DATA_PATH, usecols=[ENCOUNTER_ID])) - len(df),
                df[TARGET].mean(),
                df["number_emergency"].mean(),
                df["number_inpatient"].mean(),
                df["number_outpatient"].mean(),
                df["time_in_hospital"].median(),
            ],
        }
    )
    summary.to_csv(RESULTS_DIR / "statistical_summary.csv", index=False)
    band_results.to_csv(
        RESULTS_DIR / "readmission_by_prior_emergency_use.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    rates = band_results["readmission_rate"] * 100
    ax.bar(
        band_results["prior_emergency_band"].astype(str),
        rates,
        color="#1f4e79",
    )
    ax.set_title("30-Day Readmission by Prior Emergency Use")
    ax.set_xlabel("Prior emergency visits in preceding year")
    ax.set_ylabel("30-day readmission (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(20, rates.max() * 1.2))
    for index, value in enumerate(rates):
        ax.text(index, value + 0.4, f"{value:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "readmission_by_prior_emergency_use.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary, band_results


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def run_machine_learning(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, str, Pipeline, pd.DataFrame, pd.DataFrame]:
    """Compare models on one patient-independent held-out partition."""
    splitter = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=SEED
    )
    train_indices, test_indices = next(
        splitter.split(df, df[TARGET], groups=df[PATIENT_ID])
    )
    train = df.iloc[train_indices].copy()
    test = df.iloc[test_indices].copy()

    if set(train[PATIENT_ID]) & set(test[PATIENT_ID]):
        raise ValueError("Patient leakage detected across train and test sets.")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    trained_models: dict[str, Pipeline] = {}
    comparison_rows = []
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", _make_preprocessor()), ("model", estimator)]
        )
        pipeline.fit(train[MODEL_FEATURES], train[TARGET])
        probabilities = pipeline.predict_proba(test[MODEL_FEATURES])[:, 1]
        predictions = (probabilities >= 0.50).astype(int)
        trained_models[model_name] = pipeline
        comparison_rows.append(
            {
                "model": model_name,
                "train_records": len(train),
                "test_records": len(test),
                "test_positive_rate": test[TARGET].mean(),
                "roc_auc": roc_auc_score(test[TARGET], probabilities),
                "average_precision": average_precision_score(
                    test[TARGET], probabilities
                ),
                "accuracy": accuracy_score(test[TARGET], predictions),
                "precision": precision_score(
                    test[TARGET], predictions, zero_division=0
                ),
                "recall": recall_score(
                    test[TARGET], predictions, zero_division=0
                ),
                "f1": f1_score(test[TARGET], predictions, zero_division=0),
                "positive_prediction_rate": predictions.mean(),
                "patient_overlap": 0,
            }
        )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values(["average_precision", "roc_auc"], ascending=False)
        .reset_index(drop=True)
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    selected_name = str(comparison.loc[0, "model"])
    selected_model = trained_models[selected_name]
    selected_probabilities = selected_model.predict_proba(
        test[MODEL_FEATURES]
    )[:, 1]
    selected_predictions = (selected_probabilities >= 0.50).astype(int)

    test_predictions = test[
        [
            ENCOUNTER_ID,
            PATIENT_ID,
            *FAIRNESS_AUDIT_FIELDS,
            *MODEL_FEATURES,
            TARGET,
        ]
    ].copy()
    test_predictions["predicted_readmission_probability"] = (
        selected_probabilities
    )
    test_predictions["model_flag_for_human_review"] = selected_predictions
    test_predictions["selected_model"] = selected_name
    test_predictions.to_csv(
        RESULTS_DIR / "test_predictions.csv", index=False
    )

    transformed_names = selected_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    estimator = selected_model.named_steps["model"]
    if selected_name == "logistic_regression":
        importance_values = np.abs(estimator.coef_[0])
    else:
        importance_values = estimator.feature_importances_
    feature_importance = (
        pd.DataFrame(
            {
                "feature": [
                    name.replace("numeric__", "").replace(
                        "categorical__", ""
                    )
                    for name in transformed_names
                ],
                "importance": importance_values,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(
        RESULTS_DIR / "selected_model_feature_importance.csv", index=False
    )

    top = feature_importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["importance"], color="#b7472a")
    ax.set_title(f"Top Features: {selected_name.replace('_', ' ').title()}")
    ax.set_xlabel("Model-specific absolute importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "selected_model_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)
    return (
        comparison,
        selected_name,
        selected_model,
        test_predictions,
        feature_importance,
    )


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = norm.ppf(0.975)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        / denominator
    )
    return center - margin, center + margin


def _two_proportion_pvalue(
    successes_a: int, total_a: int, successes_b: int, total_b: int
) -> float:
    if min(total_a, total_b) == 0:
        return float("nan")
    pooled = (successes_a + successes_b) / (total_a + total_b)
    standard_error = np.sqrt(
        pooled * (1 - pooled) * (1 / total_a + 1 / total_b)
    )
    if standard_error == 0:
        return 1.0
    z_score = (
        successes_a / total_a - successes_b / total_b
    ) / standard_error
    return float(2 * norm.sf(abs(z_score)))


def _holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Return Holm-adjusted p-values while preserving original row order."""
    adjusted = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    running_max = 0.0
    total = len(valid)
    for rank, (index, pvalue) in enumerate(valid.items()):
        candidate = min(1.0, (total - rank) * pvalue)
        running_max = max(running_max, candidate)
        adjusted.loc[index] = running_max
    return adjusted


def run_fairness_audit(
    test_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Audit group-to-reference gaps with uncertainty and multiplicity control."""
    audit_rows = []
    comparison_rows = []
    metric_definitions = {
        "human_review_flag_rate": lambda frame: (
            int(frame["model_flag_for_human_review"].sum()),
            len(frame),
        ),
        "true_positive_rate": lambda frame: (
            int(
                (
                    (frame[TARGET] == 1)
                    & (frame["model_flag_for_human_review"] == 1)
                ).sum()
            ),
            int((frame[TARGET] == 1).sum()),
        ),
        "false_positive_rate": lambda frame: (
            int(
                (
                    (frame[TARGET] == 0)
                    & (frame["model_flag_for_human_review"] == 1)
                ).sum()
            ),
            int((frame[TARGET] == 0).sum()),
        ),
    }

    for field in FAIRNESS_AUDIT_FIELDS:
        groups = {
            str(name): group.copy()
            for name, group in test_predictions.groupby(field, dropna=False)
        }
        reference_name = max(groups, key=lambda name: len(groups[name]))

        for group_name, group in groups.items():
            for metric, counter in metric_definitions.items():
                successes, denominator = counter(group)
                rate = (
                    successes / denominator
                    if denominator
                    else float("nan")
                )
                ci_low, ci_high = _wilson_interval(successes, denominator)
                eligible = (
                    len(group) >= MIN_GROUP_TOTAL
                    and denominator >= MIN_CONDITION_DENOMINATOR
                )
                audit_rows.append(
                    {
                        "audit_field": field,
                        "group": group_name,
                        "reference_group": reference_name,
                        "metric": metric,
                        "records": len(group),
                        "successes": successes,
                        "denominator": denominator,
                        "rate": rate,
                        "ci_95_low": ci_low,
                        "ci_95_high": ci_high,
                        "eligible_for_comparison": eligible,
                    }
                )

        reference = groups[reference_name]
        for group_name, group in groups.items():
            if group_name == reference_name:
                continue
            for metric, counter in metric_definitions.items():
                group_successes, group_total = counter(group)
                ref_successes, ref_total = counter(reference)
                eligible = (
                    len(group) >= MIN_GROUP_TOTAL
                    and len(reference) >= MIN_GROUP_TOTAL
                    and group_total >= MIN_CONDITION_DENOMINATOR
                    and ref_total >= MIN_CONDITION_DENOMINATOR
                )
                gap = (
                    group_successes / group_total
                    - ref_successes / ref_total
                    if eligible
                    else float("nan")
                )
                comparison_rows.append(
                    {
                        "audit_field": field,
                        "group": group_name,
                        "reference_group": reference_name,
                        "metric": metric,
                        "group_denominator": group_total,
                        "reference_denominator": ref_total,
                        "rate_difference": gap,
                        "absolute_difference": (
                            abs(gap) if eligible else float("nan")
                        ),
                        "raw_p_value": (
                            _two_proportion_pvalue(
                                group_successes,
                                group_total,
                                ref_successes,
                                ref_total,
                            )
                            if eligible
                            else float("nan")
                        ),
                        "eligible_for_comparison": eligible,
                    }
                )

    audit = pd.DataFrame(audit_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["holm_adjusted_p_value"] = _holm_adjust(
        comparisons["raw_p_value"]
    )
    comparisons["practically_large"] = (
        comparisons["absolute_difference"] >= PRACTICAL_DIFFERENCE
    )
    comparisons["statistically_significant"] = (
        comparisons["holm_adjusted_p_value"] < FAMILYWISE_ALPHA
    )
    comparisons["governance_action"] = np.select(
        [
            ~comparisons["eligible_for_comparison"],
            comparisons["practically_large"]
            & comparisons["statistically_significant"],
        ],
        ["INSUFFICIENT_DATA", "REVIEW_REQUIRED"],
        default="MONITOR",
    )
    status = (
        "REVIEW_REQUIRED"
        if (comparisons["governance_action"] == "REVIEW_REQUIRED").any()
        else "MONITOR"
    )
    audit.to_csv(RESULTS_DIR / "fairness_audit.csv", index=False)
    comparisons.to_csv(
        RESULTS_DIR / "fairness_group_comparisons.csv", index=False
    )
    return audit, comparisons, status


def validate_fairness_method(
    test_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate familywise false alarms and power under known disparity."""
    rng = np.random.default_rng(SEED)
    observed = test_predictions["model_flag_for_human_review"].to_numpy(
        dtype=int
    )
    actual = test_predictions[TARGET].to_numpy(dtype=int)
    total = len(observed)
    null_triggers = 0
    injected_triggers = 0
    realized_differences = []

    def comparison_family(
        flags: np.ndarray, group_assignments: list[np.ndarray]
    ) -> pd.DataFrame:
        rows = []
        masks = {
            "human_review_flag_rate": np.ones(total, dtype=bool),
            "true_positive_rate": actual == 1,
            "false_positive_rate": actual == 0,
        }
        for field_number, assignment in enumerate(group_assignments, start=1):
            for metric, metric_mask in masks.items():
                a_mask = (assignment == 0) & metric_mask
                b_mask = (assignment == 1) & metric_mask
                a_total = int(a_mask.sum())
                b_total = int(b_mask.sum())
                a_successes = int(flags[a_mask].sum())
                b_successes = int(flags[b_mask].sum())
                eligible = min(a_total, b_total) >= MIN_CONDITION_DENOMINATOR
                gap = (
                    b_successes / b_total - a_successes / a_total
                    if eligible
                    else float("nan")
                )
                rows.append(
                    {
                        "field": field_number,
                        "metric": metric,
                        "gap": gap,
                        "pvalue": (
                            _two_proportion_pvalue(
                                b_successes,
                                b_total,
                                a_successes,
                                a_total,
                            )
                            if eligible
                            else float("nan")
                        ),
                    }
                )
        family = pd.DataFrame(rows)
        family["adjusted_pvalue"] = _holm_adjust(family["pvalue"])
        family["trigger"] = (
            family["gap"].abs() >= PRACTICAL_DIFFERENCE
        ) & (family["adjusted_pvalue"] < FAMILYWISE_ALPHA)
        return family

    for _ in range(AUDIT_SIMULATIONS):
        simulated_groups = [
            rng.integers(0, 2, size=total) for _ in FAIRNESS_AUDIT_FIELDS
        ]
        null_family = comparison_family(observed, simulated_groups)
        if null_family["trigger"].any():
            null_triggers += 1

        injected = observed.copy()
        injected_group = simulated_groups[0]
        b_indices = np.flatnonzero(injected_group == 1)
        b_zero_indices = b_indices[injected[b_indices] == 0]
        target_flips = min(
            len(b_zero_indices),
            int(np.ceil(INJECTED_DIFFERENCE * len(b_indices))),
        )
        if target_flips:
            flip_indices = rng.choice(
                b_zero_indices, size=target_flips, replace=False
            )
            injected[flip_indices] = 1

        injected_a = injected[injected_group == 0]
        injected_b = injected[injected_group == 1]
        injected_gap = injected_b.mean() - injected_a.mean()
        realized_differences.append(injected_gap)
        injected_family = comparison_family(
            injected, simulated_groups
        )
        target_row = injected_family[
            (injected_family["field"] == 1)
            & (
                injected_family["metric"]
                == "human_review_flag_rate"
            )
        ]
        if bool(target_row["trigger"].iloc[0]):
            injected_triggers += 1

    results = pd.DataFrame(
        [
            {
                "validation_scenario": "independent_null_group",
                "simulations": AUDIT_SIMULATIONS,
                "target_difference": 0.0,
                "mean_realized_difference": 0.0,
                "trigger_rate": null_triggers / AUDIT_SIMULATIONS,
                "interpretation": (
                    "Estimated familywise false-positive rate across three "
                    "independent audit fields and three metrics per field."
                ),
            },
            {
                "validation_scenario": "injected_flag_rate_disparity",
                "simulations": AUDIT_SIMULATIONS,
                "target_difference": INJECTED_DIFFERENCE,
                "mean_realized_difference": float(
                    np.mean(realized_differences)
                ),
                "trigger_rate": injected_triggers / AUDIT_SIMULATIONS,
                "interpretation": "Estimated detection rate for a known disparity.",
            },
        ]
    )
    results.to_csv(
        RESULTS_DIR / "fairness_method_validation.csv", index=False
    )
    return results


PROHIBITED_GENERATION_PATTERNS = {
    "diagnosis": r"\bdiagnos(?:e|is|ed|ing)\b",
    "prescribing": r"\bprescrib(?:e|es|ed|ing)\b",
    "dose_change": r"\b(?:increase|decrease|change|stop)\b.{0,25}\bdose\b",
    "autonomous_eligibility": r"\bautomatically eligible\b",
    "autonomous_contact": r"\bautomatically contact\b",
    "protected_demographic": (
        r"\b(?:race|racial|gender|female|male|caucasian|"
        r"african american|hispanic|asian|age group)\b"
    ),
}


def validate_generated_material(material: dict) -> list[str]:
    """Apply deterministic safety checks after LLM inference."""
    combined = " ".join(
        [
            str(material.get("case_summary", "")),
            " ".join(material.get("review_questions", [])),
            " ".join(material.get("coordination_options", [])),
            str(material.get("limitations", "")),
            str(material.get("human_review_notice", "")),
        ]
    ).lower()
    violations = [
        label
        for label, pattern in PROHIBITED_GENERATION_PATTERNS.items()
        if re.search(pattern, combined, flags=re.IGNORECASE)
    ]
    for required in [
        "deidentified research record",
        "human review required",
        "not a clinical recommendation",
    ]:
        if required not in combined:
            violations.append(f"missing required phrase: {required}")
    return violations


def _llm_input_records(test_predictions: pd.DataFrame) -> list[dict]:
    selected = (
        test_predictions.sort_values(
            "predicted_readmission_probability", ascending=False
        )
        .head(LLM_CASE_COUNT)
        .reset_index(drop=True)
    )
    records = []
    for index, row in selected.iterrows():
        records.append(
            {
                "case_id": f"CASE-{index + 1:03d}",
                "prior_emergency_visits": int(row["number_emergency"]),
                "prior_inpatient_visits": int(row["number_inpatient"]),
                "prior_outpatient_visits": int(row["number_outpatient"]),
                "time_in_hospital_days": int(row["time_in_hospital"]),
                "medication_count": int(row["num_medications"]),
                "procedure_count": int(row["num_procedures"]),
                "diagnosis_count": int(row["number_diagnoses"]),
                "model_probability": round(
                    float(row["predicted_readmission_probability"]), 4
                ),
            }
        )
    return records


def generate_llm_case_materials(
    test_predictions: pd.DataFrame,
    model: str = DEFAULT_LLM_MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute genuine structured LLM drafting on demographics-free inputs."""
    prepare_output_directories()
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Configure it in the current "
            "PowerShell window before using --generate-llm."
        )

    from openai import OpenAI
    from pydantic import BaseModel, Field

    class CaseDraft(BaseModel):
        case_id: str
        case_summary: str
        review_questions: list[str] = Field(min_length=3, max_length=3)
        coordination_options: list[str] = Field(min_length=2, max_length=3)
        limitations: str
        human_review_notice: str

    class DraftBatch(BaseModel):
        drafts: list[CaseDraft]

    DraftBatch.model_rebuild(_types_namespace={"CaseDraft": CaseDraft})

    records = _llm_input_records(test_predictions)
    prompt = (
        "Create one nonclinical care-transition review draft for each input "
        "record. Use only the supplied facts. Do not infer diagnoses, causes, "
        "social needs, protected demographics, eligibility, consent, or patient "
        "preferences. Do not recommend treatment, medication changes, or "
        "autonomous contact. Each case_summary must call the input a "
        "'deidentified research record'. Each limitations field must contain "
        "'not a clinical recommendation'. Each human_review_notice must contain "
        "'human review required'. Review questions must tell an authorized "
        "reviewer what to verify; coordination options must remain conditional "
        "and nonclinical. Preserve every case_id exactly.\n\nINPUT RECORDS:\n"
        + json.dumps(records, indent=2)
    )

    client = OpenAI()
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You draft constrained educational case-review material "
                    "for a deidentified, nonclinical research prototype."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text_format=DraftBatch,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise RuntimeError("The model did not return a parsed structured output.")
    if len(parsed.drafts) != len(records):
        raise RuntimeError("The model returned an unexpected number of drafts.")

    input_by_id = {record["case_id"]: record for record in records}
    generated_rows = []
    for draft in parsed.drafts:
        material = draft.model_dump()
        if material["case_id"] not in input_by_id:
            raise ValueError(f"Unexpected case_id: {material['case_id']}")
        violations = validate_generated_material(material)
        generated_rows.append(
            {
                **input_by_id[material["case_id"]],
                **material,
                "generation_mode": "openai_responses_api_structured_output",
                "model": model,
                "demographics_included": False,
                "human_review_required": True,
                "safety_validation_passed": len(violations) == 0,
                "safety_violations": (
                    "NONE" if not violations else " | ".join(violations)
                ),
            }
        )

    generated = pd.DataFrame(generated_rows)
    generated.to_csv(RESULTS_DIR / "llm_case_materials.csv", index=False)
    metadata = pd.DataFrame(
        [
            {
                "response_id": response.id,
                "model_requested": model,
                "drafts_requested": len(records),
                "drafts_returned": len(generated),
                "drafts_passing_safety_validation": int(
                    generated["safety_validation_passed"].sum()
                ),
                "demographic_fields_sent": False,
                "human_review_required": True,
            }
        ]
    )
    metadata.to_csv(RESULTS_DIR / "llm_generation_metadata.csv", index=False)

    if not generated["safety_validation_passed"].all():
        raise ValueError(
            "One or more LLM drafts failed deterministic safety validation."
        )
    return generated, metadata


def run_integrated_system(
    generate_llm: bool = False,
    model: str = DEFAULT_LLM_MODEL,
) -> dict:
    """Execute the analytic system and optionally the real LLM component."""
    df, dictionary = load_and_prepare_data()
    statistical_summary, utilization_results = run_statistical_analysis(df)
    (
        model_comparison,
        selected_model_name,
        selected_model,
        test_predictions,
        feature_importance,
    ) = run_machine_learning(df)
    fairness_audit, fairness_comparisons, fairness_status = (
        run_fairness_audit(test_predictions)
    )
    fairness_validation = validate_fairness_method(test_predictions)

    generated_materials = None
    generation_metadata = None
    if generate_llm:
        generated_materials, generation_metadata = (
            generate_llm_case_materials(test_predictions, model=model)
        )

    selected_metrics = model_comparison.iloc[0]
    null_rate = fairness_validation.loc[
        fairness_validation["validation_scenario"]
        == "independent_null_group",
        "trigger_rate",
    ].iloc[0]
    detection_rate = fairness_validation.loc[
        fairness_validation["validation_scenario"]
        == "injected_flag_rate_disparity",
        "trigger_rate",
    ].iloc[0]
    integrated_summary = pd.DataFrame(
        {
            "system_metric": [
                "real_deidentified_encounters",
                "unique_patients",
                "model_features",
                "protected_demographics_used_as_model_features",
                "selected_model",
                "test_records",
                "selected_model_roc_auc",
                "selected_model_average_precision",
                "fairness_governance_status",
                "fairness_comparisons_requiring_review",
                "fairness_null_false_positive_rate",
                "fairness_injected_disparity_detection_rate",
                "genuine_llm_inference_executed",
                "llm_drafts_generated",
                "llm_drafts_passing_safety_validation",
                "human_review_required",
                "autonomous_contact_allowed",
                "deployment_status",
            ],
            "value": [
                len(df),
                df[PATIENT_ID].nunique(),
                len(MODEL_FEATURES),
                False,
                selected_model_name,
                len(test_predictions),
                selected_metrics["roc_auc"],
                selected_metrics["average_precision"],
                fairness_status,
                int(
                    (
                        fairness_comparisons["governance_action"]
                        == "REVIEW_REQUIRED"
                    ).sum()
                ),
                null_rate,
                detection_rate,
                generate_llm,
                0 if generated_materials is None else len(generated_materials),
                (
                    0
                    if generated_materials is None
                    else int(
                        generated_materials[
                            "safety_validation_passed"
                        ].sum()
                    )
                ),
                True,
                False,
                "EDUCATIONAL DEIDENTIFIED-DATA RESEARCH PROTOTYPE",
            ],
        }
    )
    integrated_summary.to_csv(
        RESULTS_DIR / "integrated_system_summary.csv", index=False
    )
    return {
        "data": df,
        "dictionary": dictionary,
        "statistical_summary": statistical_summary,
        "utilization_results": utilization_results,
        "model_comparison": model_comparison,
        "selected_model_name": selected_model_name,
        "selected_model": selected_model,
        "test_predictions": test_predictions,
        "feature_importance": feature_importance,
        "fairness_audit": fairness_audit,
        "fairness_comparisons": fairness_comparisons,
        "fairness_status": fairness_status,
        "fairness_validation": fairness_validation,
        "generated_materials": generated_materials,
        "generation_metadata": generation_metadata,
        "integrated_summary": integrated_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generate-llm",
        action="store_true",
        help="Execute the OpenAI drafting component using OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_LLM_MODEL),
        help="OpenAI model used only when --generate-llm is supplied.",
    )
    args = parser.parse_args()
    outputs = run_integrated_system(
        generate_llm=args.generate_llm, model=args.model
    )
    print("Familiar Faces integrated AI system")
    print("===================================")
    print(outputs["integrated_summary"].to_string(index=False))
    print()
    print("End-to-end analytic pipeline: PASSED")
    print(
        "LLM component:",
        "EXECUTED AND VALIDATED" if args.generate_llm else "NOT REQUESTED",
    )
    print("Human review required: True")


if __name__ == "__main__":
    main()
