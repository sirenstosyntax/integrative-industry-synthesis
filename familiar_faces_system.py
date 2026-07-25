"""Core pipeline for the Familiar Faces integrative AI system.

This educational prototype processes only synthetic records. Its outputs are
advisory drafts that require human review.
"""

from pathlib import Path

SEED = 20260724

DATA_PATH = Path("data/familiar_faces_synthetic.csv")
DICTIONARY_PATH = Path("data/data_dictionary.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

PATIENT_ID = "synthetic_patient_id"
TARGET = "outreach_response_90d"

MODEL_FEATURES = [
    "housing_instability",
    "transportation_barrier",
    "food_insecurity",
    "medication_access_barrier",
    "behavioral_health_support_need",
    "substance_use_support_need",
    "primary_care_connected",
    "chronic_condition_count",
    "ems_calls_12m",
    "ed_visits_12m",
    "hospital_admissions_12m",
    "missed_appointments_12m",
    "days_since_last_911_call",
    "prior_outreach_attempts",
    "prior_outreach_engaged",
    "social_need_count",
]

FAIRNESS_AUDIT_FIELDS = [
    "age_group",
    "sex",
    "race_ethnicity",
]

PROHIBITED_MODEL_FIELDS = [
    PATIENT_ID,
    "age_group",
    "sex",
    "race_ethnicity",
    "data_source",
]

SYSTEM_BOUNDARIES = {
    "synthetic_data_only": True,
    "human_review_required": True,
    "autonomous_contact_allowed": False,
    "diagnosis_allowed": False,
    "treatment_recommendation_allowed": False,
    "autonomous_eligibility_decision_allowed": False,
}

import pandas as pd


def prepare_output_directories():
    """Create folders used by later pipeline stages."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def validate_system_boundaries():
    """Fail immediately if a required safety boundary is weakened."""
    required_boundaries = {
        "synthetic_data_only": True,
        "human_review_required": True,
        "autonomous_contact_allowed": False,
        "diagnosis_allowed": False,
        "treatment_recommendation_allowed": False,
        "autonomous_eligibility_decision_allowed": False,
    }

    if SYSTEM_BOUNDARIES != required_boundaries:
        raise ValueError("Required system safety boundaries have changed.")

    prohibited_overlap = set(MODEL_FEATURES) & set(PROHIBITED_MODEL_FIELDS)
    if prohibited_overlap:
        raise ValueError(
            f"Prohibited fields found in model features: {sorted(prohibited_overlap)}"
        )


def load_and_validate_data():
    """Load the synthetic records and enforce schema and safety requirements."""
    validate_system_boundaries()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not DICTIONARY_PATH.exists():
        raise FileNotFoundError(f"Data dictionary not found: {DICTIONARY_PATH}")

    df = pd.read_csv(DATA_PATH)
    dictionary = pd.read_csv(DICTIONARY_PATH)

    expected_columns = dictionary["field_name"].tolist()

    if df.columns.tolist() != expected_columns:
        raise ValueError("Dataset columns do not exactly match the data dictionary.")

    if not df[PATIENT_ID].is_unique:
        raise ValueError("Synthetic patient identifiers must be unique.")

    if df.isna().any().any():
        raise ValueError("The dataset contains unexpected missing values.")

    if set(df["data_source"].unique()) != {"synthetic"}:
        raise ValueError("The pipeline accepts only explicitly synthetic data.")

    if not set(MODEL_FEATURES).issubset(df.columns):
        raise ValueError("One or more required model features are missing.")

    if set(df[TARGET].unique()) - {0, 1}:
        raise ValueError("The target must contain only binary values.")

    prepare_output_directories()
    return df, dictionary

import matplotlib.pyplot as plt


def run_statistical_analysis(df):
    """Summarize utilization, support needs, and synthetic outcomes."""
    frequent_utilizer = df["ems_calls_12m"] >= 4
    high_social_need = df["social_need_count"] >= 3

    summary = pd.DataFrame(
        {
            "metric": [
                "synthetic_records",
                "outreach_response_rate",
                "mean_ems_calls_12m",
                "median_ems_calls_12m",
                "frequent_utilizer_rate_4plus",
                "frequent_utilizer_response_rate",
                "high_social_need_rate_3plus",
                "high_social_need_response_rate",
                "primary_care_connected_rate",
            ],
            "value": [
                len(df),
                df[TARGET].mean(),
                df["ems_calls_12m"].mean(),
                df["ems_calls_12m"].median(),
                frequent_utilizer.mean(),
                df.loc[frequent_utilizer, TARGET].mean(),
                high_social_need.mean(),
                df.loc[high_social_need, TARGET].mean(),
                df["primary_care_connected"].mean(),
            ],
        }
    )

    utilization_band = pd.cut(
        df["ems_calls_12m"],
        bins=[-1, 0, 3, 6, float("inf")],
        labels=["0", "1-3", "4-6", "7+"],
    )

    utilization_results = (
        df.assign(utilization_band=utilization_band)
        .groupby("utilization_band", observed=True)
        .agg(
            synthetic_records=(PATIENT_ID, "count"),
            mean_social_needs=("social_need_count", "mean"),
            outreach_response_rate=(TARGET, "mean"),
        )
        .reset_index()
    )

    summary.to_csv(RESULTS_DIR / "statistical_summary.csv", index=False)
    utilization_results.to_csv(
        RESULTS_DIR / "utilization_band_analysis.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        utilization_results["utilization_band"].astype(str),
        utilization_results["outreach_response_rate"] * 100,
        color="#1f4e79",
    )
    ax.set_title("Synthetic Outreach Response by EMS Utilization")
    ax.set_xlabel("EMS calls during preceding 12 months")
    ax.set_ylabel("Beneficial outreach response (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)

    for index, value in enumerate(
        utilization_results["outreach_response_rate"] * 100
    ):
        ax.text(index, value + 1.5, f"{value:.1f}%", ha="center")

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "outreach_response_by_utilization.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return summary, utilization_results

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_machine_learning(df):
    """Train and compare two outreach-response models."""
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.25,
        random_state=SEED,
        stratify=df[TARGET],
    )

    X_train = df.loc[train_indices, MODEL_FEATURES]
    X_test = df.loc[test_indices, MODEL_FEATURES]
    y_train = df.loc[train_indices, TARGET]
    y_test = df.loc[test_indices, TARGET]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    trained_models = {}
    comparison_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        trained_models[model_name] = model
        comparison_rows.append(
            {
                "model": model_name,
                "test_records": len(y_test),
                "roc_auc": roc_auc_score(y_test, probabilities),
                "average_precision": average_precision_score(
                    y_test, probabilities
                ),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
                "positive_prediction_rate": predictions.mean(),
            }
        )

    comparison = (
        pd.DataFrame(comparison_rows)
        .sort_values("roc_auc", ascending=False)
        .reset_index(drop=True)
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    selected_model_name = comparison.loc[0, "model"]
    selected_model = trained_models[selected_model_name]
    selected_probabilities = selected_model.predict_proba(X_test)[:, 1]
    selected_predictions = (selected_probabilities >= 0.50).astype(int)

    test_predictions = df.loc[
        test_indices,
        [PATIENT_ID, *FAIRNESS_AUDIT_FIELDS, TARGET],
    ].copy()
    test_predictions["predicted_response_probability"] = (
        selected_probabilities
    )
    test_predictions["model_flag_for_human_review"] = selected_predictions
    test_predictions["selected_model"] = selected_model_name
    test_predictions.to_csv(
        RESULTS_DIR / "test_predictions.csv",
        index=False,
    )

    if selected_model_name == "logistic_regression":
        importance_values = abs(
            selected_model.named_steps["model"].coef_[0]
        )
    else:
        importance_values = selected_model.feature_importances_

    feature_importance = (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance": importance_values,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(
        RESULTS_DIR / "selected_model_feature_importance.csv",
        index=False,
    )

    top_features = feature_importance.head(10).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top_features["feature"],
        top_features["importance"],
        color="#b7472a",
    )
    ax.set_title(
        f"Top Features: {selected_model_name.replace('_', ' ').title()}"
    )
    ax.set_xlabel("Model-specific importance")
    ax.set_ylabel("")
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
        selected_model_name,
        selected_model,
        test_predictions,
        feature_importance,
    )

MIN_AUDIT_GROUP_SIZE = 25


def safe_rate(numerator, denominator):
    """Return a rate or NaN when its denominator is zero."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def run_fairness_audit(test_predictions):
    """Audit model behavior across protected demographic groups."""
    audit_rows = []

    for audit_field in FAIRNESS_AUDIT_FIELDS:
        for group_name, group in test_predictions.groupby(
            audit_field,
            dropna=False,
        ):
            actual = group[TARGET].astype(int)
            predicted = group[
                "model_flag_for_human_review"
            ].astype(int)

            true_positive = ((actual == 1) & (predicted == 1)).sum()
            false_positive = ((actual == 0) & (predicted == 1)).sum()
            true_negative = ((actual == 0) & (predicted == 0)).sum()
            false_negative = ((actual == 1) & (predicted == 0)).sum()

            audit_rows.append(
                {
                    "audit_field": audit_field,
                    "group": group_name,
                    "records": len(group),
                    "actual_response_rate": actual.mean(),
                    "mean_predicted_probability": group[
                        "predicted_response_probability"
                    ].mean(),
                    "human_review_flag_rate": predicted.mean(),
                    "true_positive_rate": safe_rate(
                        true_positive,
                        true_positive + false_negative,
                    ),
                    "false_positive_rate": safe_rate(
                        false_positive,
                        false_positive + true_negative,
                    ),
                    "precision": safe_rate(
                        true_positive,
                        true_positive + false_positive,
                    ),
                    "included_in_disparity_range": (
                        len(group) >= MIN_AUDIT_GROUP_SIZE
                    ),
                }
            )

    fairness_audit = pd.DataFrame(audit_rows)

    disparity_rows = []

    for audit_field in FAIRNESS_AUDIT_FIELDS:
        eligible = fairness_audit[
            (fairness_audit["audit_field"] == audit_field)
            & fairness_audit["included_in_disparity_range"]
        ]

        for metric in [
            "human_review_flag_rate",
            "true_positive_rate",
            "false_positive_rate",
        ]:
            values = eligible[metric].dropna()

            disparity_rows.append(
                {
                    "audit_field": audit_field,
                    "metric": metric,
                    "eligible_groups": len(values),
                    "minimum_group_rate": (
                        values.min() if len(values) else float("nan")
                    ),
                    "maximum_group_rate": (
                        values.max() if len(values) else float("nan")
                    ),
                    "max_minus_min_difference": (
                        values.max() - values.min()
                        if len(values)
                        else float("nan")
                    ),
                }
            )

    disparity_summary = pd.DataFrame(disparity_rows)

    fairness_audit.to_csv(
        RESULTS_DIR / "fairness_audit.csv",
        index=False,
    )
    disparity_summary.to_csv(
        RESULTS_DIR / "fairness_disparity_summary.csv",
        index=False,
    )

    return fairness_audit, disparity_summary

FAIRNESS_MONITORING_DIFFERENCE = 0.10


def assess_fairness_governance(fairness_audit, disparity_summary):
    """Convert descriptive fairness results into governance actions.

    The 0.10 difference is an internal monitoring trigger, not a legal
    definition of fairness and not an automatic model-rejection rule.
    """
    assessed = disparity_summary.copy()
    assessed["monitoring_threshold"] = FAIRNESS_MONITORING_DIFFERENCE

    def assign_action(row):
        difference = row["max_minus_min_difference"]

        if row["eligible_groups"] < 2 or pd.isna(difference):
            return "INSUFFICIENT_DATA"

        if difference > FAIRNESS_MONITORING_DIFFERENCE:
            return "REVIEW_REQUIRED"

        return "MONITOR"

    assessed["governance_action"] = assessed.apply(
        assign_action,
        axis=1,
    )

    excluded_groups = fairness_audit[
        ~fairness_audit["included_in_disparity_range"]
    ][
        ["audit_field", "group", "records"]
    ].copy()

    review_required = (
        assessed["governance_action"] == "REVIEW_REQUIRED"
    ).any()

    overall_status = (
        "REVIEW_REQUIRED" if review_required else "MONITOR"
    )

    governance_notes = pd.DataFrame(
        {
            "governance_item": [
                "overall_status",
                "demographics_used_as_model_features",
                "human_review_required",
                "minimum_audit_group_size",
                "monitoring_difference",
                "interpretation",
                "deployment_certification",
            ],
            "value": [
                overall_status,
                False,
                True,
                MIN_AUDIT_GROUP_SIZE,
                FAIRNESS_MONITORING_DIFFERENCE,
                (
                    "Observed differences require contextual review; "
                    "they do not independently establish model bias."
                ),
                (
                    "This educational synthetic-data audit is not a "
                    "legal, clinical, or deployment fairness certification."
                ),
            ],
        }
    )

    assessed.to_csv(
        RESULTS_DIR / "fairness_governance_assessment.csv",
        index=False,
    )
    excluded_groups.to_csv(
        RESULTS_DIR / "fairness_excluded_small_groups.csv",
        index=False,
    )
    governance_notes.to_csv(
        RESULTS_DIR / "fairness_governance_notes.csv",
        index=False,
    )

    return assessed, excluded_groups, overall_status

GENERATION_SAMPLE_SIZE = 12

SUPPORT_ACTIONS = {
    "housing_instability": (
        "review possible housing-resource navigation needs"
    ),
    "transportation_barrier": (
        "review transportation-assistance options"
    ),
    "food_insecurity": (
        "review community food-access resources"
    ),
    "medication_access_barrier": (
        "review medication-access coordination needs without providing "
        "medication advice"
    ),
    "behavioral_health_support_need": (
        "review behavioral-health service-navigation options"
    ),
    "substance_use_support_need": (
        "review voluntary substance-use support-navigation options"
    ),
}

PROHIBITED_GENERATION_PHRASES = [
    "diagnose",
    "prescribe",
    "stop taking",
    "increase the dose",
    "decrease the dose",
    "automatically eligible",
    "automatically contact",
]


def validate_generated_material(case_summary, outreach_plan):
    """Validate controlled text against the system's safety boundaries."""
    combined_text = f"{case_summary} {outreach_plan}".lower()
    violations = []

    for phrase in PROHIBITED_GENERATION_PHRASES:
        if phrase in combined_text:
            violations.append(f"prohibited phrase: {phrase}")

    if "human review required" not in combined_text:
        violations.append("missing human-review requirement")

    if "synthetic" not in combined_text:
        violations.append("missing synthetic-data designation")

    return violations


def generate_controlled_case_materials(df, test_predictions):
    """Create constrained, nonclinical drafts for selected synthetic records."""
    generation_input = test_predictions.merge(
        df,
        on=PATIENT_ID,
        how="left",
        validate="one_to_one",
    )

    generation_input = (
        generation_input.sort_values(
            "predicted_response_probability",
            ascending=False,
        )
        .head(GENERATION_SAMPLE_SIZE)
        .copy()
    )

    generated_rows = []

    for _, record in generation_input.iterrows():
        identified_actions = [
            action
            for feature, action in SUPPORT_ACTIONS.items()
            if int(record[feature]) == 1
        ]

        if int(record["primary_care_connected"]) == 0:
            identified_actions.append(
                "review primary-care connection and navigation options"
            )

        if not identified_actions:
            identified_actions.append(
                "review the synthetic record for appropriate nonclinical "
                "support-navigation opportunities"
            )

        case_summary = (
            f"Synthetic record {record[PATIENT_ID]} contains "
            f"{int(record['ems_calls_12m'])} EMS calls, "
            f"{int(record['ed_visits_12m'])} emergency-department visits, "
            f"and {int(record['social_need_count'])} recorded support "
            f"needs during the simulated review period. The model estimated "
            f"a {record['predicted_response_probability']:.1%} historical "
            f"outreach-response probability. This estimate is an advisory "
            f"prioritization aid only. Human review required."
        )

        outreach_plan = (
            "Draft for reviewer consideration: "
            + "; ".join(identified_actions)
            + ". Confirm the record context, available resources, and "
            "appropriate consent process before any action. Do not provide clinical "
            "assessment or treatment advice, determine eligibility automatically, or "
            "contact anyone without approval. Synthetic-data prototype; "
            "human review required."
        )

        violations = validate_generated_material(
            case_summary,
            outreach_plan,
        )

        generated_rows.append(
            {
                PATIENT_ID: record[PATIENT_ID],
                "predicted_response_probability": record[
                    "predicted_response_probability"
                ],
                "model_flag_for_human_review": record[
                    "model_flag_for_human_review"
                ],
                "case_summary": case_summary,
                "draft_outreach_plan": outreach_plan,
                "generation_mode": "deterministic_constrained",
                "demographics_included": False,
                "human_review_required": True,
                "safety_validation_passed": len(violations) == 0,
                "safety_violations": (
                    "NONE" if not violations else " | ".join(violations)
                ),
            }
        )

    generated_materials = pd.DataFrame(generated_rows)

    if not generated_materials["safety_validation_passed"].all():
        raise ValueError(
            "One or more generated drafts failed safety validation."
        )

    generated_materials.to_csv(
        RESULTS_DIR / "controlled_case_materials.csv",
        index=False,
    )

    return generated_materials



def run_integrated_system():
    """Execute the complete synthetic-data integrative AI workflow."""
    df, dictionary = load_and_validate_data()

    statistical_summary, utilization_results = (
        run_statistical_analysis(df)
    )

    (
        model_comparison,
        selected_model_name,
        selected_model,
        test_predictions,
        feature_importance,
    ) = run_machine_learning(df)

    fairness_audit, disparity_summary = run_fairness_audit(
        test_predictions
    )

    (
        fairness_assessment,
        excluded_groups,
        fairness_status,
    ) = assess_fairness_governance(
        fairness_audit,
        disparity_summary,
    )

    generated_materials = generate_controlled_case_materials(
        df,
        test_predictions,
    )

    selected_metrics = model_comparison[
        model_comparison["model"] == selected_model_name
    ].iloc[0]

    integrated_summary = pd.DataFrame(
        {
            "system_metric": [
                "synthetic_records",
                "data_dictionary_fields",
                "model_features",
                "selected_model",
                "test_records",
                "selected_model_roc_auc",
                "selected_model_average_precision",
                "fairness_governance_status",
                "fairness_comparisons_requiring_review",
                "small_audit_groups_excluded",
                "controlled_drafts_generated",
                "controlled_drafts_passing_safety_validation",
                "demographics_used_as_model_features",
                "demographics_in_generated_materials",
                "human_review_required",
                "autonomous_contact_allowed",
            ],
            "value": [
                len(df),
                len(dictionary),
                len(MODEL_FEATURES),
                selected_model_name,
                len(test_predictions),
                selected_metrics["roc_auc"],
                selected_metrics["average_precision"],
                fairness_status,
                (
                    fairness_assessment["governance_action"]
                    == "REVIEW_REQUIRED"
                ).sum(),
                len(excluded_groups),
                len(generated_materials),
                generated_materials[
                    "safety_validation_passed"
                ].sum(),
                False,
                generated_materials[
                    "demographics_included"
                ].any(),
                SYSTEM_BOUNDARIES["human_review_required"],
                SYSTEM_BOUNDARIES["autonomous_contact_allowed"],
            ],
        }
    )

    integrated_summary.to_csv(
        RESULTS_DIR / "integrated_system_summary.csv",
        index=False,
    )

    if fairness_status != "REVIEW_REQUIRED":
        raise ValueError(
            "Expected fairness governance review status was not preserved."
        )

    if not generated_materials[
        "safety_validation_passed"
    ].all():
        raise ValueError(
            "Controlled generation safety validation did not pass."
        )

    return {
        "data": df,
        "statistical_summary": statistical_summary,
        "utilization_results": utilization_results,
        "model_comparison": model_comparison,
        "selected_model_name": selected_model_name,
        "selected_model": selected_model,
        "test_predictions": test_predictions,
        "feature_importance": feature_importance,
        "fairness_audit": fairness_audit,
        "disparity_summary": disparity_summary,
        "fairness_assessment": fairness_assessment,
        "excluded_groups": excluded_groups,
        "fairness_status": fairness_status,
        "generated_materials": generated_materials,
        "integrated_summary": integrated_summary,
    }


if __name__ == "__main__":
    outputs = run_integrated_system()

    print("Familiar Faces integrated AI system")
    print("===================================")
    print(outputs["integrated_summary"].to_string(index=False))
    print()
    print("End-to-end pipeline: PASSED")
    print("Deployment status: EDUCATIONAL SYNTHETIC-DATA PROTOTYPE")
    print("Human review required: True")
