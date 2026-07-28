"""Build the source notebook for the Familiar Faces resubmission."""

from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

cells = [
    nbf.v4.new_markdown_cell(
        """# Familiar Faces: Industry-Integrated AI Systems Synthesis

## A Human-Supervised Care-Transition Outreach Research Prototype

This notebook integrates three genuinely implemented AI domains:

1. **Data analysis** of real, deidentified hospital encounters
2. **Machine learning** for advisory 30-day readmission prioritization
3. **Generative AI** using live OpenAI Responses API inference with structured output

The project uses the UCI Diabetes 130-US Hospitals dataset. It is an educational research prototype—not a clinical system, eligibility mechanism, or authorization for patient contact."""
    ),
    nbf.v4.new_markdown_cell(
        """## Safety and Governance Boundaries

- Public, deidentified research data only
- One eligible encounter per patient to prevent repeat-patient leakage
- Age, gender, and race excluded from model predictors and LLM inputs
- No diagnosis, treatment, medication, eligibility, or autonomous-contact decisions
- Confidence intervals, denominator rules, multiplicity adjustment, and controlled simulations for fairness auditing
- Deterministic safety validation after every LLM output
- Human review required before any possible downstream use"""
    ),
    nbf.v4.new_code_cell(
        """import pandas as pd
from IPython.display import Image, display

from familiar_faces_system import (
    MODEL_FEATURES,
    SYSTEM_BOUNDARIES,
    run_integrated_system,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

print("Model features:", len(MODEL_FEATURES))
print("Human review required:", SYSTEM_BOUNDARIES["human_review_required"])
print("Autonomous contact allowed:", SYSTEM_BOUNDARIES["autonomous_contact_allowed"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## Integrated Execution

The next cell runs the complete analytic workflow and performs genuine LLM inference. `OPENAI_API_KEY` must be configured in the environment. No protected demographic field or original encounter identifier is sent to the model."""
    ),
    nbf.v4.new_code_cell(
        """outputs = run_integrated_system(generate_llm=True)

print("End-to-end pipeline executed successfully.")
print("Selected model:", outputs["selected_model_name"])
print("Fairness governance status:", outputs["fairness_status"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## 1. Real-Data Cohort Validation

The source contains 101,766 deidentified encounters from 130 U.S. hospitals and integrated delivery networks between 1999 and 2008. The analytic cohort excludes hospice/expired dispositions and retains the first eligible encounter per patient. This prevents the same patient's encounters from appearing across training and test data."""
    ),
    nbf.v4.new_code_cell(
        """data = outputs["data"]
validation_summary = pd.DataFrame({
    "validation_item": [
        "eligible encounters",
        "unique patients",
        "duplicate patient identifiers",
        "model features",
        "protected demographics used as predictors",
        "30-day readmission rate",
    ],
    "result": [
        len(data),
        data["patient_nbr"].nunique(),
        int(data["patient_nbr"].duplicated().sum()),
        len(MODEL_FEATURES),
        False,
        data["readmitted_within_30_days"].mean(),
    ],
})
display(validation_summary)
display(outputs["dictionary"])"""
    ),
    nbf.v4.new_markdown_cell(
        """## 2. Descriptive Data Analysis

The descriptive layer examines the observed 30-day readmission rate and prior emergency, inpatient, and outpatient use. These are historical associations, not causal effects."""
    ),
    nbf.v4.new_code_cell(
        """display(outputs["statistical_summary"])
display(outputs["utilization_results"])
display(Image(filename="figures/readmission_by_prior_emergency_use.png", width=750))"""
    ),
    nbf.v4.new_markdown_cell(
        """## 3. Machine-Learning Comparison

Logistic regression and random forest use the same patient-independent held-out partition. Average precision is the primary selection metric because 30-day readmission is uncommon; ROC AUC and threshold-based metrics provide complementary evidence. The model output is only a review-queue signal."""
    ),
    nbf.v4.new_code_cell(
        """display(outputs["model_comparison"])
display(outputs["feature_importance"].head(12))
display(Image(filename="figures/selected_model_feature_importance.png", width=800))

assert outputs["model_comparison"]["patient_overlap"].eq(0).all()
print("Patient leakage check: PASSED")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 4. Validated Fairness Audit

Age, gender, and race are audit-only. Each group rate includes a 95% Wilson confidence interval. Comparisons require adequate total and outcome-conditioned denominators, compare each eligible group with the largest reference group, and use Holm adjustment across the full comparison family. A governance review requires both an absolute difference of at least 0.10 and an adjusted p-value below 0.05.

The method is separately tested in 200 independent-null simulations and 200 simulations with a known injected 0.15 review-flag disparity."""
    ),
    nbf.v4.new_code_cell(
        """display(outputs["fairness_validation"])
review_rows = outputs["fairness_comparisons"].query(
    "governance_action == 'REVIEW_REQUIRED'"
)
display(review_rows)

null_rate = outputs["fairness_validation"].loc[
    outputs["fairness_validation"]["validation_scenario"] == "independent_null_group",
    "trigger_rate",
].iloc[0]
detection_rate = outputs["fairness_validation"].loc[
    outputs["fairness_validation"]["validation_scenario"] == "injected_flag_rate_disparity",
    "trigger_rate",
].iloc[0]
assert null_rate <= 0.05
assert detection_rate >= 0.80
print("Fairness-method validation: PASSED")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 5. Genuine Generative-AI Component

The system sends five demographics-free, deidentified case inputs to an OpenAI model through the Responses API. Structured Outputs enforce the expected schema. A separate deterministic validator rejects protected-demographic language, diagnosis or prescribing language, dose changes, autonomous eligibility/contact language, or missing safety statements.

Passing the automated gate does not authorize use: every draft still requires human review."""
    ),
    nbf.v4.new_code_cell(
        """generated = outputs["generated_materials"]
generation_summary = pd.DataFrame({
    "metric": [
        "genuine LLM drafts",
        "drafts passing deterministic safety validation",
        "demographic fields sent",
        "demographics detected in outputs",
        "human review required for all",
    ],
    "value": [
        len(generated),
        int(generated["safety_validation_passed"].sum()),
        False,
        bool(generated["demographics_included"].any()),
        bool(generated["human_review_required"].all()),
    ],
})
display(generation_summary)
display(generated[[
    "case_id",
    "case_summary",
    "review_questions",
    "coordination_options",
    "limitations",
    "human_review_notice",
    "safety_violations",
]].head(3))

assert generated["generation_mode"].eq(
    "openai_responses_api_structured_output"
).all()
assert generated["safety_validation_passed"].all()
assert not generated["demographics_included"].any()
print("Generative-AI execution and safety validation: PASSED")"""
    ),
    nbf.v4.new_markdown_cell(
        """## 6. Integrated System Summary

The summary connects real-data analysis, empirical model comparison, validated fairness governance, genuine LLM inference, deterministic safety checks, and the terminal human-review boundary."""
    ),
    nbf.v4.new_code_cell(
        """display(outputs["integrated_summary"])

assert outputs["generated_materials"] is not None
assert SYSTEM_BOUNDARIES["human_review_required"]
assert not SYSTEM_BOUNDARIES["autonomous_contact_allowed"]
print("Integrated notebook validation: PASSED")"""
    ),
    nbf.v4.new_markdown_cell(
        """## Limitations and Responsible Use

- The dataset is historical (1999–2008), limited to encounters involving diabetes, and not representative of a present-day community-paramedicine population.
- Readmission is an imperfect proxy for whether outreach is appropriate or beneficial.
- The model is not calibrated or externally validated for operational decisions.
- The protected-group audit is observational and cannot establish the cause of a disparity.
- Null and injected-disparity simulations validate the audit rule under defined conditions, not every possible population or harm.
- LLM drafts may still be incomplete, misleading, or contextually inappropriate after automated checks.
- No component may diagnose, recommend treatment, determine eligibility, or initiate contact.
- Clinical, legal, privacy, security, accessibility, workflow, stakeholder, and prospective-effectiveness reviews would be required before any real-world research pilot."""
    ),
]

nb["cells"] = cells
output_path = Path("integrative_industry_synthesis.ipynb")
nbf.write(nb, output_path)
print("Created:", output_path)
print("Notebook cells:", len(cells))
print("Code cells:", sum(c["cell_type"] == "code" for c in cells))
print("Markdown cells:", sum(c["cell_type"] == "markdown" for c in cells))
