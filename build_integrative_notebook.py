from pathlib import Path

import nbformat as nbf


nb = nbf.v4.new_notebook()

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3",
    },
}

cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# Familiar Faces: Integrative Industry Synthesis

## A Human-Supervised AI Prototype for Community Paramedicine Outreach

This notebook demonstrates an integrated statistical, machine-learning, fairness-governance, and controlled-drafting workflow for a proposed **Familiar Faces** community paramedicine program.

The system uses only synthetic EHR-like records. It is an educational prototype and is not approved for clinical care, operational deployment, autonomous eligibility decisions, or autonomous patient contact."""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Safety and Governance Boundaries

The prototype enforces the following boundaries:

- Synthetic data only
- No protected demographic fields used as model predictors
- No demographics included in generated case materials
- No diagnosis or treatment recommendations
- No autonomous eligibility decisions
- No autonomous patient contact
- Human review required for every model flag and drafted outreach plan
- Fairness differences trigger governance review rather than automatic approval or rejection"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """import pandas as pd
from IPython.display import Image, display

from familiar_faces_system import (
    MODEL_FEATURES,
    SYSTEM_BOUNDARIES,
    run_integrated_system,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)

print("Notebook environment initialized.")
print("Model features:", len(MODEL_FEATURES))
print("Human review required:", SYSTEM_BOUNDARIES["human_review_required"])
print("Autonomous contact allowed:", SYSTEM_BOUNDARIES["autonomous_contact_allowed"])"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Integrated System Architecture

The workflow contains five connected layers:

1. **Synthetic data and schema validation**
2. **Descriptive statistical analysis**
3. **Machine-learning model comparison and prioritization**
4. **Protected-group fairness auditing and governance assessment**
5. **Deterministic, constrained drafting for human review**

The machine-learning output is an advisory prioritization signal. It does not determine program eligibility or initiate outreach."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """outputs = run_integrated_system()

print("End-to-end pipeline executed successfully.")
print("Deployment status: EDUCATIONAL SYNTHETIC-DATA PROTOTYPE")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. Synthetic Data Validation

The dataset represents simulated community-paramedicine records. Synthetic identifiers and the explicit `data_source` field help prevent the prototype from being mistaken for a real clinical system."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """data = outputs["data"]

validation_summary = pd.DataFrame(
    {
        "validation_item": [
            "records",
            "columns",
            "unique_patient_identifiers",
            "missing_values",
            "data_sources",
        ],
        "result": [
            len(data),
            len(data.columns),
            data["synthetic_patient_id"].nunique(),
            int(data.isna().sum().sum()),
            ", ".join(sorted(data["data_source"].unique())),
        ],
    }
)

display(validation_summary)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Statistical Analysis

Descriptive analysis examines utilization, social-support needs, primary-care connection, and the simulated outreach-response outcome. These associations describe the synthetic dataset and do not establish causal relationships."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """display(outputs["statistical_summary"])
display(outputs["utilization_results"])

display(
    Image(
        filename="figures/outreach_response_by_utilization.png",
        width=750,
    )
)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Machine-Learning Comparison

Logistic regression and random forest models are compared on the same stratified test set. Selection is based on ROC AUC, while average precision and classification metrics provide additional context.

The moderate model performance supports limited use as a human-reviewed prioritization aid—not as an autonomous decision system."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """print("Selected model:", outputs["selected_model_name"])
display(outputs["model_comparison"])
display(outputs["feature_importance"].head(10))

display(
    Image(
        filename="figures/selected_model_feature_importance.png",
        width=800,
    )
)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Fairness Audit and Governance Review

Age group, sex, and race/ethnicity are excluded from model training but retained for post-model auditing. Groups with fewer than 25 test records are excluded from disparity-range calculations.

The internal 0.10 difference threshold is a monitoring trigger. It is not a legal definition of fairness or a deployment-certification standard."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """display(outputs["fairness_assessment"])

print("Groups excluded from disparity ranges because of small samples:")
display(outputs["excluded_groups"])

print("Overall fairness governance status:", outputs["fairness_status"])"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 5. Controlled Case-Material Drafting

The prototype creates deterministic, constrained summaries and nonclinical outreach-plan drafts for selected synthetic records.

This component demonstrates how safeguards learned from generative-AI development can be applied without claiming that the current implementation performs live large-language-model inference. Every draft must pass automated checks and then receive human review."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """generated = outputs["generated_materials"]

generation_summary = pd.DataFrame(
    {
        "metric": [
            "drafts_generated",
            "drafts_passing_validation",
            "demographics_included",
            "human_review_required_for_all",
        ],
        "value": [
            len(generated),
            int(generated["safety_validation_passed"].sum()),
            bool(generated["demographics_included"].any()),
            bool(generated["human_review_required"].all()),
        ],
    }
)

display(generation_summary)

display(
    generated[
        [
            "synthetic_patient_id",
            "predicted_response_probability",
            "case_summary",
            "draft_outreach_plan",
            "safety_violations",
        ]
    ].head(3)
)"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 6. Integrated System Summary

The final summary connects the analytical, predictive, fairness, and controlled-drafting components while preserving the prototype's operational boundaries."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """display(outputs["integrated_summary"])

assert outputs["fairness_status"] == "REVIEW_REQUIRED"
assert generated["safety_validation_passed"].all()
assert not generated["demographics_included"].any()
assert SYSTEM_BOUNDARIES["human_review_required"]
assert not SYSTEM_BOUNDARIES["autonomous_contact_allowed"]

print("Integrated notebook validation: PASSED")
print("Human review required: True")
print("Autonomous contact allowed: False")"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Limitations and Responsible-Use Statement

This prototype has several important limitations:

- All records and outcomes are synthetic.
- The modeled outcome represents simulated outreach response, not clinical benefit.
- Moderate predictive performance limits operational usefulness.
- Observed subgroup disparities require investigation and continued monitoring.
- Small demographic groups produce unstable estimates.
- Feature importance does not establish causality.
- The drafting component is deterministic and constrained rather than live LLM inference.
- The system has not undergone clinical, legal, privacy, cybersecurity, accessibility, or deployment validation.

Before any real-world pilot, the organization would require formal data governance, HIPAA and legal review, community and stakeholder participation, security controls, model validation using appropriately authorized data, workflow testing, consent procedures, audit logging, escalation policies, and ongoing fairness monitoring."""
    )
)

nb["cells"] = cells

output_path = Path("integrative_industry_synthesis.ipynb")
nbf.write(nb, output_path)

print("Created:", output_path)
print("Notebook cells:", len(nb["cells"]))
print(
    "Code cells:",
    sum(cell["cell_type"] == "code" for cell in nb["cells"]),
)
print(
    "Markdown cells:",
    sum(cell["cell_type"] == "markdown" for cell in nb["cells"]),
)
print("Notebook structure: PASSED")
