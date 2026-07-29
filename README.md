# Familiar Faces: Industry-Integrated AI Systems Synthesis

Familiar Faces is a human-supervised research prototype exploring how a community-paramedicine or care-transition team might organize review after hospitalization.

The revised system genuinely implements and connects three AI domains:

1. Real-data statistical analysis
2. Supervised machine learning
3. Generative AI using live OpenAI Responses API inference

The project also implements a statistically validated fairness-governance layer, deterministic LLM-output safety checks, and a mandatory human-review boundary.

This is an educational, deidentified-data research prototype. It is not approved for clinical use, eligibility decisions, operational deployment, or autonomous patient contact.

## Industry Problem and System Boundary

Repeated emergency use and hospital readmission may indicate a need for additional care-transition review, but readmission does not by itself establish outreach need, patient preference, eligibility, or likely benefit.

The system may:

- Describe historical encounter patterns
- Estimate an observed 30-day-readmission outcome
- Order records for authorized human review
- Draft nonclinical verification questions and conditional coordination options
- Audit protected-group model behavior

The system may not:

- Diagnose or recommend treatment
- Recommend medication or dose changes
- Determine program eligibility
- Infer undocumented social or clinical needs
- Use protected demographics as predictors or LLM inputs
- Contact anyone automatically
- Replace authorized professional review

## Data

The project uses the public **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository:

> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J

The source contains 101,766 deidentified hospital encounters. The analytic cohort:

- Excludes discharge dispositions recorded as hospice or expired
- Sorts encounters by deidentified encounter identifier
- Retains the first eligible encounter per patient
- Contains 69,990 unique-patient encounters
- Defines the positive outcome as recorded readmission within 30 days

Keeping one encounter per patient prevents repeat-patient leakage across the training and test sets.

## Machine Learning

Twenty nonprotected encounter and prior-utilization fields are used as predictors. Four are transparent engineered features: total prior utilization and indicators for any prior emergency, inpatient, or outpatient use. Age, gender, race, patient number, and encounter number are excluded.

The workflow compares and tunes:

- Logistic regression
- Random forest
- Histogram gradient boosting

The final 25% test partition is held untouched during model selection. Hyperparameters are selected using four-fold, patient-independent cross-validation within the remaining training data. Mean cross-validated average precision is the primary selection metric because the positive outcome is uncommon, with ROC AUC as the secondary criterion. Only the selected configuration is evaluated on the final test set. ROC AUC and average precision receive 1,000-sample bootstrap 95% confidence intervals; accuracy, precision, recall, and F1 are also reported descriptively at the fixed 0.50 threshold.

Current analytic results:

| Metric | Result |
|---|---:|
| Eligible unique-patient encounters | 69,990 |
| Training records | 52,492 |
| Test records | 17,498 |
| Test positive rate | 8.98% |
| Selected model | Random forest |
| ROC AUC (95% bootstrap CI) | 0.644 (0.629–0.659) |
| Average precision (95% bootstrap CI) | 0.166 (0.153–0.181) |
| Precision at 0.50 threshold | 0.144 |
| Recall at 0.50 threshold | 0.502 |
| Patient overlap | 0 |

Random forest and histogram gradient boosting were effectively tied in cross-validated average precision (0.168727 versus 0.168709); the deterministic selection rule chose random forest. The prior workflow produced ROC AUC 0.644 and average precision 0.164 but reused the test set for model selection. The revised score improvement is small; its value is the more credible evaluation design, not evidence of materially stronger discrimination. These results support only a limited, human-reviewed research demonstration.

## Validated Fairness Governance

Age, gender, and race are retained only for post-model auditing.

The audit:

- Reports 95% Wilson confidence intervals for group rates
- Requires at least 500 total group records
- Requires at least 200 observations in the relevant outcome-conditioned denominator
- Compares eligible groups with the largest reference group
- Applies Holm adjustment across the complete comparison family
- Requires both an absolute difference of at least 0.10 and adjusted *p* < .05 for `REVIEW_REQUIRED`

The rule is tested in controlled simulations:

| Validation scenario | Simulations | Result |
|---|---:|---:|
| Independent-null groups across three audit fields and three metrics | 200 | 0.0% familywise trigger rate |
| Injected 0.15 review-flag disparity | 200 | 100.0% detection rate |

The real-data audit currently returns `REVIEW_REQUIRED`. This is a governance signal requiring investigation; it is not a legal conclusion, causal finding, or automatic model-rejection decision.

## Genuine Generative-AI Component

When `--generate-llm` is supplied, the system:

1. Selects five high-probability held-out records.
2. Converts them to demographics-free case identifiers and numerical encounter facts.
3. Sends those inputs to an OpenAI model through the Responses API.
4. Uses Structured Outputs to require a defined response schema.
5. Applies a separate deterministic safety validator.
6. Stops if any draft fails validation.

The validator rejects:

- Protected-demographic language
- Diagnosis or prescribing language
- Dose-change language
- Autonomous eligibility or contact language
- Missing deidentified-data, nonclinical, or human-review notices

No API key is stored in the notebook, source code, output files, or repository.

## Project Structure

- `data/raw/diabetic_data.csv` — original UCI encounter data
- `data/raw/IDS_mapping.csv` — UCI identifier mapping file
- `data/processed/familiar_faces_uci_encounters.csv` — reproducible analytic cohort
- `data/processed/data_dictionary.csv` — field roles and descriptions
- `familiar_faces_system.py` — integrated analysis, modeling, fairness, and LLM pipeline
- `build_integrative_notebook.py` — source-notebook builder
- `integrative_industry_synthesis.ipynb` — source notebook
- `integrative_industry_synthesis_executed.ipynb` — executed notebook
- `create_architecture_diagram.py` — architecture figure builder
- `create_integrative_report.py` — synthesis PDF builder
- `figures/` — architecture and analytical figures
- `results/` — model, fairness, simulation, generation, and summary artifacts
  - `model_selection_cv_results.csv` — all training-only tuning results
  - `selected_model_bootstrap_intervals.csv` — final-test uncertainty intervals

## Environment Setup

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the Analytic Workflow

Run the real-data analysis, model comparison, and fairness validation without making an API call:

```powershell
python .\familiar_faces_system.py
```

A successful run ends with:

```text
End-to-end analytic pipeline: PASSED
LLM component: NOT REQUESTED
Human review required: True
```

## Running the Genuine LLM Component

Configure `OPENAI_API_KEY` in the current PowerShell window without displaying it. Then run:

```powershell
python .\familiar_faces_system.py --generate-llm
```

The default model is `gpt-5.6`. A different available model can be supplied explicitly:

```powershell
python .\familiar_faces_system.py --generate-llm --model gpt-5.6
```

A successful full run ends with:

```text
End-to-end analytic pipeline: PASSED
LLM component: EXECUTED AND VALIDATED
Human review required: True
```

## Rebuilding and Executing the Notebook

The API key must remain configured in the same PowerShell window:

```powershell
python .\build_integrative_notebook.py
jupyter nbconvert --to notebook --execute .\integrative_industry_synthesis.ipynb --output .\integrative_industry_synthesis_executed.ipynb --ExecutePreprocessor.timeout=1200
```

The executed notebook performs the genuine LLM call and asserts:

- Zero patient overlap
- Fairness null trigger rate no greater than 5%
- Injected-disparity detection rate at least 80%
- Genuine structured LLM inference occurred
- Every draft passed deterministic safety validation
- No demographic field was sent or detected
- Human review remains required

## Rebuilding the Figure and Report

After the full LLM run:

```powershell
python .\create_architecture_diagram.py
python .\create_integrative_report.py
```

The report builder requires the LLM output and refuses to create a final PDF if the genuine-inference artifact is missing.

## Limitations

- The data are historical (1999–2008) and limited to hospitalized patients with diabetes.
- The dataset is not representative of a present-day community-paramedicine population.
- Thirty-day readmission is an imperfect proxy for outreach appropriateness or benefit.
- Model performance is moderate and not externally validated.
- The fairness audit is observational and does not identify the cause of a disparity.
- Simulation results apply to the tested denominators, metrics, and injected effect size.
- Structured output and deterministic checks cannot guarantee contextual accuracy.
- No prospective, clinical, legal, privacy, security, accessibility, or workflow validation has occurred.

## Responsible-Use Statement

Familiar Faces demonstrates a governed technical workflow, not a deployable clinical product. Any real-world research would require authorized data access, privacy and security controls, stakeholder participation, independent validation, consent and outreach policies, reviewer training, audit logging, escalation and appeal procedures, continuous monitoring, and prospective evaluation using patient-centered outcomes.
