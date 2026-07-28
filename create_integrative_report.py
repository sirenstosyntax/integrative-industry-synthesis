"""Create the Familiar Faces Industry-Integrated AI Systems Synthesis report."""

from pathlib import Path
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PATH = Path("Integrative_Industry_Synthesis_Report.pdf")
ARCHITECTURE_PATH = Path("figures/familiar_faces_system_architecture.png")
RESULTS_DIR = Path("results")
REPORT_DATE = "July 28, 2026"

required_files = [
    ARCHITECTURE_PATH,
    RESULTS_DIR / "model_comparison.csv",
    RESULTS_DIR / "fairness_group_comparisons.csv",
    RESULTS_DIR / "fairness_method_validation.csv",
    RESULTS_DIR / "llm_case_materials.csv",
    RESULTS_DIR / "llm_generation_metadata.csv",
    RESULTS_DIR / "integrated_system_summary.csv",
]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Run `python familiar_faces_system.py --generate-llm` before "
        f"building the final report. Missing: {missing}"
    )

models = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
fairness = pd.read_csv(RESULTS_DIR / "fairness_group_comparisons.csv")
validation = pd.read_csv(RESULTS_DIR / "fairness_method_validation.csv")
drafts = pd.read_csv(RESULTS_DIR / "llm_case_materials.csv")
summary = pd.read_csv(RESULTS_DIR / "integrated_system_summary.csv")
summary_values = dict(zip(summary["system_metric"], summary["value"]))

if not drafts["generation_mode"].eq(
    "openai_responses_api_structured_output"
).all():
    raise ValueError("Final report requires genuine structured LLM inference.")
if not drafts["safety_validation_passed"].all():
    raise ValueError("Final report requires all LLM drafts to pass validation.")
if str(summary_values.get("genuine_llm_inference_executed", "")).lower() != "true":
    raise ValueError("Integrated summary does not record genuine LLM execution.")

selected = models.iloc[0]
review_count = int(
    (fairness["governance_action"] == "REVIEW_REQUIRED").sum()
)
null_rate = float(
    validation.loc[
        validation["validation_scenario"] == "independent_null_group",
        "trigger_rate",
    ].iloc[0]
)
detection_rate = float(
    validation.loc[
        validation["validation_scenario"]
        == "injected_flag_rate_disparity",
        "trigger_rate",
    ].iloc[0]
)
draft_passes = int(drafts["safety_validation_passed"].sum())


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#263238"),
        spaceAfter=14,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11.5,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#546E7A"),
        spaceAfter=16,
    )
)
styles.add(
    ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#B7472A"),
        spaceBefore=9,
        spaceAfter=7,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.7,
        leading=14.2,
        textColor=colors.HexColor("#263238"),
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.2,
        leading=11.5,
        textColor=colors.HexColor("#546E7A"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="APA_Number",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="APA_Title",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=12,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="ReferenceBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.5,
        leftIndent=0.35 * inch,
        firstLineIndent=-0.35 * inch,
        spaceAfter=8,
    )
)


def draw_page(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#CFD8DC"))
    canvas.line(
        doc.leftMargin,
        0.55 * inch,
        width - doc.rightMargin,
        0.55 * inch,
    )
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607D8B"))
    canvas.drawString(
        doc.leftMargin,
        0.35 * inch,
        "Familiar Faces | Industry-Integrated AI Systems Synthesis",
    )
    canvas.drawRightString(
        width - doc.rightMargin, 0.35 * inch, f"Page {doc.page}"
    )
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUTPUT_PATH),
    pagesize=letter,
    rightMargin=0.7 * inch,
    leftMargin=0.7 * inch,
    topMargin=0.68 * inch,
    bottomMargin=0.75 * inch,
    title="Familiar Faces: Industry-Integrated AI Systems Synthesis",
    author="Grant Collings",
)
story = []
counted_body = []


def add_body(text):
    counted_body.append(re.sub(r"<[^>]+>", "", text))
    story.append(Paragraph(text, styles["ReportBody"]))


story.extend(
    [
        Spacer(1, 0.52 * inch),
        Paragraph("Familiar Faces", styles["ReportTitle"]),
        Paragraph(
            "Industry-Integrated AI Systems Synthesis",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 0.16 * inch),
        Paragraph(
            "A Human-Supervised Care-Transition Outreach Research Prototype",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 0.32 * inch),
        Paragraph("<b>Prepared by:</b> Grant Collings", styles["ReportSubtitle"]),
        Paragraph(
            "Udacity AI for Industry Applications Capstone",
            styles["ReportSubtitle"],
        ),
        Paragraph(REPORT_DATE, styles["ReportSubtitle"]),
        Spacer(1, 0.32 * inch),
        Paragraph(
            "EDUCATIONAL DEIDENTIFIED-DATA RESEARCH PROTOTYPE",
            styles["ReportSubtitle"],
        ),
        Paragraph(
            "Not approved for clinical care, operational deployment, eligibility "
            "decisions, or autonomous patient contact.",
            styles["SmallNote"],
        ),
        PageBreak(),
        Paragraph("Executive Summary", styles["SectionHeading"]),
    ]
)

add_body(
    "Familiar Faces explores how a community paramedicine or care-transition team "
    "might use limited AI support to organize human review after hospitalization. "
    "The revised prototype implements three connected domains: descriptive analysis "
    "of real, deidentified hospital encounters; supervised machine learning for an "
    "advisory 30-day-readmission signal; and genuine large-language-model inference "
    "for structured, nonclinical review drafts. A validated fairness layer, "
    "deterministic output checks, and mandatory human review govern those components. "
    "The source is the public UCI Diabetes 130-US Hospitals dataset, which contains "
    "101,766 encounters from 1999 through 2008 (Clore et al., 2014). After excluding "
    "hospice or expired dispositions and retaining one eligible encounter per "
    "patient, the analytic cohort contains 69,990 encounters."
)
add_body(
    f"Logistic regression was selected empirically over random forest on a common, "
    f"patient-independent test partition. It achieved ROC AUC {selected['roc_auc']:.3f} "
    f"and average precision {selected['average_precision']:.3f}. Those results are "
    f"moderate and support only a review-queue experiment. The fairness rule produced "
    f"{review_count} review triggers. In 200 independent-null simulations its "
    f"familywise false-positive rate was {null_rate:.1%}; in 200 simulations with an "
    f"injected 0.15 flag-rate disparity, detection was {detection_rate:.1%}. Live "
    f"structured inference generated {len(drafts)} case drafts, and {draft_passes} "
    f"passed the separate safety validator. Passing does not authorize action; the "
    f"prototype preserves an explicit human terminal boundary."
)

story.append(Paragraph("1. Industry Problem and Scope", styles["SectionHeading"]))
add_body(
    "Emergency response resolves immediate threats, but repeated emergency use and "
    "hospital readmission may also reflect needs that a single encounter cannot "
    "address. Community paramedicine and care-transition teams can coordinate "
    "follow-up, verify discharge understanding, and connect people with authorized "
    "resources. The responsible AI opportunity is therefore narrow: help staff sort "
    "a review queue and prepare consistent questions. It is not to diagnose, predict "
    "who deserves care, or replace professional judgment. Readmission is used because "
    "it is an observed outcome in the source data, not because it fully represents "
    "outreach need or likely benefit."
)
add_body(
    "The system estimates whether an encounter was followed by readmission within 30 "
    "days. Staff would have to interpret that signal alongside consent, current "
    "circumstances, program criteria, and information unavailable to the model. No "
    "component may determine eligibility, recommend treatment or medication changes, "
    "or contact anyone. This boundary recognizes that prioritization can itself "
    "affect access and attention. The appropriate capstone claim is technical and "
    "governance feasibility—not clinical effectiveness."
)
add_body(
    "The intended reviewer is an authorized professional, not the patient and not an "
    "automated outreach service. The queue would provide a starting point for review, "
    "while staff remain responsible for deciding whether a record is current, whether "
    "contact is permitted, and whether the program has an appropriate resource. A low "
    "score must never block review, and a high score must never compel contact. This "
    "asymmetry is deliberate because the cost of mistaken exclusion differs from the "
    "cost of asking a reviewer to inspect another record."
)

story.append(
    Paragraph("2. Cross-Domain Integration and Rationale", styles["SectionHeading"])
)
add_body(
    "<b>Data Workflow.</b> Earlier data work established the importance of explicit "
    "cohort definitions, traceable transformations, schema validation, and "
    "reproducible outputs. Those practices now operate on a real outcome rather than "
    "a formula-generated synthetic label. The pipeline documents exclusions, retains "
    "the first eligible encounter per patient, and writes a processed cohort plus "
    "data dictionary. This directly removes circular evaluation: neither predictors "
    "nor labels are generated from the model being evaluated."
)
add_body(
    "<b>Machine Learning Workflow.</b> The NEMSIS modeling work informed fixed seeds, "
    "patient-independent partitioning, common evaluation data, and comparison of an "
    "interpretable linear model with a nonlinear ensemble. Average precision is the "
    "primary selection measure because only about nine percent of held-out encounters "
    "have the positive outcome. ROC AUC, precision, recall, F1, and accuracy remain "
    "visible so that no single score hides the operational tradeoff. Model feature "
    "importance is an explanation of model behavior, not evidence of causality."
)
add_body(
    "<b>Generative AI.</b> The earlier narrative-generation work showed that fluent "
    "text can invent unsupported detail and sound more authoritative than its "
    "evidence. This synthesis therefore implements a genuine but constrained LLM "
    "component. Five records are converted to demographics-free, deidentified inputs "
    "containing only utilization counts, encounter counts, and the model probability. "
    "The OpenAI Responses API returns a defined schema containing a summary, three "
    "verification questions, conditional coordination options, limitations, and a "
    "human-review notice. A deterministic validator then checks every output for "
    "protected-demographic language, clinical or prescribing language, dose changes, "
    "autonomous eligibility or contact, and required warning phrases. Structured "
    "output improves schema reliability but does not make generated content true or "
    "safe by itself (OpenAI, 2026)."
)
add_body(
    "<b>Deep Learning Systems and Agentic AI insights.</b> These subjects influenced "
    "design choices but are not misrepresented as implemented domains. Sequential "
    "deep models were unnecessary for the selected tabular encounter features, where "
    "interpretability and limited evidence favored simpler classifiers. Likewise, a "
    "fixed, human-supervised workflow does not need an autonomous agent. Agentic AI "
    "work contributed the stop conditions, permission mindset, and escalation pattern. "
    "The three implemented domains are data analysis, machine learning, and generative "
    "AI; the other subjects supply justified constraints."
)

story.append(PageBreak())
story.append(Paragraph("3. System Architecture", styles["SectionHeading"]))
add_body(
    "The architecture separates cohort preparation, descriptive analysis, model "
    "comparison, protected-group auditing, LLM drafting, deterministic validation, "
    "and human review. Age, gender, and race follow an audit-only path and never "
    "enter the predictive or generative feature sets. The model probability can "
    "select records for draft preparation, but neither the model nor the LLM can "
    "initiate action. This separation keeps prediction, auditing, language generation, "
    "and accountability inspectable rather than collapsing them into an opaque score."
)

architecture = Image(str(ARCHITECTURE_PATH))
architecture.drawHeight = architecture.imageHeight * (
    6.85 * inch / architecture.imageWidth
)
architecture.drawWidth = 6.85 * inch
story.extend(
    [
        Paragraph("Figure 1", styles["APA_Number"]),
        Paragraph(
            "Familiar Faces Human-Supervised System Architecture",
            styles["APA_Title"],
        ),
        architecture,
        Paragraph(
            "<i>Note.</i> Protected attributes are audit-only. LLM inputs are "
            "deidentified and demographics-free. No component authorizes contact "
            "or clinical action.",
            styles["SmallNote"],
        ),
    ]
)

results_data = [
    ["System metric", "Observed result"],
    ["Eligible unique-patient encounters", "69,990"],
    ["Held-out encounters", f"{int(selected['test_records']):,}"],
    ["Selected model", "Logistic regression"],
    ["ROC AUC", f"{selected['roc_auc']:.3f}"],
    ["Average precision", f"{selected['average_precision']:.3f}"],
    ["Test precision", f"{selected['precision']:.3f}"],
    ["Test recall", f"{selected['recall']:.3f}"],
    ["Fairness review triggers", str(review_count)],
    ["Null-simulation trigger rate", f"{null_rate:.1%}"],
    ["Injected-disparity detection", f"{detection_rate:.1%}"],
    ["LLM drafts passing safety gate", f"{draft_passes} of {len(drafts)}"],
]
results_table = Table(
    results_data, colWidths=[3.75 * inch, 2.45 * inch], repeatRows=1
)
results_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.7),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#ECEFF1")],
            ),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )
)
story.extend(
    [
        Paragraph("4. Results and Evaluation", styles["SectionHeading"]),
        Paragraph("Table 1", styles["APA_Number"]),
        Paragraph("Integrated Prototype Results", styles["APA_Title"]),
        results_table,
        Spacer(1, 0.1 * inch),
    ]
)
add_body(
    "The real outcome makes model comparison empirical. Logistic regression's "
    "average precision exceeded random forest by a small margin, while random forest "
    "produced a somewhat higher threshold-based F1. Selection followed the "
    "predeclared average-precision rule, not whichever metric made the system appear "
    "strongest. The result is intentionally described as modest. It shows some "
    "ranking information beyond the base rate, but it does not establish calibration, "
    "transportability, treatment effect, or outreach benefit."
)

story.append(
    Paragraph("5. Fairness Validation and Governance", styles["SectionHeading"])
)
add_body(
    "Removing protected attributes from training does not guarantee equitable "
    "behavior because other features and the outcome can encode structural "
    "differences. Health-management algorithms have produced racial disparity even "
    "when designers used a seemingly reasonable proxy objective (Obermeyer et al., "
    "2019). Familiar Faces therefore retains protected fields only for post-model "
    "auditing. Each group rate includes a 95% Wilson interval. A comparison requires "
    "at least 500 group records and at least 200 observations in the relevant outcome "
    "denominator. Each eligible group is compared with the largest reference group."
)
add_body(
    "A review trigger requires both practical and statistical evidence: an absolute "
    "rate difference of at least 0.10 and a Holm-adjusted p-value below .05 across the "
    "complete comparison family. The method itself is tested rather than assumed "
    "valid. In each null simulation, three independent audit fields are randomly "
    "assigned and flag, true-positive, and false-positive rates are evaluated. The "
    f"observed familywise false-positive rate was {null_rate:.1%}. In the matched "
    f"injected-disparity experiment, a 0.15 review-flag difference was detected in "
    f"{detection_rate:.1%} of runs. These experiments address false alarms and power "
    "under stated conditions; they do not certify fairness in all settings."
)
add_body(
    f"The real-data audit produced {review_count} REVIEW_REQUIRED comparisons, "
    "including age-related flag and error-rate differences and one race-related "
    "true-positive-rate difference. These are governance signals, not proof of "
    "discrimination or permission to alter services automatically. They require "
    "contextual investigation, stakeholder review, sensitivity analysis, and "
    "potential model revision. NIST's AI Risk Management Framework similarly treats "
    "risk work as continuing governance, mapping, measurement, and management rather "
    "than a one-time technical score (Tabassi, 2023)."
)

story.append(
    Paragraph("6. Ethics, Constraints, and Responsible Use", styles["SectionHeading"])
)
add_body(
    "The primary ethical risk is unequal allocation of staff attention. Additional "
    "risks include automation bias, privacy misuse, historical-data drift, unsupported "
    "LLM inferences, and purpose expansion from review support to eligibility control. "
    "The system responds with layered safeguards: deidentified public research data; "
    "audit-only protected attributes; patient-independent evaluation; explicit "
    "uncertainty; constrained structured generation; a separate deterministic text "
    "gate; no autonomous action; and mandatory human review. Human oversight is not "
    "treated as a slogan: an authorized reviewer must verify facts, consent, current "
    "context, available resources, and the appropriateness of any next step."
)
add_body(
    "Important limitations remain. The data are from 1999–2008, involve hospitalized "
    "patients with diabetes, and do not represent a modern community-paramedicine "
    "population. One encounter per patient improves independence but discards later "
    "history. Readmission may be planned, unavoidable, or unrelated to appropriate "
    "outreach. The models have not been externally validated or prospectively tested. "
    "The fairness simulations use controlled binary assignments and one injected "
    "effect size. LLM safety checks cannot detect every misleading implication. The "
    "prototype has not undergone clinical, legal, privacy, cybersecurity, "
    "accessibility, usability, or organizational validation."
)
add_body(
    "Before any real-world research pilot, an organization would need an approved "
    "protocol, lawful and authorized data access, privacy and security controls, "
    "community and patient participation, independent model validation, calibration "
    "and subgroup analysis on representative data, reviewer training, audit logging, "
    "appeal and escalation procedures, incident ownership, and continuous monitoring. "
    "Prospective evaluation would need patient-centered and program outcomes—not just "
    "readmission prediction—and a comparison showing whether the AI-assisted workflow "
    "improves decisions without worsening inequity."
)

story.append(
    Paragraph("7. Professional Reflection and Conclusion", styles["SectionHeading"])
)
add_body(
    "This revision sharpened the difference between assembling components and "
    "integrating valid evidence. A model that predicts its own label-generating "
    "formula is reproducible but not informative. A fairness threshold without "
    "false-positive and detection testing can create confidence without validity. A "
    "text template can demonstrate safeguards but is not generative AI. Replacing "
    "those shortcuts with a real outcome, a validated audit, and executed LLM "
    "inference made the portfolio artifact more honest and technically meaningful."
)
add_body(
    "Familiar Faces now demonstrates system-level judgment across data analysis, "
    "machine learning, generative AI, fairness evaluation, and governance. Its value "
    "does not depend on claiming deployment readiness. The strongest conclusion is "
    "that the integrated workflow runs, exposes its limitations, and stops at the "
    "correct boundary. Further work should begin with stakeholder and governance "
    "review, not operational use."
)

story.extend(
    [
        PageBreak(),
        Paragraph("References", styles["SectionHeading"]),
        Paragraph(
            "Clore, J., Cios, K., DeShazo, J., &amp; Strack, B. (2014). "
            "<i>Diabetes 130-US Hospitals for Years 1999-2008</i> [Data set]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5230J",
            styles["ReferenceBody"],
        ),
        Paragraph(
            "Obermeyer, Z., Powers, B., Vogeli, C., &amp; Mullainathan, S. "
            "(2019). Dissecting racial bias in an algorithm used to manage the "
            "health of populations. <i>Science, 366</i>(6464), 447-453. "
            "https://doi.org/10.1126/science.aax2342",
            styles["ReferenceBody"],
        ),
        Paragraph(
            "OpenAI. (2026). <i>Structured model outputs</i>. "
            "https://developers.openai.com/api/docs/guides/structured-outputs",
            styles["ReferenceBody"],
        ),
        Paragraph(
            "Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., "
            "Cios, K. J., &amp; Clore, J. N. (2014). Impact of HbA1c "
            "measurement on hospital readmission rates: Analysis of 70,000 "
            "clinical database patient records. <i>BioMed Research "
            "International, 2014</i>, Article 781670. "
            "https://doi.org/10.1155/2014/781670",
            styles["ReferenceBody"],
        ),
        Paragraph(
            "Tabassi, E. (2023). <i>Artificial intelligence risk management "
            "framework (AI RMF 1.0)</i> (NIST AI 100-1). National Institute of "
            "Standards and Technology. https://doi.org/10.6028/NIST.AI.100-1",
            styles["ReferenceBody"],
        ),
    ]
)

body_word_count = len(
    re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", " ".join(counted_body))
)
if not 1500 <= body_word_count <= 2000:
    raise ValueError(
        f"Synthesis body word count {body_word_count} is outside 1,500-2,000."
    )

doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
print(f"Created PDF: {OUTPUT_PATH}")
print(f"Synthesis body word count: {body_word_count}")
