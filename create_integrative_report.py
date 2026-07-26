from pathlib import Path
import re

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
REPORT_DATE = "July 26, 2026"


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
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
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#546E7A"),
        spaceAfter=18,
    )
)
styles.add(
    ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#B7472A"),
        spaceBefore=10,
        spaceAfter=8,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Subheading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#263238"),
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#263238"),
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallNote",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#546E7A"),
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="APA_Number",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#263238"),
        spaceBefore=6,
        spaceAfter=2,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="APA_Title",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#263238"),
        spaceAfter=6,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="ReferenceBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#263238"),
        leftIndent=0.35 * inch,
        firstLineIndent=-0.35 * inch,
        spaceAfter=9,
    )
)


def draw_page(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#CFD8DC"))
    canvas.line(doc.leftMargin, 0.55 * inch, width - doc.rightMargin, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607D8B"))
    canvas.drawString(
        doc.leftMargin,
        0.35 * inch,
        "Familiar Faces | Integrative Industry Synthesis",
    )
    canvas.drawRightString(
        width - doc.rightMargin,
        0.35 * inch,
        f"Page {doc.page}",
    )
    canvas.restoreState()


doc = SimpleDocTemplate(
    str(OUTPUT_PATH),
    pagesize=letter,
    rightMargin=0.7 * inch,
    leftMargin=0.7 * inch,
    topMargin=0.7 * inch,
    bottomMargin=0.75 * inch,
    title="Familiar Faces: Integrative Industry Synthesis",
    author="Grant Collings",
)

story = []
counted_body = []


def add_body(text):
    counted_body.append(re.sub(r"<[^>]+>", "", text))
    story.append(Paragraph(text, styles["ReportBody"]))


story.extend(
    [
        Spacer(1, 0.55 * inch),
        Paragraph("Familiar Faces", styles["ReportTitle"]),
        Paragraph("Integrative Industry Synthesis", styles["ReportSubtitle"]),
        Spacer(1, 0.2 * inch),
        Paragraph(
            "A Human-Supervised AI Prototype for Community Paramedicine Outreach Planning",
            styles["ReportSubtitle"],
        ),
        Spacer(1, 0.35 * inch),
        Paragraph("<b>Prepared by:</b> Grant Collings", styles["ReportSubtitle"]),
        Paragraph(
            "Udacity AI for Industry Applications Capstone",
            styles["ReportSubtitle"],
        ),
        Paragraph(REPORT_DATE, styles["ReportSubtitle"]),
        Spacer(1, 0.35 * inch),
        Paragraph(
            "EDUCATIONAL SYNTHETIC-DATA PROTOTYPE",
            styles["ReportSubtitle"],
        ),
        Paragraph(
            "Not approved for clinical use, operational deployment, eligibility "
            "decisions, or autonomous patient contact.",
            styles["SmallNote"],
        ),
        PageBreak(),
        Paragraph("Executive Summary", styles["SectionHeading"]),
    ]
)

add_body(
    "Familiar Faces is a human-supervised artificial intelligence prototype that "
    "explores how a community paramedicine program might prioritize outreach review "
    "for people who frequently use emergency medical services. Frequent emergency "
    "use may reflect unresolved barriers involving primary care, transportation, "
    "medication access, housing, behavioral health, or other social needs. Research "
    "on frequent emergency department users suggests that coordinated interventions "
    "can reduce utilization for some populations, but findings vary and do not "
    "justify assuming that every frequent user needs the same intervention "
    "(Althaus et al., 2011). The prototype therefore frames prediction as a limited "
    "screening aid for authorized staff, not as a clinical judgment or eligibility "
    "decision."
)
add_body(
    "The system integrates descriptive analysis, supervised machine learning, "
    "fairness auditing, controlled nonclinical drafting, automated safety checks, and "
    "mandatory human review. It operates on 3,000 entirely synthetic EHR-like "
    "records and contains no real protected health information. Logistic regression "
    "was selected over random forest because it slightly outperformed the alternative "
    "while remaining easier to explain. Its test ROC AUC was 0.685 and average "
    "precision was 0.579, which represent a moderate advisory signal rather than "
    "deployment-quality evidence. Seven subgroup comparisons exceeded the project's "
    "monitoring threshold, producing a governance status of <b>REVIEW_REQUIRED</b>. "
    "These results demonstrate successful technical integration while also showing "
    "why model outputs must remain subordinate to accountable human judgment."
)

story.append(Paragraph("1. Industry Problem and Scope", styles["SectionHeading"]))
add_body(
    "The industry context is emergency medical services and community paramedicine. "
    "Traditional emergency response is designed to stabilize immediate problems, but "
    "repeated calls may originate from needs that cannot be resolved during a single "
    "encounter. Community paramedicine programs are positioned to coordinate "
    "nonemergency follow-up, connect people with existing resources, and complement "
    "rather than replace primary or emergency care. The realistic AI opportunity is "
    "not to diagnose patients. It is to help authorized personnel organize a review "
    "queue and prepare consistent planning material so that limited outreach capacity "
    "can be used more deliberately."
)
add_body(
    "The system's goal is consequently narrow: estimate simulated outreach response, "
    "surface records for human review, and generate structured nonclinical drafts. "
    "It does not predict medical need, future deterioration, program eligibility, or "
    "expected clinical benefit. It cannot contact a patient, approve services, or "
    "recommend diagnosis or treatment. Those boundaries make the design appropriate "
    "to an educational capstone and reduce the risk that a technically successful "
    "prototype will be mistaken for an operational clinical tool."
)

story.append(
    Paragraph("2. Integration of Prior Capstone Projects", styles["SectionHeading"])
)
add_body(
    "<b>Project 4: machine-learning workflow.</b> The NEMSIS cardiac-arrest project "
    "established the analytical foundation for this synthesis. It required careful "
    "cohort definition, documented preprocessing, reproducible train-test separation, "
    "comparison of models, and interpretation of evaluation metrics in an EMS "
    "context. Familiar Faces carries those practices forward through fixed random "
    "seeds, a stratified split, identical evaluation partitions, and explicit "
    "comparison of logistic regression with random forest. Project 4 also reinforced "
    "that predictive association is not causation. Accordingly, feature importance "
    "in this project is presented as a model explanation and not as proof that any "
    "barrier causes outreach engagement."
)
add_body(
    "<b>Project 5: deep-learning systems.</b> The Seattle Fire 911 forecasting "
    "project demonstrated the value of recurrent neural networks and LSTMs when the "
    "problem contains sufficient sequential structure. It also taught an equally "
    "important design lesson: advanced architecture should follow the problem rather "
    "than precede it. Familiar Faces uses a small tabular synthetic dataset and "
    "requires transparent review of individual factors. A deep neural network would "
    "add complexity without a persuasive data or performance justification. Choosing "
    "simpler classifiers is therefore an intentional transfer of learning from "
    "Project 5, not an omission of cross-domain knowledge."
)
add_body(
    "<b>Project 6: generative AI.</b> The NEISS narrative transformer project showed "
    "how generated text can vary, reproduce undesirable patterns, or appear more "
    "authoritative than its evidence supports. That experience shaped the decision to "
    "use deterministic, template-constrained drafting in this prototype instead of a "
    "live language model. Drafts can summarize documented barriers and matched "
    "resources, but they cannot invent facts or produce clinical language. Automated "
    "validation checks each draft for prohibited content, demographic references, "
    "and the required human-review statement. The generative-AI contribution is thus "
    "expressed through safe content design and validation, while the known risks of "
    "unbounded generation remain outside the system boundary."
)
add_body(
    "<b>Project 7: agentic AI.</b> The earlier Familiar Faces outreach agent supplied "
    "the operational pattern for tool use, permission checking, explicit state, and "
    "human escalation. That project demonstrated that an agent should stop when "
    "permission is denied or uncertain, urgent symptoms are present, a patient cannot "
    "be found, or a tool fails. Project 8 translates those lessons into a broader "
    "system architecture: analytical outputs feed controlled case materials, safety "
    "rules constrain downstream actions, and the final action boundary remains with "
    "authorized personnel. Integrating these projects produced one coherent workflow "
    "rather than four disconnected technical demonstrations."
)

story.append(PageBreak())
story.append(Paragraph("3. System Architecture", styles["SectionHeading"]))
add_body(
    "The architecture separates data preparation, analysis, model evaluation, "
    "fairness governance, drafting, safety validation, and human review. Synthetic "
    "records first pass schema and quality checks. Descriptive analysis then provides "
    "context for model development. Two classifiers are trained on the same "
    "stratified partition, and the selected model produces probabilities used only "
    "to order a review queue. Protected demographic fields are excluded from model "
    "training and generated material; they are retained in a segregated audit path "
    "for post-model disparity assessment. Controlled drafts cannot proceed to any "
    "real-world action, because human approval is a required terminal boundary."
)

architecture = Image(str(ARCHITECTURE_PATH))
architecture.drawHeight = architecture.imageHeight * (6.85 * inch / architecture.imageWidth)
architecture.drawWidth = 6.85 * inch
story.extend(
    [
        Spacer(1, 0.06 * inch),
        Paragraph("Figure 1", styles["APA_Number"]),
        Paragraph(
            "Familiar Faces Human-Supervised System Architecture",
            styles["APA_Title"],
        ),
        architecture,
        Spacer(1, 0.06 * inch),
        Paragraph(
            "<i>Note.</i> Protected attributes follow an audit-only path. No component "
            "authorizes autonomous outreach or clinical action.",
            styles["SmallNote"],
        ),
    ]
)

story.append(
    Paragraph("4. Data, Modeling, and Design Rationale", styles["SectionHeading"])
)
add_body(
    "The dataset contains 3,000 synthetic records and 22 documented fields. Sixteen "
    "nonprotected fields are available to the classifiers. The analysis uses fixed "
    "seeds and a 75/25 stratified train-test split so that results are reproducible "
    "and the positive-class proportion is represented in both partitions. Logistic "
    "regression and random forest were evaluated with ROC AUC, average precision, "
    "accuracy, precision, recall, and F1. Multiple metrics are necessary because a "
    "single aggregate score can conceal operationally important tradeoffs between "
    "missed opportunities and unnecessary staff review."
)
add_body(
    "Logistic regression was selected because it achieved slightly stronger test "
    "performance than random forest and offers coefficients that can be inspected "
    "more directly. The choice favors interpretability, stability, and governance "
    "over marginal complexity. Its predicted probabilities are not converted into "
    "automatic action. They support prioritization within a human workflow where "
    "staff can consider context unavailable to the model. The outcome itself is "
    "simulated outreach response, which is deliberately less consequential than a "
    "clinical target but still insufficient for deployment because synthetic labels "
    "cannot establish real-world validity."
)

results_data = [
    ["System metric", "Result"],
    ["Synthetic records", "3,000"],
    ["Selected model", "Logistic regression"],
    ["Test ROC AUC", "0.685"],
    ["Average precision", "0.579"],
    ["Test F1", "0.558"],
    ["Fairness governance status", "REVIEW_REQUIRED"],
    ["Comparisons requiring review", "7"],
    ["Small audit groups excluded", "2"],
    ["Drafts passing safety validation", "12 of 12"],
    ["Human review required", "Yes"],
]

results_table = Table(
    results_data,
    colWidths=[3.9 * inch, 2.3 * inch],
    repeatRows=1,
)
results_table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#ECEFF1")],
            ),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )
)

story.extend(
    [
        Paragraph("5. Results and Evaluation", styles["SectionHeading"]),
        Paragraph("Table 1", styles["APA_Number"]),
        Paragraph("Integrated Prototype Results", styles["APA_Title"]),
        results_table,
        Spacer(1, 0.12 * inch),
    ]
)
add_body(
    "The pipeline executed successfully and produced all expected artifacts. "
    "Logistic regression achieved a test ROC AUC of 0.6852, average precision of "
    "0.5791, accuracy of 0.6467, and F1 of 0.5576. Random forest achieved a ROC AUC "
    "of 0.6783 and F1 of 0.5179. The selected model therefore offered the stronger "
    "overall balance, but neither result supports autonomous prioritization. The "
    "controlled drafting stage produced 12 case materials, and all 12 passed "
    "automated safety validation. This confirms that the implementation behaves as "
    "designed; it does not demonstrate clinical effectiveness."
)

story.append(
    Paragraph("6. Ethics, Fairness, and Accountability", styles["SectionHeading"])
)
add_body(
    "The central ethical concern is that an outreach model could distribute attention "
    "unequally or encode existing differences in access to care. Excluding protected "
    "attributes from prediction is not enough to guarantee fairness because other "
    "variables can act as proxies and the outcome label can reproduce structural "
    "inequity. Obermeyer et al. (2019) demonstrated that a health-management "
    "algorithm generated substantial racial bias even though its proxy outcome "
    "appeared predictive. Familiar Faces therefore separates prediction from "
    "subgroup auditing and treats observed disparities as reasons for investigation, "
    "not as proof that the system is fair."
)
add_body(
    "The audit compares review-flag rates, true-positive rates, and false-positive "
    "rates across eligible groups. Groups with fewer than 25 test records are "
    "excluded from disparity calculations to avoid overinterpreting unstable "
    "estimates, but exclusion is documented rather than hidden. Seven comparisons "
    "exceeded the project threshold, so the system returns <b>REVIEW_REQUIRED</b>. "
    "That status is a governance signal, not a legal conclusion or a score that can "
    "be averaged away."
)
add_body(
    "Responsible deployment also requires accountability beyond the model. NIST's AI "
    "Risk Management Framework emphasizes governance, contextual mapping, "
    "measurement, and ongoing management across the AI lifecycle (Tabassi, 2023). "
    "The prototype reflects that reasoning through documented scope, reproducible "
    "evaluation, audit outputs, human review, prohibited-use statements, and explicit "
    "escalation. Core safeguards include synthetic data only, no autonomous contact, "
    "no diagnosis or treatment recommendations, no eligibility decisions, and no "
    "use of demographic fields in prediction or drafting."
)

story.append(
    Paragraph("7. Tradeoffs, Constraints, and Deployment Requirements", styles["SectionHeading"])
)
add_body(
    "Several tradeoffs shaped the design. Logistic regression sacrifices some ability "
    "to represent nonlinear relationships in exchange for clearer explanations and "
    "simpler validation. Deterministic drafting sacrifices the flexibility of a live "
    "language model in exchange for repeatability and tighter content control. "
    "Audit-only demographic data support disparity detection but also create privacy "
    "and governance obligations. Human review limits scale and introduces human "
    "variation, yet it preserves accountability where errors could affect access to "
    "support. These choices are appropriate for a high-consequence domain in which "
    "efficiency cannot be the only objective."
)
add_body(
    "The largest constraint is external validity. Synthetic records can demonstrate "
    "code paths, metrics, and governance logic, but they cannot establish prevalence, "
    "causal relationships, model calibration, subgroup performance, or program "
    "benefit in a real population. Moderate model performance, small audit groups, "
    "and the REVIEW_REQUIRED result further limit interpretation. The project has not "
    "undergone clinical, legal, privacy, cybersecurity, accessibility, usability, or "
    "deployment validation."
)
add_body(
    "Before any pilot, an organization would need authorized data-governance "
    "procedures; privacy and legal review; security controls and audit logging; "
    "independent model validation; calibration and subgroup analysis on representative "
    "data; stakeholder and community participation; documented reviewer training; "
    "appeal and escalation procedures; continuous performance and fairness "
    "monitoring; and assigned responsibility for incidents and model retirement. "
    "Outreach effectiveness would also need prospective evaluation against outcomes "
    "that matter to patients and programs, not merely response probability."
)

story.append(
    Paragraph("8. Professional Reflection and Conclusion", styles["SectionHeading"])
)
add_body(
    "This synthesis changed my understanding of AI development from building an "
    "accurate component to designing an accountable system. My EMS experience made "
    "the problem recognizable, but the sequence of capstone projects supplied the "
    "technical discipline to address it responsibly: Project 4 contributed "
    "reproducible analytics, Project 5 clarified when not to use deep learning, "
    "Project 6 exposed the risks of unconstrained generation, and Project 7 supplied "
    "the permission and escalation pattern. The resulting artifact demonstrates "
    "cross-domain fluency while remaining honest about what the evidence does not "
    "support."
)
add_body(
    "Familiar Faces is valuable as a portfolio piece because the code, executed "
    "notebook, architecture, model comparison, fairness artifacts, controlled drafts, "
    "and report all tell the same story. The system works as an educational prototype, "
    "and its limitations are part of the result rather than footnotes to it. The "
    "appropriate next stage is stakeholder and governance evaluation, followed by "
    "carefully authorized research if the concept remains justified. It must not be "
    "used to make decisions about real people or to authorize clinical, eligibility, "
    "or outreach actions."
)

story.extend(
    [
        PageBreak(),
        Paragraph("References", styles["SectionHeading"]),
        Paragraph(
            "Althaus, F., Paroz, S., Hugli, O., Ghali, W. A., Daeppen, J. B., "
            "Peytremann-Bridevaux, I., &amp; Bodenmann, P. (2011). Effectiveness of "
            "interventions targeting frequent users of emergency departments: A "
            "systematic review. <i>Annals of Emergency Medicine, 58</i>(1), "
            "41-52.e42. https://doi.org/10.1016/j.annemergmed.2011.03.007",
            styles["ReferenceBody"],
        ),
        Paragraph(
            "Obermeyer, Z., Powers, B., Vogeli, C., &amp; Mullainathan, S. (2019). "
            "Dissecting racial bias in an algorithm used to manage the health of "
            "populations. <i>Science, 366</i>(6464), 447-453. "
            "https://doi.org/10.1126/science.aax2342",
            styles["ReferenceBody"],
        ),
        Paragraph(
            "Tabassi, E. (2023). <i>Artificial intelligence risk management framework "
            "(AI RMF 1.0)</i> (NIST AI 100-1). National Institute of Standards and "
            "Technology. https://doi.org/10.6028/NIST.AI.100-1",
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
