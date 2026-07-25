from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUTPUT_PATH = Path("Integrative_Industry_Synthesis_Report.pdf")
ARCHITECTURE_PATH = Path("figures/familiar_faces_system_architecture.png")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, alignment=TA_CENTER, textColor=colors.HexColor("#263238"), spaceAfter=14))
styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#546E7A"), spaceAfter=18))
styles.add(ParagraphStyle(name="SectionHeading", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=colors.HexColor("#B7472A"), spaceBefore=10, spaceAfter=8))
styles.add(ParagraphStyle(name="Subheading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#263238"), spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="ReportBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=15, textColor=colors.HexColor("#263238"), spaceAfter=8))
styles.add(ParagraphStyle(name="SmallNote", parent=styles["BodyText"], fontName="Helvetica-Oblique", fontSize=8.5, leading=12, textColor=colors.HexColor("#546E7A"), spaceAfter=6))

def draw_page(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#CFD8DC"))
    canvas.line(doc.leftMargin, 0.55 * inch, width - doc.rightMargin, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607D8B"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "Familiar Faces | Integrative Industry Synthesis")
    canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
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

story.extend([
    Spacer(1, 0.65 * inch),
    Paragraph("Familiar Faces", styles["ReportTitle"]),
    Paragraph("Integrative Industry Synthesis", styles["ReportSubtitle"]),
    Spacer(1, 0.25 * inch),
    Paragraph("A Human-Supervised AI Prototype for Community Paramedicine Outreach Planning", styles["ReportSubtitle"]),
    Spacer(1, 0.45 * inch),
    Paragraph("<b>Prepared by:</b> Grant Collings", styles["ReportSubtitle"]),
    Paragraph("Udacity AI for Industry Applications Capstone", styles["ReportSubtitle"]),
    Spacer(1, 0.45 * inch),
    Paragraph("EDUCATIONAL SYNTHETIC-DATA PROTOTYPE", styles["ReportSubtitle"]),
    Paragraph("Not approved for clinical use, operational deployment, eligibility decisions, or autonomous patient contact.", styles["SmallNote"]),
    PageBreak(),
])

story.extend([
    Paragraph("Executive Summary", styles["SectionHeading"]),
    Paragraph("Familiar Faces is a human-supervised AI prototype exploring how community paramedicine programs might prioritize outreach for frequent users of emergency medical services. It uses 3,000 entirely synthetic EHR-like records, with no real patient data or protected health information.", styles["ReportBody"]),
    Paragraph("The integrated workflow combines descriptive statistical analysis, machine-learning comparison, protected-group fairness auditing, controlled nonclinical drafting, automated safety validation, and mandatory human review.", styles["ReportBody"]),
    Paragraph("Logistic regression was selected with a test ROC AUC of 0.685 and average precision of 0.579. Fairness monitoring produced a REVIEW_REQUIRED status, so model outputs cannot be treated as deployment-ready decisions.", styles["ReportBody"]),
    Paragraph("<b>Overall conclusion:</b> The project demonstrates technically successful integration with explicit governance boundaries, but it remains an educational synthetic-data prototype and is not approved for clinical or operational use.", styles["ReportBody"]),
])

story.extend([
    Paragraph("1. Problem Context and Project Objectives", styles["SectionHeading"]),
    Paragraph("Frequent use of emergency medical services can reflect unmet needs involving health-care access, transportation, medication access, housing stability, behavioral-health support, and other social barriers. Traditional emergency response may address the immediate call without resolving the conditions contributing to repeated utilization.", styles["ReportBody"]),
    Paragraph("Familiar Faces explores a community-paramedicine workflow that could help authorized personnel identify synthetic records for human review and prepare structured, nonclinical outreach-planning materials.", styles["ReportBody"]),
    Paragraph("<b>Project objectives:</b> integrate statistical analysis, predictive modeling, fairness auditing, controlled drafting, and safety validation; demonstrate traceable human oversight; and define the governance requirements that would precede any real-world pilot.", styles["ReportBody"]),
    Paragraph("The modeled outcome is simulated outreach response, not medical need, clinical risk, eligibility, or expected health benefit.", styles["SmallNote"]),
])

story.append(PageBreak())

architecture = Image(str(ARCHITECTURE_PATH))
architecture.drawHeight = architecture.imageHeight * (6.9 * inch / architecture.imageWidth)
architecture.drawWidth = 6.9 * inch

story.extend([
    Paragraph("2. System Architecture and Integrated Workflow", styles["SectionHeading"]),
    Paragraph("The architecture separates synthetic-data preparation, analytical processing, governance controls, controlled drafting, automated safety validation, and human review. No component independently authorizes outreach or other real-world action.", styles["ReportBody"]),
    Spacer(1, 0.08 * inch),
    architecture,
    Spacer(1, 0.08 * inch),
    Paragraph("Figure 1. Familiar Faces human-supervised system architecture.", styles["SmallNote"]),
    Paragraph("Protected demographic attributes are excluded from model training and generated materials. They are retained only for post-model fairness auditing. Every selected record and draft remains subject to authorized human review.", styles["ReportBody"]),
    PageBreak(),
])

story.extend([
    Paragraph("3. Data and Analytical Methods", styles["SectionHeading"]),
    Paragraph("The project uses 3,000 entirely synthetic EHR-like records containing 22 documented fields. No real patient data or protected health information is included.", styles["ReportBody"]),
    Paragraph("Sixteen nonprotected features are used for model development. Age group, sex, and race/ethnicity are excluded from model predictors and generated materials; they are retained only for post-model fairness auditing.", styles["ReportBody"]),
    Paragraph("The workflow validates the synthetic dataset, summarizes utilization and simulated outcomes, compares logistic regression and random forest classifiers on identical stratified partitions, audits subgroup performance, and creates constrained outreach-planning drafts.", styles["ReportBody"]),
    Paragraph("Because the data and outcomes are synthetic, all findings are demonstrations of analytical methods rather than evidence about real patients or program effectiveness.", styles["SmallNote"]),
])

results_data = [
    ["System metric", "Result"],
    ["Synthetic records", "3,000"],
    ["Selected model", "Logistic regression"],
    ["Test ROC AUC", "0.685"],
    ["Average precision", "0.579"],
    ["Fairness governance status", "REVIEW_REQUIRED"],
    ["Fairness comparisons requiring review", "7"],
    ["Small audit groups excluded", "2"],
    ["Controlled drafts passing validation", "12 of 12"],
    ["Human review required", "Yes"],
]

results_table = Table(results_data, colWidths=[3.9 * inch, 2.3 * inch], repeatRows=1)
results_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#ECEFF1")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))

story.extend([
    Paragraph("4. Results and Evaluation", styles["SectionHeading"]),
    Paragraph("The integrated pipeline completed successfully and produced the following principal results:", styles["ReportBody"]),
    results_table,
    Spacer(1, 0.12 * inch),
    Paragraph("The selected model provides a moderate advisory signal rather than reliable autonomous decision-making. The fairness status requires contextual investigation before any further consideration of real-world use.", styles["ReportBody"]),
])

story.extend([
    Paragraph("5. Fairness Governance and Safety Controls", styles["SectionHeading"]),
    Paragraph("Protected demographic attributes were excluded from model training and retained only for post-model auditing. The audit compared human-review flag rates, true-positive rates, and false-positive rates across eligible demographic groups.", styles["ReportBody"]),
    Paragraph("Groups with fewer than 25 test records were excluded from disparity calculations. Seven eligible comparisons exceeded the project monitoring threshold, producing an overall governance status of <b>REVIEW_REQUIRED</b>. This threshold is a screening mechanism, not a legal definition of fairness or evidence of deployment readiness.", styles["ReportBody"]),
    Paragraph("Core safeguards include synthetic data only, mandatory human approval, no clinical diagnosis or treatment recommendations, no autonomous eligibility decisions, no autonomous patient contact, and fairness review whenever monitoring thresholds are exceeded.", styles["ReportBody"]),
    Paragraph("These controls reduce foreseeable risk within the educational prototype but do not replace formal privacy, legal, clinical, cybersecurity, or organizational review.", styles["SmallNote"]),
])

story.extend([
    Paragraph("6. Limitations and Real-World Requirements", styles["SectionHeading"]),
    Paragraph("This project has important limitations: all records and outcomes are synthetic; the modeled outcome is simulated outreach response rather than clinical benefit; predictive performance is moderate; subgroup differences require contextual investigation; and some demographic groups have small test samples.", styles["ReportBody"]),
    Paragraph("Feature importance does not establish causality. The controlled drafting component is deterministic rather than live large-language-model inference. The system has not undergone clinical, legal, privacy, security, accessibility, usability, or deployment validation.", styles["ReportBody"]),
    Paragraph("Before any real-world pilot, the organization would require formal HIPAA, privacy, and legal review; authorized data-governance procedures; cybersecurity controls and audit logging; stakeholder and community participation; independent model validation; human-review and escalation procedures; ongoing fairness monitoring; and defined accountability and incident-response processes.", styles["ReportBody"]),
    Paragraph("Until those requirements are satisfied, Familiar Faces must remain an educational synthetic-data prototype.", styles["SmallNote"]),
])

story.extend([
    Paragraph("7. Conclusion and Responsible-Use Statement", styles["SectionHeading"]),
    Paragraph("Familiar Faces demonstrates that statistical analysis, machine learning, fairness governance, controlled drafting, safety validation, and human oversight can be integrated into one traceable community-paramedicine concept.", styles["ReportBody"]),
    Paragraph("The prototype completed its intended educational workflow, but its moderate predictive performance and REVIEW_REQUIRED fairness status reinforce that the outputs must remain advisory and subject to human judgment.", styles["ReportBody"]),
    Paragraph("<b>Responsible-use statement:</b> This system must not be used to make decisions about real people. It is not a deployable clinical system and does not authorize diagnosis, treatment, eligibility decisions, outreach, or autonomous action.", styles["ReportBody"]),
    Paragraph("The appropriate next stage is governance and stakeholder evaluation, not operational deployment.", styles["SmallNote"]),
])

doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
print(f"Created PDF: {OUTPUT_PATH}")

