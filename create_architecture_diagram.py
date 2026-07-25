from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_PATH = Path("figures/familiar_faces_system_architecture.png")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis("off")


def add_box(x, y, width, height, text, color, fontsize=11):
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=color,
        edgecolor="#263238",
        linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        wrap=True,
    )
    return box


def add_arrow(start, end, connectionstyle="arc3,rad=0"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": "#37474f",
            "linewidth": 1.8,
            "shrinkA": 4,
            "shrinkB": 4,
            "connectionstyle": connectionstyle,
        },
    )


ax.text(
    7,
    8.05,
    "Familiar Faces Integrated AI System Architecture",
    ha="center",
    va="center",
    fontsize=20,
    fontweight="bold",
    color="#263238",
)

ax.text(
    7,
    7.65,
    "Educational synthetic-data prototype with mandatory human oversight",
    ha="center",
    va="center",
    fontsize=11,
    color="#546e7a",
)

add_box(
    0.4,
    3.35,
    2.0,
    1.25,
    "Synthetic EHR-like\nDataset and\nData Dictionary",
    "#d7ebf7",
)

add_box(
    3.0,
    3.35,
    2.0,
    1.25,
    "Schema and\nSafety-Boundary\nValidation",
    "#d9ead3",
)

add_box(
    5.8,
    5.4,
    2.2,
    1.15,
    "Descriptive\nStatistical Analysis",
    "#fff2cc",
)

add_box(
    5.8,
    3.35,
    2.2,
    1.25,
    "Machine-Learning\nModel Comparison and\nAdvisory Prioritization",
    "#fce5cd",
)

add_box(
    5.8,
    1.3,
    2.2,
    1.15,
    "Controlled,\nConstrained Drafting",
    "#eadcf8",
)

add_box(
    8.9,
    4.25,
    2.2,
    1.25,
    "Protected-Group\nFairness Audit and\nGovernance Assessment",
    "#f4cccc",
)

add_box(
    8.9,
    1.3,
    2.2,
    1.15,
    "Automated Draft\nSafety Validation",
    "#d9d2e9",
)

add_box(
    11.7,
    3.0,
    1.9,
    1.75,
    "Human Reviewer\n\nConfirms context,\nconsent, resources,\nand appropriate action",
    "#cfe2f3",
    fontsize=10,
)

add_arrow((2.4, 3.98), (3.0, 3.98))
add_arrow((5.0, 4.25), (5.8, 5.80))
add_arrow((5.0, 3.98), (5.8, 3.98))
add_arrow((8.0, 4.15), (8.9, 4.75))
add_arrow((8.0, 3.55), (8.9, 2.00))
add_arrow((5.0, 3.70), (5.8, 1.90))
add_arrow((8.0, 1.88), (8.9, 1.88))
add_arrow((11.1, 4.75), (11.7, 4.15))
add_arrow((11.1, 1.88), (11.7, 3.35))
add_arrow((8.0, 5.90), (11.7, 4.55), "angle3,angleA=12,angleB=90")

boundary = FancyBboxPatch(
    (1.0, 0.12),
    12.0,
    0.82,
    boxstyle="round,pad=0.04,rounding_size=0.10",
    facecolor="#eceff1",
    edgecolor="#b7472a",
    linewidth=2.0,
)

ax.add_patch(boundary)
ax.text(
    7,
    0.53,
    (
        "Governance boundary: no clinical decisions, no autonomous eligibility,\n"
        "no autonomous contact, and no use of demographic fields as predictors"
    ),
    ha="center",
    va="center",
    fontsize=10.0,
    fontweight="bold",
    color="#8b2f1c",
)

fig.tight_layout()
fig.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

print("Created:", OUTPUT_PATH)
print("Architecture diagram: PASSED")



