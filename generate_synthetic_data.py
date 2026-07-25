from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260724
N_RECORDS = 3000
OUTPUT_PATH = Path("data/familiar_faces_synthetic.csv")

rng = np.random.default_rng(SEED)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


# Demographic attributes are retained only for fairness auditing.
age_group = rng.choice(
    ["18-34", "35-49", "50-64", "65-79", "80+"],
    size=N_RECORDS,
    p=[0.10, 0.16, 0.26, 0.30, 0.18],
)
sex = rng.choice(
    ["Female", "Male", "Another_or_unknown"],
    size=N_RECORDS,
    p=[0.49, 0.49, 0.02],
)
race_ethnicity = rng.choice(
    [
        "White",
        "Black",
        "Hispanic",
        "Asian",
        "Native_American",
        "Multiracial_or_other",
    ],
    size=N_RECORDS,
    p=[0.54, 0.17, 0.16, 0.07, 0.02, 0.04],
)

vulnerability = rng.normal(0, 1, N_RECORDS)

housing_instability = rng.binomial(
    1, sigmoid(-1.05 + 0.95 * vulnerability)
)
transportation_barrier = rng.binomial(
    1, sigmoid(-0.75 + 0.80 * vulnerability)
)
food_insecurity = rng.binomial(
    1, sigmoid(-0.90 + 0.85 * vulnerability)
)
medication_access_barrier = rng.binomial(
    1, sigmoid(-1.00 + 0.75 * vulnerability)
)
behavioral_health_support_need = rng.binomial(
    1, sigmoid(-0.95 + 0.70 * vulnerability)
)
substance_use_support_need = rng.binomial(
    1, sigmoid(-1.35 + 0.85 * vulnerability)
)
primary_care_connected = rng.binomial(
    1, sigmoid(0.95 - 0.55 * vulnerability)
)

social_need_count = (
    housing_instability
    + transportation_barrier
    + food_insecurity
    + medication_access_barrier
    + behavioral_health_support_need
    + substance_use_support_need
)

age_index = pd.Series(age_group).map(
    {"18-34": 0, "35-49": 1, "50-64": 2, "65-79": 3, "80+": 4}
).to_numpy()

chronic_condition_count = np.clip(
    rng.poisson(0.5 + 0.55 * age_index + 0.20 * sigmoid(vulnerability)),
    0,
    8,
)

ems_calls_12m = rng.poisson(
    0.8
    + 0.55 * social_need_count
    + 0.30 * chronic_condition_count
    + 0.65 * housing_instability
)
ed_visits_12m = rng.poisson(
    0.4 + 0.50 * ems_calls_12m + 0.25 * medication_access_barrier
)
hospital_admissions_12m = np.clip(
    rng.poisson(0.15 + 0.12 * ed_visits_12m + 0.08 * chronic_condition_count),
    0,
    12,
)
missed_appointments_12m = np.clip(
    rng.poisson(
        0.3
        + 0.55 * transportation_barrier
        + 0.45 * housing_instability
        + 0.30 * food_insecurity
    ),
    0,
    20,
)

days_since_last_911_call = np.where(
    ems_calls_12m > 0,
    np.clip(
        rng.exponential(scale=120 / np.maximum(ems_calls_12m, 1)),
        0,
        365,
    ).astype(int),
    rng.integers(180, 366, N_RECORDS),
)

prior_outreach_attempts = np.clip(
    rng.poisson(0.10 + 0.18 * ems_calls_12m),
    0,
    10,
)
engagement_probability = sigmoid(
    -0.45
    + 0.80 * primary_care_connected
    - 0.30 * housing_instability
    + 0.20 * transportation_barrier
)
prior_outreach_engaged = np.where(
    prior_outreach_attempts > 0,
    rng.binomial(1, engagement_probability),
    0,
)

# The outcome represents a simulated historical benefit from outreach.
# Race, ethnicity, sex, and patient ID are deliberately excluded.
response_probability = sigmoid(
    -1.70
    + 0.16 * ems_calls_12m
    + 0.22 * social_need_count
    + 0.35 * medication_access_barrier
    + 0.28 * (1 - primary_care_connected)
    + 0.55 * prior_outreach_engaged
    + 0.07 * chronic_condition_count
)
outreach_response_90d = rng.binomial(1, response_probability)

df = pd.DataFrame(
    {
        "synthetic_patient_id": [
            f"FF-{number:05d}" for number in range(1, N_RECORDS + 1)
        ],
        "age_group": age_group,
        "sex": sex,
        "race_ethnicity": race_ethnicity,
        "housing_instability": housing_instability,
        "transportation_barrier": transportation_barrier,
        "food_insecurity": food_insecurity,
        "medication_access_barrier": medication_access_barrier,
        "behavioral_health_support_need": behavioral_health_support_need,
        "substance_use_support_need": substance_use_support_need,
        "primary_care_connected": primary_care_connected,
        "chronic_condition_count": chronic_condition_count,
        "ems_calls_12m": ems_calls_12m,
        "ed_visits_12m": ed_visits_12m,
        "hospital_admissions_12m": hospital_admissions_12m,
        "missed_appointments_12m": missed_appointments_12m,
        "days_since_last_911_call": days_since_last_911_call,
        "prior_outreach_attempts": prior_outreach_attempts,
        "prior_outreach_engaged": prior_outreach_engaged,
        "social_need_count": social_need_count,
        "outreach_response_90d": outreach_response_90d,
        "data_source": "synthetic",
    }
)

assert df["synthetic_patient_id"].is_unique
assert not df.isna().any().any()
assert set(df["data_source"]) == {"synthetic"}
assert df["days_since_last_911_call"].between(0, 365).all()
assert df["social_need_count"].between(0, 6).all()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Created: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Unique patient IDs: {df['synthetic_patient_id'].nunique():,}")
print(f"Outreach response rate: {df['outreach_response_90d'].mean():.2%}")
print(f"Mean EMS calls: {df['ems_calls_12m'].mean():.2f}")
print("Synthetic-data validation: PASSED")
