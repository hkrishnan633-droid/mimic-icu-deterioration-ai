from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ICU Deterioration Early-Warning System",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

FEATURE_PATH = RESULTS_DIR / "model_features_2h.csv"
RISK_PATH = RESULTS_DIR / "xgboost_oof_risk_table.csv"
DYNAMIC_PATH = RESULTS_DIR / "dynamic_risk_predictions.csv"
LANDMARK_PATH = RESULTS_DIR / "dynamic_landmark_summary.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    features = pd.read_csv(FEATURE_PATH)
    risk = pd.read_csv(RISK_PATH)
    dynamic = pd.read_csv(DYNAMIC_PATH)
    landmark = pd.read_csv(LANDMARK_PATH)

    return features, risk, dynamic, landmark


try:
    features, risk, dynamic, landmark = load_data()

except FileNotFoundError as e:

    st.error(
        "A required results file could not be found."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "Explainable ICU Deterioration Early-Warning System"
)

st.caption(
    "Proof-of-concept using the MIMIC-IV Demo dataset"
)

st.warning(
    "RESEARCH DEMONSTRATION ONLY — NOT FOR CLINICAL USE. "
    "The displayed probabilities are exploratory machine-learning "
    "outputs and are not validated clinical risk scores."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Patient Selection")

available_stays = sorted(
    dynamic["stay_id"]
    .dropna()
    .unique()
    .tolist()
)

selected_stay = st.sidebar.selectbox(
    "Select ICU stay",
    available_stays
)

patient_dynamic = (
    dynamic[
        dynamic["stay_id"] == selected_stay
    ]
    .sort_values("landmark_hour")
    .copy()
)


# ============================================================
# IDENTIFIERS
# ============================================================

subject_id = patient_dynamic[
    "subject_id"
].iloc[0]

hadm_id = patient_dynamic[
    "hadm_id"
].iloc[0]


st.sidebar.markdown("---")

st.sidebar.write(
    f"**Subject ID:** {subject_id}"
)

st.sidebar.write(
    f"**Hospital admission:** {hadm_id}"
)

st.sidebar.write(
    f"**ICU stay:** {selected_stay}"
)


# ============================================================
# MATCH FIXED 2-HOUR RISK
# ============================================================

patient_risk = risk[
    risk["stay_id"] == selected_stay
].copy()


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

st.header("Patient Overview")

col1, col2, col3, col4 = st.columns(4)


# Current dynamic probability = latest available landmark
latest_row = patient_dynamic.iloc[-1]

latest_probability = float(
    latest_row["risk_probability"]
)

latest_landmark = int(
    latest_row["landmark_hour"]
)


col1.metric(
    "Latest Estimated Probability",
    f"{latest_probability:.1%}"
)

col2.metric(
    "Latest Landmark",
    f"{latest_landmark} h"
)

col3.metric(
    "Trajectory Points",
    len(patient_dynamic)
)


# Determine whether any dynamic prediction window contains
# deterioration
ever_future_event = int(
    patient_dynamic[
        "future_deterioration"
    ].max()
)

col4.metric(
    "Observed Future Deterioration",
    "Yes" if ever_future_event == 1 else "No"
)


# ============================================================
# DYNAMIC TRAJECTORY
# ============================================================

st.header("Dynamic Deterioration Probability")

trajectory_chart = (
    patient_dynamic[
        [
            "landmark_hour",
            "risk_probability"
        ]
    ]
    .set_index("landmark_hour")
)

st.line_chart(
    trajectory_chart,
    y="risk_probability"
)

st.caption(
    "Each point is an out-of-fold probability from a "
    "landmark-specific Random Forest model. The model uses only "
    "information available up to that landmark and predicts "
    "deterioration during the following six hours."
)


# ============================================================
# TRAJECTORY TABLE
# ============================================================

with st.expander(
    "View trajectory data"
):

    trajectory_display = patient_dynamic[
        [
            "landmark_hour",
            "risk_probability",
            "future_deterioration",
            "fold"
        ]
    ].copy()

    trajectory_display[
        "risk_probability"
    ] = (
        trajectory_display[
            "risk_probability"
        ] * 100
    ).round(1)

    trajectory_display = (
        trajectory_display.rename(
            columns={
                "landmark_hour":
                    "Landmark (hours)",
                "risk_probability":
                    "Estimated probability (%)",
                "future_deterioration":
                    "Deterioration in next 6 h",
                "fold":
                    "CV fold"
            }
        )
    )

    st.dataframe(
        trajectory_display,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FIXED 2-HOUR XGBOOST MODEL
# ============================================================

st.header("2-Hour Landmark Model")

if not patient_risk.empty:

    fixed_probability = float(
        patient_risk[
            "risk_probability"
        ].iloc[0]
    )

    actual = int(
        patient_risk[
            "deterioration"
        ].iloc[0]
    )

    predicted = int(
        patient_risk[
            "predicted_class"
        ].iloc[0]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "XGBoost Probability",
        f"{fixed_probability:.1%}"
    )

    c2.metric(
        "Model Classification",
        "Positive"
        if predicted == 1
        else "Negative"
    )

    c3.metric(
        "Observed Outcome",
        "Deterioration"
        if actual == 1
        else "No deterioration"
    )

    if actual == 1 and predicted == 1:
        case_type = "True Positive"

    elif actual == 0 and predicted == 0:
        case_type = "True Negative"

    elif actual == 0 and predicted == 1:
        case_type = "False Positive"

    else:
        case_type = "False Negative"

    st.info(
        f"**Model behaviour for this stay:** {case_type}"
    )

else:

    st.info(
        "This ICU stay does not have an eligible "
        "2-hour XGBoost prediction."
    )


# ============================================================
# PHYSIOLOGICAL FEATURES
# ============================================================

st.header("Physiological Features at the 2-Hour Landmark")

patient_features = features[
    features["stay_id"] == selected_stay
].copy()

if patient_features.empty:

    st.info(
        "No 2-hour feature record is available "
        "for this ICU stay."
    )

else:

    patient_features = patient_features.iloc[0]

    feature_groups = {
        "Heart Rate": "heart_rate",
        "Respiratory Rate": "respiratory_rate",
        "SpO₂": "spo2",
        "Systolic BP": "systolic_bp",
        "Diastolic BP": "diastolic_bp",
        "Mean BP": "mean_bp",
        "Temperature": "temperature"
    }

    summary_rows = []

    for display_name, prefix in feature_groups.items():

        row = {
            "Variable": display_name
        }

        found = False

        for stat in [
            "mean",
            "min",
            "max",
            "last",
            "count"
        ]:

            column = f"{prefix}_{stat}"

            if column in patient_features.index:

                row[
                    stat.capitalize()
                ] = patient_features[column]

                found = True

        if found:
            summary_rows.append(row)

    if summary_rows:

        physiology_table = pd.DataFrame(
            summary_rows
        )

        st.dataframe(
            physiology_table,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Selected physiological feature columns "
            "were not found."
        )


# ============================================================
# MODEL CONTEXT
# ============================================================

st.header("Model Context")

context1, context2, context3 = st.columns(3)

context1.metric(
    "Development ICU Stays",
    "99"
)

context2.metric(
    "Development Patients",
    "79"
)

context3.metric(
    "Deterioration Events",
    "18"
)

st.markdown(
    """
**Fixed 2-hour model**

- Observation window: ICU admission to 2 hours
- Prediction window: >2 to 8 hours
- Validation: patient-separated grouped cross-validation
- Models evaluated: Logistic Regression, Random Forest and XGBoost

**Dynamic demonstration**

- Landmarks: 1, 2, 3 and 4 hours
- Prediction horizon: following 6 hours
- Model: Random Forest
- Validation: patient-separated 3-fold grouped cross-validation
"""
)


# ============================================================
# LIMITATIONS
# ============================================================

st.header("Important Limitations")

st.markdown(
    """
- This system was developed using the **MIMIC-IV Demo dataset**.
- The sample size and number of deterioration events are very small.
- Later dynamic landmarks contain particularly few positive events.
- Predictions are exploratory and may be unstable.
- The models have not undergone external validation.
- The probabilities should not be interpreted as calibrated clinical risk.
- Feature importance represents predictive association, not causation.
- Measurement frequency may itself contribute predictive information.
- This application must not be used for patient-care decisions.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Explainable Machine-Learning Early-Warning System for "
    "Clinical Deterioration — MIMIC-IV Demo proof-of-concept."
)