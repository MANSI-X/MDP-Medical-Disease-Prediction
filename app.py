import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Page setup and visual styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="GlucoGuide | Diabetes ML Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f5f8ff 0%, #f8fbff 48%, #eef7f5 100%); }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }
    .hero {
        padding: 2rem 2.2rem; border-radius: 24px; color: white;
        background: linear-gradient(115deg, #142d5b 0%, #176b78 58%, #27a58b 100%);
        box-shadow: 0 14px 35px rgba(20,45,91,.16); margin-bottom: 1.2rem;
    }
    .hero h1 { font-family: 'Manrope', sans-serif; color: white; font-size: 2.25rem; margin: 0 0 .35rem 0; }
    .hero p { color: #e4f5f4; font-size: 1.02rem; margin: 0; }
    .eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; color: #b9eeea; }
    .soft-card {
        background: rgba(255,255,255,.92); border: 1px solid #e3eaf4;
        border-radius: 18px; padding: 1.1rem 1.25rem; box-shadow: 0 6px 20px rgba(29,55,90,.05);
    }
    .small-muted { color: #63738b; font-size: .88rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e3eaf4; padding: 14px 16px;
        border-radius: 16px; box-shadow: 0 5px 18px rgba(29,55,90,.04);
    }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700;
        background: linear-gradient(100deg, #176b78, #15977f); color: white;
    }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        color: white; border: 0; filter: brightness(1.06);
    }
    h2, h3 { font-family: 'Manrope', sans-serif; color: #17345d; }
    [data-testid="stSidebar"] { background: #f0f5fb; }
    .result-positive { padding: 1rem 1.2rem; border-radius: 14px; background: #fff0ed; border: 1px solid #ffd2c9; color: #8f3325; }
    .result-negative { padding: 1rem 1.2rem; border-radius: 14px; background: #eaf8f1; border: 1px solid #c6ead7; color: #176342; }
</style>
""", unsafe_allow_html=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
TARGET = "Outcome"
DATASET_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# ---------------------------------------------------------
# Data and model
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_remote_data():
    return pd.read_csv(DATASET_URL)

@st.cache_resource(show_spinner="Training the machine-learning model...")
def train_model(data_frame):
    data = data_frame[FEATURES + [TARGET]].copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].isin([0, 1])]
    data[ZERO_AS_MISSING] = data[ZERO_AS_MISSING].replace(0, np.nan)

    X = data[FEATURES]
    y = data[TARGET].astype(int)

    if len(data) < 50 or y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Dataset must contain at least 50 usable rows and at least 5 examples in each class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "matrix": confusion_matrix(y_test, predictions),
    }

    # Cross-validation is performed only on the training portion.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

    return pipeline, metrics, cv_scores, len(data)

def clean_input(frame):
    frame = frame.copy()
    frame[ZERO_AS_MISSING] = frame[ZERO_AS_MISSING].replace(0, np.nan)
    return frame

# ---------------------------------------------------------
# Header and sidebar
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">Machine Learning • Educational Project</div>
  <h1>🩺 GlucoGuide</h1>
  <p>A clean, simple diabetes classification demo powered by a Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Project controls")
    st.caption("Choose the dataset used to train the demonstration model.")
    uploaded_csv = st.file_uploader("Upload a compatible diabetes CSV (optional)", type=["csv"])
    st.markdown("---")
    st.markdown("### 📌 Input guide")
    st.caption("Enter values from a recent laboratory report where possible. Do not guess missing measurements.")
    st.markdown("---")
    st.markdown("**Dataset fields required**")
    st.caption(", ".join(FEATURES + [TARGET]))
    st.markdown("---")
    st.caption("This app is for learning and demonstration only—not for diagnosis.")

st.warning(
    "⚠️ **Medical safety notice:** This educational demo is not a medical device. "
    "Its output is not a diagnosis, a clinically validated risk score, or a substitute for a blood test "
    "and advice from a qualified healthcare professional."
)

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
try:
    if uploaded_csv is not None:
        raw_data = pd.read_csv(uploaded_csv)
        dataset_label = f"Uploaded file: {uploaded_csv.name}"
    else:
        raw_data = load_remote_data()
        dataset_label = "Sample diabetes CSV (loaded from the project dataset URL)"

    missing_cols = [c for c in FEATURES + [TARGET] if c not in raw_data.columns]
    if missing_cols:
        st.error("The selected CSV is missing required columns: " + ", ".join(missing_cols))
        st.stop()

    model, metrics, cv_scores, row_count = train_model(raw_data)

except Exception as error:
    st.error(f"Could not prepare the model: {error}")
    st.info("Check your internet connection or upload a CSV containing the required columns.")
    st.stop()

# ---------------------------------------------------------
# Overview metrics
# ---------------------------------------------------------
st.markdown("### 📊 Model overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Usable records", f"{row_count:,}")
m2.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
m3.metric("5-fold CV accuracy", f"{cv_scores.mean():.1%}", f"± {cv_scores.std():.1%}")
m4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

st.caption(
    f"Dataset: {dataset_label}. The holdout test set contains 20% of usable records. "
    "Scores can change with a different dataset and split; they do not represent clinical accuracy."
)

tab_predict, tab_model, tab_about = st.tabs(["🧑‍⚕️ Patient estimator", "📈 Model performance", "ℹ️ About this project"])

# ---------------------------------------------------------
# Prediction form
# ---------------------------------------------------------
with tab_predict:
    st.markdown("### Enter patient measurements")
    st.write("Complete the fields below, then select **Estimate category**. Values are used only for this prediction in the running app.")

    with st.form("patient_form", clear_on_submit=False):
        left, right = st.columns(2)
        with left:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1,
                                           help="Number of times pregnant.")
            glucose = st.number_input("Glucose (mg/dL)", min_value=1.0, max_value=300.0, value=120.0, step=1.0,
                                      help="Plasma glucose measurement. Use a lab result.")
            blood_pressure = st.number_input("Diastolic blood pressure (mm Hg)", min_value=1.0, max_value=200.0, value=70.0, step=1.0)
            skin = st.number_input("Skin thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0,
                                   help="Enter 0 only if this measurement is unavailable.")
        with right:
            insulin = st.number_input("Insulin (µU/mL)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0,
                                      help="Enter 0 only if this measurement is unavailable.")
            bmi = st.number_input("BMI (kg/m²)", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
            dpf = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=3.0, value=0.5, step=0.01,
                                  help="Dataset-specific family-history score.")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

        submitted = st.form_submit_button("✨ Estimate category", use_container_width=True)

    if submitted:
        patient = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }], columns=FEATURES)
        patient = clean_input(patient)
        predicted_class = int(model.predict(patient)[0])
        positive_probability = float(model.predict_proba(patient)[0][list(model.classes_).index(1)])

        st.markdown("### Result")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.metric("Model output score", f"{positive_probability:.1%}")
            st.caption("Model-estimated probability for the positive class—not an individual's actual medical risk.")
        with r2:
            if predicted_class == 1:
                st.markdown(
                    "<div class='result-positive'><b>Positive class predicted</b><br>"
                    "The model classified these entries in the diabetes-positive category. "
                    "This does not confirm diabetes.</div>", unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='result-negative'><b>Negative class predicted</b><br>"
                    "The model classified these entries in the diabetes-negative category. "
                    "This does not rule out diabetes.</div>", unsafe_allow_html=True
                )
        st.progress(positive_probability)
        st.info("For a real health concern, consult a clinician and use recommended laboratory testing (such as HbA1c or fasting plasma glucose). Do not make treatment decisions from this demo.")

# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------
with tab_model:
    st.markdown("### How did the model perform?")
    st.write("The metrics below are calculated on a held-out test set that was not used to fit the model.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{metrics['precision']:.1%}")
    c2.metric("Recall", f"{metrics['recall']:.1%}")
    c3.metric("F1 score", f"{metrics['f1']:.1%}")
    c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("#### Confusion matrix")
    matrix_df = pd.DataFrame(
        metrics["matrix"],
        index=["Actual negative", "Actual positive"],
        columns=["Predicted negative", "Predicted positive"],
    )
    st.dataframe(matrix_df, use_container_width=True)
    st.markdown("#### Cross-validation accuracy")
    cv_df = pd.DataFrame({"Fold": [f"Fold {i}" for i in range(1, len(cv_scores) + 1)],
                          "Accuracy": cv_scores})
    st.bar_chart(cv_df.set_index("Fold"))
    st.caption(
        "A single dataset and test split cannot prove 95% real-world accuracy. "
        "For a credible performance claim, evaluate on a large, independent, representative dataset "
        "and report sensitivity, specificity, calibration, and confidence intervals."
    )

# ---------------------------------------------------------
# About
# ---------------------------------------------------------
with tab_about:
    st.markdown("### About GlucoGuide")
    st.write(
        "GlucoGuide is an academic machine-learning demonstration. It uses a Random Forest classifier "
        "with median imputation for unavailable measurements. A stratified 80/20 split is used for "
        "the holdout evaluation, and 5-fold cross-validation is run on the training portion."
    )
    st.markdown("**Features used by the model**")
    st.write(", ".join(FEATURES))
    st.markdown("**What the result means**")
    st.write(
        "The positive/negative label is the model's classification for the supplied values. "
        "The displayed score is the model's probability estimate for its positive class; it is not "
        "a calibrated or clinically validated probability of disease."
    )
    st.markdown("**Data and privacy**")
    st.write(
        "The sample dataset is loaded from the configured CSV URL unless a CSV is uploaded. "
        "This code does not intentionally save patient form entries to a database. "
        "If deployed publicly, review the hosting provider's logging and privacy settings before use."
    )

st.markdown("---")
st.caption("GlucoGuide • Academic project • Not for diagnosis, treatment, or emergency decisions.")
