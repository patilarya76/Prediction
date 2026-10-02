"""
app.py
------
Interactive Streamlit Web Application for Titanic Survival Prediction.
Allows recruiters and users to input custom passenger attributes,
test both trained models (Logistic Regression & Random Forest) in real time,
and explore model metrics, confusion matrices, and feature importances.

Deployable for FREE to Streamlit Community Cloud or Hugging Face Spaces.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor | ML Project",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load serialized models and artifacts."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    lr_path = os.path.join(models_dir, "logistic_regression.joblib")
    rf_path = os.path.join(models_dir, "random_forest.joblib")

    if not (os.path.exists(lr_path) and os.path.exists(rf_path)):
        st.error("Model artifacts not found in 'models/'. Please run 'python src/train_model.py' first.")
        st.stop()

    lr_model = joblib.load(lr_path)
    rf_model = joblib.load(rf_path)
    return lr_model, rf_model, models_dir


lr_model, rf_model, models_dir = load_models()

# Sidebar: Project Info & Model Selection
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/640px-RMS_Titanic_3.jpg", use_container_width=True)
    st.title("🚢 Navigation")
    app_mode = st.radio("Go to:", ["🔮 Live Survival Predictor", "📊 Model Benchmark & Metrics", "💡 Project Insights & Resume"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Select Classifier")
    chosen_model_name = st.selectbox(
        "Active Model:",
        ["Logistic Regression (Best - 85.5% Acc)", "Random Forest (82.7% Acc)"]
    )
    active_model = lr_model if "Logistic" in chosen_model_name else rf_model

    st.markdown("---")
    st.markdown("**Author:** Data Science Engineering Candidate")
    st.markdown("**Tech:** Python, scikit-learn, pandas, Streamlit")


# TAB 1: LIVE SURVIVAL PREDICTOR
if app_mode == "🔮 Live Survival Predictor":
    st.markdown('<div class="main-header">🚢 RMS Titanic: Real-Time Survival Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Input passenger credentials below to predict their survival probability using machine learning.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("👤 Demographics")
        sex = st.selectbox("Sex", ["female", "male"], index=0)
        age = st.slider("Age (Years)", min_value=1, max_value=80, value=28)
        
        # Auto-suggest Title based on sex & age
        default_title = "Miss" if sex == "female" else "Mr"
        title = st.selectbox(
            "Social Title",
            ["Mr", "Mrs", "Miss", "Master", "Rare"],
            index=2 if sex == "female" else 0,
            help="'Master' was reserved for young boys; 'Rare' includes officers and aristocracy."
        )

    with col2:
        st.subheader("🎟️ Ticket & Class")
        pclass = st.selectbox("Passenger Ticket Class", [1, 2, 3], index=0, format_func=lambda x: f"{x}st Class" if x==1 else (f"{x}nd Class" if x==2 else f"{x}rd Class"))
        fare = st.slider("Ticket Fare (£)", min_value=0.0, max_value=512.0, value=75.0, step=1.0)
        embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0, format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x])

    with col3:
        st.subheader("👨‍👩‍👧 Family Onboard")
        sibsp = st.number_input("Siblings / Spouses Aboard", min_value=0, max_value=8, value=0)
        parch = st.number_input("Parents / Children Aboard", min_value=0, max_value=6, value=0)
        
        # Derived features
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        st.info(f"**Computed Traveling Party:**\n* Total Family Size: **{family_size}**\n* Traveling Alone: **{'Yes' if is_alone == 1 else 'No'}**")

    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Prediction", use_container_width=True):
        input_data = pd.DataFrame([{
            'Age': age,
            'Fare': fare,
            'SibSp': sibsp,
            'Parch': parch,
            'FamilySize': family_size,
            'Sex': sex,
            'Pclass': pclass,
            'Embarked': embarked,
            'IsAlone': is_alone,
            'Title': title
        }])

        # Predict
        prediction = active_model.predict(input_data)[0]
        survival_prob = active_model.predict_proba(input_data)[0][1]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prediction == 1:
                st.success("### 🎉 Prediction: SURVIVED")
                st.markdown(f"The passenger is predicted to have **survived** the disaster under {chosen_model_name}.")
            else:
                st.error("### ⚠️ Prediction: PERISHED")
                st.markdown(f"The passenger is predicted to have **perished** under {chosen_model_name}.")

        with res_col2:
            st.metric(label="Survival Probability", value=f"{survival_prob * 100:.1f}%")
            st.progress(survival_prob)

        st.markdown("#### Input Features Passed to Inference Pipeline:")
        st.dataframe(input_data, use_container_width=True)


# TAB 2: MODEL BENCHMARK & METRICS
elif app_mode == "📊 Model Benchmark & Metrics":
    st.markdown('<div class="main-header">📊 Model Evaluation & Benchmarks</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Verified test metrics evaluated on a 20% holdout test set (N = 179) with zero data leakage.</div>', unsafe_allow_html=True)

    csv_path = os.path.join(models_dir, "model_comparison.csv")
    if os.path.exists(csv_path):
        comp_df = pd.read_csv(csv_path)
        st.dataframe(comp_df, use_container_width=True)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Logistic Regression Accuracy", "85.47%", "+2.79% vs RF")
    with mcol2:
        st.metric("Logistic Regression F1-Score", "0.8030", "+0.0361 vs RF")
    with mcol3:
        st.metric("Logistic Regression ROC-AUC", "0.8791", "+0.0150 vs RF")
    with mcol4:
        st.metric("5-Fold CV Accuracy (Train)", "82.31%", "± 2.04%")

    st.markdown("---")
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        cm_path = os.path.join(models_dir, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(Image.open(cm_path), caption="Confusion Matrix: Logistic Regression vs Random Forest", use_container_width=True)

    with gcol2:
        roc_path = os.path.join(models_dir, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(Image.open(roc_path), caption="Receiver Operating Characteristic (ROC) Curves", use_container_width=True)

    fi_path = os.path.join(models_dir, "rf_feature_importance.png")
    if os.path.exists(fi_path):
        st.markdown("---")
        st.subheader("🌲 Random Forest Feature Importance (MDI)")
        st.image(Image.open(fi_path), caption="Top 10 Most Influential Features", use_container_width=True)


# TAB 3: PROJECT INSIGHTS & RESUME
elif app_mode == "💡 Project Insights & Resume":
    st.markdown('<div class="main-header">💡 Project Findings & Interview Highlights</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🏆 Key Data Science Takeaways:
    1. **Data Leakage Prevention**:
       * Imputation and feature scaling are strictly isolated inside scikit-learn `Pipeline` objects. No statistics from the test set were computed during preprocessing.
    2. **Linear Separability vs Ensemble Variance**:
       * Logistic Regression outperformed Random Forest (**85.47% vs 82.68% accuracy**) because the dominant survival factors (`Sex`, `Title`, `Pclass`) were linearly separable, allowing the regularized linear model to generalize without overfitting on a small dataset (712 training rows).
    3. **Evacuation Hierarchy**:
       * `Title_Master` increased survival odds by **3.99x** ($e^{1.383}$).
       * `Title_Mr` carried a coefficient of **-1.227** (odds ratio **0.293**), indicating a **70.7% decrease in odds of survival** compared to baseline.
       * 1st Class passengers had an odds ratio of **2.721**, while 3rd Class passengers had an odds ratio of **0.382** (61.8% lower survival chance).

    ### 📋 Resume Ready Bullets:
    * *Engineered an end-to-end classification pipeline in Python using scikit-learn, pandas, and NumPy, building modular ColumnTransformer pipelines with median imputation and one-hot encoding to eliminate data leakage across 891 records.*
    * *Extracted domain-specific features (FamilySize, IsAlone, social Title) from raw passenger manifests, improving model signal and identifying socio-demographic factors that increased survival odds by up to 3.99x.*
    * *Trained and benchmarked Logistic Regression vs. Random Forest across 5-fold stratified cross-validation, achieving 85.47% test accuracy, 0.8030 F1-score, and 0.8791 ROC-AUC with serialized production-ready artifacts.*
    """)
