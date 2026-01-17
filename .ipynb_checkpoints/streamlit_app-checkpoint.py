import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="CreditWise - Bank Loan UI", page_icon="💳", layout="centered")

# --- Minimal styling for a clean bank-like card ---
st.markdown(
    """
    <style>
    .card {
      background: white;
      color: #091A2B;
      padding: 18px;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(10,25,47,0.08);
    }
    .small-muted { color: #6b7280; font-size: 0.9rem; }
    .result-card { padding: 16px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💳 CreditWise — Bank Loan Decision")
st.write("Simple form — enter applicant details and get an instant decision and probability.")

# -----------------------
# Data & model utils
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

@st.cache_data
def train_pipeline(df):
    # drop id if present
    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])

    # basic NaN handling
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # normalize target to numeric 0/1
    def normalize_target(series):
        s = series.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1, "yes": 1, "y": 1, "approved": 1, "true": 1,
            "0": 0, "no": 0, "n": 0, "rejected": 0, "false": 0
        }
        return s.map(mapping)

    df["Loan_Approved_num"] = normalize_target(df["Loan_Approved"])
    df["Loan_Approved_num"] = df["Loan_Approved_num"].fillna(0).astype(int)

    X = df.drop(columns=["Loan_Approved", "Loan_Approved_num"])
    y = df["Loan_Approved_num"]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = Pipeline([("preproc", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))

    return pipeline, df, acc

# Load & train (cached)
df = load_data()
pipeline, df_clean, model_acc = train_pipeline(df)

# -----------------------
# Simple centered form
# -----------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("Applicant Information")
st.markdown('<div class="small-muted">Fill the form below. All fields are optional — defaults are sensible.</div>', unsafe_allow_html=True)

# Layout two columns for compactness
c1, c2 = st.columns(2)

with c1:
    Applicant_Income = st.number_input("Applicant Income (₹)", min_value=0.0, value=5000.0, step=500.0)
    Coapplicant_Income = st.number_input("Coapplicant Income (₹)", min_value=0.0, value=0.0, step=500.0)
    Loan_Amount = st.number_input("Loan Amount (₹)", min_value=0.0, value=20000.0, step=500.0)
    Loan_Term = st.number_input("Loan Term (months)", min_value=1, value=60, step=1)
    Loan_Purpose = st.selectbox("Loan Purpose", sorted(df_clean["Loan_Purpose"].unique()))

with c2:
    Credit_Score = st.number_input("Credit Score", min_value=300, max_value=900, value=650, step=1)
    Existing_Loans = st.number_input("Existing Loans (count)", min_value=0, value=0, step=1)
    DTI_Ratio = st.slider("DTI Ratio (0 - 1)", 0.0, 1.0, 0.30, step=0.01)
    Collateral_Value = st.number_input("Collateral Value (₹)", min_value=0.0, value=50000.0, step=500.0)
    Property_Area = st.selectbox("Property Area", sorted(df_clean["Property_Area"].unique()))

st.markdown("---")

# Second row: personal & employment
st.subheader("Personal & Employment")
r1, r2 = st.columns(2)

with r1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    Dependents = st.number_input("Dependents", min_value=0, value=0, step=1)
    Marital_Status = st.selectbox("Marital Status", sorted(df_clean["Marital_Status"].unique()))

with r2:
    Employment_Status = st.selectbox("Employment Status", sorted(df_clean["Employment_Status"].unique()))
    Education_Level = st.selectbox("Education Level", sorted(df_clean["Education_Level"].unique()))
    Gender = st.selectbox("Gender", sorted(df_clean["Gender"].unique()))
    Employer_Category = st.selectbox("Employer Category", sorted(df_clean["Employer_Category"].unique()))

# Buttons
colA, colB = st.columns([1, 1])
with colA:
    submit = st.button("Predict Decision", type="primary")
with colB:
    reset = st.button("Reset Form")

if reset:
    st.experimental_rerun()

# Show a small model status line
st.markdown(f"<div class='small-muted'>Model accuracy (approx): {model_acc:.2f} — trained on uploaded dataset.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Prediction result area
# -----------------------
st.markdown("<br>")
st.markdown("<div class='card result-card'>", unsafe_allow_html=True)

if not submit:
    st.info("No prediction yet — fill the fields and click **Predict Decision**.")
else:
    # assemble input row exactly matching pipeline's expected columns
    input_df = pd.DataFrame([{
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Employment_Status": Employment_Status,
        "Age": Age,
        "Marital_Status": Marital_Status,
        "Dependents": Dependents,
        "Credit_Score": Credit_Score,
        "Existing_Loans": Existing_Loans,
        "DTI_Ratio": DTI_Ratio,
        "Savings": 0 if "Savings" not in df_clean.columns else df_clean["Savings"].median(),
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,
        "Loan_Purpose": Loan_Purpose,
        "Property_Area": Property_Area,
        "Education_Level": Education_Level,
        "Gender": Gender,
        "Employer_Category": Employer_Category,
    }])

    # Predict
    try:
        pred = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0]
        approve_prob = float(proba[1]) if len(proba) > 1 else 0.0
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Nicely formatted result
        if pred == 1:
            st.success(f"✅ Loan likely APPROVED — probability {approve_prob*100:.1f}%")
            st.write("Next steps: verification of documents and rate offer.")
        else:
            st.error(f"❌ Loan likely REJECTED — probability of approval {approve_prob*100:.1f}%")
            st.write("Suggestions: improve credit score, reduce DTI, or add collateral.")

        # Compact applicant snapshot
        st.markdown("**Applicant snapshot**")
        snap = {
            "Income": f"₹{Applicant_Income:,.0f}",
            "Loan amount": f"₹{Loan_Amount:,.0f}",
            "Credit score": f"{Credit_Score}",
            "DTI": f"{DTI_Ratio:.2f}"
        }
        for k, v in snap.items():
            st.markdown(f"- **{k}**: {v}")

        st.markdown("</div>", unsafe_allow_html=True)

# small footer
st.markdown("<br>")
st.caption("CreditWise — Simple Bank-style Loan Decision UI")
