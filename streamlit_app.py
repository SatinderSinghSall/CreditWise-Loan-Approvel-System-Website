import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="CreditWise Bank Loan Approval",
    page_icon="🏦",
    layout="centered"
)

# ----------------------------
# Minimal CSS (Bank Style)
# ----------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 850px; }
.card {
    border: 1px solid rgba(150,150,150,0.25);
    border-radius: 14px;
    padding: 16px;
    background: rgba(255,255,255,0.02);
}
.small-muted { color: rgba(120,120,120,0.9); font-size: 0.9rem; }
hr { border: none; height: 1px; background: rgba(150,150,150,0.25); margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

if "Applicant_ID" in df.columns:
    df = df.drop(columns=["Applicant_ID"])

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ----------------------------
# Target normalization
# ----------------------------
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

model = RandomForestClassifier(n_estimators=250, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
acc = accuracy_score(y_test, pipeline.predict(X_test))

# ----------------------------
# Header
# ----------------------------
st.markdown("## 🏦 CreditWise Bank Loan Approval")
st.markdown("<div class='small-muted'>Fill the application form and click <b>Check Eligibility</b>.</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3)
m1.metric("Model Accuracy", f"{acc:.2f}")
m2.metric("Records", f"{len(df):,}")
m3.metric("Fields", f"{X.shape[1]}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Bank Loan Form (ALL fields)
# ----------------------------
st.markdown("### 📄 Loan Application Form")
st.markdown("<div class='card'>", unsafe_allow_html=True)

with st.form("bank_loan_form"):

    st.markdown("#### 1) Applicant Information")
    c1, c2 = st.columns(2)
    with c1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        Gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        Marital_Status = st.selectbox("Marital Status", sorted(df["Marital_Status"].unique()))
    with c2:
        Dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0, step=1)
        Education_Level = st.selectbox("Education Level", sorted(df["Education_Level"].unique()))
        Employment_Status = st.selectbox("Employment Status", sorted(df["Employment_Status"].unique()))

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("#### 2) Income & Financial Details")
    c3, c4 = st.columns(2)
    with c3:
        Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
        Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
        Savings = st.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)
    with c4:
        Credit_Score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=650.0, step=1.0)
        Existing_Loans = st.number_input("Existing Loans", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        DTI_Ratio = st.number_input("DTI Ratio (0 to 1)", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("#### 3) Loan Request")
    c5, c6 = st.columns(2)
    with c5:
        Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0, step=500.0)
        Loan_Term = st.number_input("Loan Term (months)", min_value=1.0, value=60.0, step=1.0)
        Loan_Purpose = st.selectbox("Loan Purpose", sorted(df["Loan_Purpose"].unique()))
    with c6:
        Property_Area = st.selectbox("Property Area", sorted(df["Property_Area"].unique()))
        Collateral_Value = st.number_input("Collateral Value", min_value=0.0, value=50000.0, step=500.0)
        Employer_Category = st.selectbox("Employer Category", sorted(df["Employer_Category"].unique()))

    st.markdown("<hr/>", unsafe_allow_html=True)

    submitted = st.form_submit_button("✅ Check Eligibility")

st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Prediction Result
# ----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Employment_Status": Employment_Status,
        "Age": float(Age),
        "Marital_Status": Marital_Status,
        "Dependents": float(Dependents),
        "Credit_Score": float(Credit_Score),
        "Existing_Loans": float(Existing_Loans),
        "DTI_Ratio": float(DTI_Ratio),
        "Savings": Savings,
        "Collateral_Value": Collateral_Value,
        "Loan_Amount": Loan_Amount,
        "Loan_Term": Loan_Term,
        "Loan_Purpose": Loan_Purpose,
        "Property_Area": Property_Area,
        "Education_Level": Education_Level,
        "Gender": Gender,
        "Employer_Category": Employer_Category,
    }])

    pred = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0]
    approval_prob = float(proba[1])

    st.markdown("### 🧾 Decision Summary")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    colA.metric("Approval Probability", f"{approval_prob*100:.1f}%")
    colB.metric("Decision", "Approved ✅" if pred == 1 else "Rejected ❌")

    st.progress(approval_prob)

    if pred == 1:
        st.success("Loan Approved ✅ (Eligible based on provided details)")
    else:
        st.error("Loan Rejected ❌ (Not eligible based on provided details)")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("CreditWise Bank Loan Approval • Streamlit Deployment")
