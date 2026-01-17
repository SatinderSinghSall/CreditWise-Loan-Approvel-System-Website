import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# Page Config + Minimal CSS
# ----------------------------
st.set_page_config(
    page_title="CreditWise | Loan Approval",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
/* Make the app feel more SaaS */
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
border-radius: 16px; padding: 14px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(255,255,255,0.08);}
.small-note {opacity: 0.7; font-size: 0.9rem;}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 18px;
}
hr {border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0;}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load dataset
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

# Drop Applicant_ID
if "Applicant_ID" in df.columns:
    df = df.drop(columns=["Applicant_ID"])

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Features & Target
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"].fillna(df["Loan_Approved"].mode()[0])

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# Model
model = RandomForestClassifier(n_estimators=250, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
pipeline.fit(X_train, y_train)
acc = accuracy_score(y_test, pipeline.predict(X_test))

# ----------------------------
# Header
# ----------------------------
left, right = st.columns([3, 2], vertical_alignment="center")

with left:
    st.markdown("## 💳 CreditWise Loan Approval System")
    st.markdown(
        "<div class='small-note'>Modern SaaS-style loan approval prediction dashboard.</div>",
        unsafe_allow_html=True
    )

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Model Health")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Features", f"{X.shape[1]}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Sidebar Form (SaaS vibe)
# ----------------------------
st.sidebar.markdown("## 🧾 Applicant Details")
st.sidebar.markdown("<div class='small-note'>Fill details and click Predict.</div>", unsafe_allow_html=True)
st.sidebar.markdown("")

with st.sidebar.form("loan_form"):
    st.markdown("### Income & Loan")
    Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
    Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0, step=500.0)
    Loan_Term = st.number_input("Loan Term (months)", min_value=1.0, value=60.0, step=1.0)

    st.markdown("### Credit & Risk")
    Credit_Score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=650.0, step=1.0)
    Existing_Loans = st.number_input("Existing Loans", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    DTI_Ratio = st.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01)

    st.markdown("### Savings & Collateral")
    Savings = st.number_input("Savings", min_value=0.0, value=10000.0, step=500.0)
    Collateral_Value = st.number_input("Collateral Value", min_value=0.0, value=50000.0, step=500.0)

    st.markdown("### Personal & Background")
    Age = st.slider("Age", min_value=18, max_value=100, value=30, step=1)
    Dependents = st.slider("Dependents", min_value=0, max_value=10, value=0, step=1)

    Employment_Status = st.selectbox("Employment Status", sorted(df["Employment_Status"].unique()))
    Marital_Status = st.selectbox("Marital Status", sorted(df["Marital_Status"].unique()))
    Education_Level = st.selectbox("Education Level", sorted(df["Education_Level"].unique()))
    Gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
    Employer_Category = st.selectbox("Employer Category", sorted(df["Employer_Category"].unique()))

    st.markdown("### Property & Purpose")
    Property_Area = st.selectbox("Property Area", sorted(df["Property_Area"].unique()))
    Loan_Purpose = st.selectbox("Loan Purpose", sorted(df["Loan_Purpose"].unique()))

    submitted = st.form_submit_button("🚀 Predict Loan Decision")

# ----------------------------
# Main Content
# ----------------------------
colA, colB = st.columns([2.2, 1.2], gap="large")

with colA:
    st.markdown("### 🧠 Prediction Result")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if not submitted:
        st.info("👈 Fill the form in the sidebar and click **Predict Loan Decision**.")
    else:
        input_data = pd.DataFrame([{
            "Applicant_Income": Applicant_Income,
            "Coapplicant_Income": Coapplicant_Income,
            "Employment_Status": Employment_Status,
            "Age": float(Age),
            "Marital_Status": Marital_Status,
            "Dependents": float(Dependents),
            "Credit_Score": Credit_Score,
            "Existing_Loans": Existing_Loans,
            "DTI_Ratio": DTI_Ratio,
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

        approve_prob = float(proba[1]) if len(proba) > 1 else 0.0
        reject_prob = float(proba[0]) if len(proba) > 0 else 0.0

        top1, top2, top3 = st.columns(3)
        top1.metric("Approval Probability", f"{approve_prob*100:.1f}%")
        top2.metric("Rejection Probability", f"{reject_prob*100:.1f}%")
        top3.metric("Credit Score", f"{Credit_Score:.0f}")

        st.markdown("<hr/>", unsafe_allow_html=True)

        if pred == 1:
            st.success("✅ Loan Approved")
            st.markdown("**Recommendation:** Applicant qualifies based on current profile.")
        else:
            st.error("❌ Loan Rejected")
            st.markdown("**Recommendation:** Improve credit score / reduce DTI / increase collateral.")

    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown("### 📌 Applicant Snapshot")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    snapshot = {
        "Applicant Income": Applicant_Income if submitted else "-",
        "Loan Amount": Loan_Amount if submitted else "-",
        "Loan Term (months)": Loan_Term if submitted else "-",
        "DTI Ratio": DTI_Ratio if submitted else "-",
        "Savings": Savings if submitted else "-",
        "Collateral": Collateral_Value if submitted else "-",
        "Employment": Employment_Status if submitted else "-",
        "Property Area": Property_Area if submitted else "-",
    }

    st.dataframe(pd.DataFrame(snapshot.items(), columns=["Field", "Value"]), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("© CreditWise • Built with Streamlit • Modern SaaS UI")
