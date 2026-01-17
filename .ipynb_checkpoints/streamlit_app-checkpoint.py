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
    page_title="CreditWise | Loan Approval",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Theme Toggle (Light/Dark)
# ----------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state.theme_mode = st.radio(
        "Theme Mode",
        ["Dark", "Light"],
        index=0 if st.session_state.theme_mode == "Dark" else 1,
        horizontal=True
    )
    st.markdown("---")

theme = st.session_state.theme_mode

# ----------------------------
# Theme Tokens (SaaS-ready)
# ----------------------------
if theme == "Dark":
    BG = "#0b1220"
    CARD = "rgba(255,255,255,0.04)"
    BORDER = "rgba(255,255,255,0.10)"
    TEXT = "#e5e7eb"
    MUTED = "rgba(229,231,235,0.70)"
    ACCENT = "#60a5fa"
    ACCENT2 = "#a78bfa"
    SUCCESS_BG = "rgba(34,197,94,0.15)"
    ERROR_BG = "rgba(239,68,68,0.15)"
else:
    BG = "#f7f8fb"
    CARD = "rgba(255,255,255,1)"
    BORDER = "rgba(15,23,42,0.10)"
    TEXT = "#0f172a"
    MUTED = "rgba(15,23,42,0.65)"
    ACCENT = "#2563eb"
    ACCENT2 = "#7c3aed"
    SUCCESS_BG = "rgba(34,197,94,0.12)"
    ERROR_BG = "rgba(239,68,68,0.10)"

# ----------------------------
# CSS (Modern SaaS)
# ----------------------------
st.markdown(f"""
<style>
/* Global */
html, body, [class*="css"] {{
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
}}
.stApp {{
    background: {BG};
    color: {TEXT};
}}
.block-container {{
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem;
}}

/* Cards */
.card {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 18px;
}}
.card-tight {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 14px;
}}
hr {{
    border: none;
    height: 1px;
    background: {BORDER};
    margin: 14px 0;
}}

/* Title + subtitle */
.h-title {{
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}}
.h-sub {{
    color: {MUTED};
    font-size: 1rem;
}}

/* Badge / pill */
.pill {{
    display: inline-flex;
    gap: 8px;
    align-items: center;
    padding: 6px 12px;
    border-radius: 999px;
    border: 1px solid {BORDER};
    background: {CARD};
    font-size: 0.9rem;
    color: {TEXT};
}}
.pill-dot {{
    width: 9px;
    height: 9px;
    border-radius: 999px;
    background: {ACCENT};
}}

/* Buttons */
.stButton > button {{
    border-radius: 14px !important;
    padding: 0.7rem 1rem !important;
    border: 1px solid {BORDER} !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
}}

/* Metrics */
div[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 14px;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid {BORDER};
}}

/* Success / Error boxes */
.success-box {{
    background: {SUCCESS_BG};
    border: 1px solid rgba(34,197,94,0.35);
    padding: 14px;
    border-radius: 16px;
}}
.error-box {{
    background: {ERROR_BG};
    border: 1px solid rgba(239,68,68,0.35);
    padding: 14px;
    border-radius: 16px;
}}
.muted {{
    color: {MUTED};
}}
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

# Handle NaN
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"].fillna(df["Loan_Approved"].mode()[0])

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
# Header (Top SaaS style)
# ----------------------------
topL, topR = st.columns([3, 2], vertical_alignment="center")

with topL:
    st.markdown(f"""
    <div class="h-title">💳 CreditWise Loan Approval</div>
    <div class="h-sub">SaaS-ready loan approval prediction dashboard • {theme} mode</div>
    """, unsafe_allow_html=True)

with topR:
    st.markdown('<div class="card-tight">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Features", f"{X.shape[1]}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Sidebar Form (clean + grouped)
# ----------------------------
st.sidebar.markdown("## 🧾 Applicant Details")
st.sidebar.markdown('<div class="muted">Fill details and click Predict.</div>', unsafe_allow_html=True)
st.sidebar.markdown("")

if "reset_now" not in st.session_state:
    st.session_state.reset_now = False

with st.sidebar.form("loan_form"):
    st.markdown("### Income & Loan")
    Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=5000.0, step=100.0)
    Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0, step=500.0)
    Loan_Term = st.number_input("Loan Term (months)", min_value=1.0, value=60.0, step=1.0)

    st.markdown("### Credit & Risk")
    Credit_Score = st.slider("Credit Score", min_value=300, max_value=900, value=650, step=1)
    Existing_Loans = st.slider("Existing Loans", min_value=0, max_value=10, value=0, step=1)
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

    col_btn1, col_btn2 = st.columns(2)
    submitted = col_btn1.form_submit_button("🚀 Predict")
    reset = col_btn2.form_submit_button("♻️ Reset")

if reset:
    st.rerun()

# ----------------------------
# Tabs (SaaS structure)
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "📈 Insights", "🗂 Dataset"])

# ----------------------------
# Prediction Tab
# ----------------------------
with tab1:
    colA, colB = st.columns([2.2, 1.2], gap="large")

    with colA:
        st.markdown("### Result")
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if not submitted:
            st.markdown("""
            <div class="pill"><span class="pill-dot"></span>
            Ready to predict — use the sidebar form.</div>
            """, unsafe_allow_html=True)
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.info("👈 Fill the form and click **Predict** to see decision + probability.")
        else:
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

            approve_prob = float(proba[1]) if len(proba) > 1 else 0.0
            reject_prob = float(proba[0]) if len(proba) > 0 else 0.0

            m1, m2, m3 = st.columns(3)
            m1.metric("Approval Probability", f"{approve_prob*100:.1f}%")
            m2.metric("Rejection Probability", f"{reject_prob*100:.1f}%")
            m3.metric("Credit Score", f"{Credit_Score:.0f}")

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.progress(approve_prob)

            if pred == 1:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown("### ✅ Approved")
                st.markdown("**Next step:** Offer best interest rate options and proceed to verification.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.markdown("### ❌ Rejected")
                st.markdown("**Suggestion:** Improve credit score, reduce DTI, or increase collateral.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("### Applicant Snapshot")
        st.markdown('<div class="card">', unsafe_allow_html=True)

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
        st.dataframe(
            pd.DataFrame(snapshot.items(), columns=["Field", "Value"]),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Insights Tab
# ----------------------------
with tab2:
    st.markdown("### Insights (SaaS Analytics)")
    st.markdown('<div class="card">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved %", f"{(df['Loan_Approved'].mean()*100):.1f}%")
    c2.metric("Avg Credit Score", f"{df['Credit_Score'].mean():.0f}")
    c3.metric("Avg Loan Amount", f"{df['Loan_Amount'].mean():.0f}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("#### 📌 Quick Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.bar_chart(df["Loan_Approved"].value_counts())

    with chart_col2:
        st.line_chart(df[["Credit_Score", "Loan_Amount"]].head(50))

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Dataset Tab
# ----------------------------
with tab3:
    st.markdown("### Dataset Preview")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("© CreditWise • SaaS-ready UI • Streamlit Cloud Deployment")
