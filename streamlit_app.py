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
# Theme Toggle
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
# Theme Tokens
# ----------------------------
if theme == "Dark":
    BG = "#0b1220"
    CARD = "rgba(255,255,255,0.04)"
    BORDER = "rgba(255,255,255,0.10)"
    TEXT = "#e5e7eb"
    MUTED = "rgba(229,231,235,0.70)"
    SUCCESS_BG = "rgba(34,197,94,0.15)"
    ERROR_BG = "rgba(239,68,68,0.15)"
    ACCENT = "#60a5fa"
else:
    BG = "#f7f8fb"
    CARD = "#ffffff"
    BORDER = "rgba(15,23,42,0.12)"
    TEXT = "#0f172a"
    MUTED = "rgba(15,23,42,0.65)"
    SUCCESS_BG = "rgba(34,197,94,0.12)"
    ERROR_BG = "rgba(239,68,68,0.10)"
    ACCENT = "#2563eb"

# ----------------------------
# CSS (Premium SaaS)
# ----------------------------
st.markdown(f"""
<style>
/* App background */
.stApp {{
    background: {BG};
    color: {TEXT};
}}

/* Layout padding */
.block-container {{
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}}

/* Sidebar width + border */
section[data-testid="stSidebar"] {{
    width: 340px !important;
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

/* Muted text */
.muted {{
    color: {MUTED};
}}

/* Divider */
hr {{
    border: none;
    height: 1px;
    background: {BORDER};
    margin: 14px 0;
}}

/* Success / Error cards */
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

/* Metric cards */
div[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 14px;
}}

/* Buttons */
.stButton > button {{
    border-radius: 14px !important;
    padding: 0.7rem 1rem !important;
    border: 1px solid {BORDER} !important;
    font-weight: 600 !important;
}}
.stButton > button:hover {{
    border-color: {ACCENT} !important;
}}

/* Tabs look like SaaS nav pills */
button[data-baseweb="tab"] {{
    border-radius: 999px !important;
    padding: 10px 16px !important;
    border: 1px solid {BORDER} !important;
    background: transparent !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    border: 1px solid rgba(96,165,250,0.55) !important;
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

# Drop Applicant_ID
if "Applicant_ID" in df.columns:
    df = df.drop(columns=["Applicant_ID"])

# Handle NaN
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ----------------------------
# Target Normalization (No crash)
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

# Features and Target
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
topL, topR = st.columns([3, 2], vertical_alignment="center")

with topL:
    st.markdown("## 💳 CreditWise Loan Approval")
    st.markdown(f"<div class='muted'>SaaS-ready prediction dashboard • {theme} mode</div>", unsafe_allow_html=True)

with topR:
    st.markdown("<div class='card-tight'>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Accuracy", f"{acc:.2f}")
    b.metric("Rows", f"{len(df)/1000:.1f}K" if len(df) >= 1000 else f"{len(df):,}")
    c.metric("Features", f"{X.shape[1]}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("")

# ----------------------------
# Hero Card (Premium SaaS feel)
# ----------------------------
st.markdown(f"""
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
    <div>
      <div style="font-size:1.25rem; font-weight:800;">🚀 Instant Loan Decision Engine</div>
      <div class="muted" style="margin-top:6px;">
        Fill applicant details → get approval decision + probability in seconds.
      </div>
    </div>
    <div style="text-align:right;">
      <div class="muted" style="font-size:0.85rem;">System Status</div>
      <div style="font-weight:800;">🟢 Online</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------------------
# Sidebar Form
# ----------------------------
st.sidebar.markdown("## 🧾 Applicant Details")
st.sidebar.markdown("<div class='muted'>Fill details and click Predict.</div>", unsafe_allow_html=True)

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

    col1, col2 = st.columns(2)
    submitted = col1.form_submit_button("🚀 Predict")
    reset = col2.form_submit_button("♻️ Reset")

if reset:
    st.rerun()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🧠 Prediction", "📈 Insights", "🗂 Dataset"])

# ----------------------------
# Prediction Tab
# ----------------------------
with tab1:
    colA, colB = st.columns([2.2, 1.2], gap="large")

    with colA:
        st.markdown("### Result")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if not submitted:
            st.info("👈 Fill the sidebar form and click **Predict**.")
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
            approve_prob = float(proba[1])

            m1, m2, m3 = st.columns(3)
            m1.metric("Approval Probability", f"{approve_prob*100:.1f}%")
            m2.metric("DTI Ratio", f"{DTI_Ratio:.2f}")
            m3.metric("Credit Score", f"{Credit_Score:.0f}")

            st.progress(approve_prob)

            if pred == 1:
                st.markdown("<div class='success-box'>", unsafe_allow_html=True)
                st.markdown("### ✅ Approved")
                st.markdown("**Next step:** Verification + offer best interest rate plan.")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-box'>", unsafe_allow_html=True)
                st.markdown("### ❌ Rejected")
                st.markdown("**Suggestion:** Improve credit score / reduce DTI / increase collateral.")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("### Applicant Snapshot")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        snapshot = {
            "Applicant Income": Applicant_Income if submitted else "-",
            "Coapplicant Income": Coapplicant_Income if submitted else "-",
            "Loan Amount": Loan_Amount if submitted else "-",
            "Loan Term": Loan_Term if submitted else "-",
            "DTI Ratio": DTI_Ratio if submitted else "-",
            "Savings": Savings if submitted else "-",
            "Collateral": Collateral_Value if submitted else "-",
            "Employment": Employment_Status if submitted else "-",
            "Property Area": Property_Area if submitted else "-",
        }

        # Premium snapshot list (instead of dataframe)
        for k, v in snapshot.items():
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid {BORDER};">
                    <div class="muted">{k}</div>
                    <div style="font-weight:700;">{v}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Insights Tab
# ----------------------------
with tab2:
    st.markdown("### Insights")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    approved_rate = df["Loan_Approved_num"].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Approved %", f"{approved_rate:.1f}%")
    c2.metric("Avg Credit Score", f"{df['Credit_Score'].mean():.0f}")
    c3.metric("Avg Loan Amount", f"{df['Loan_Amount'].mean():.0f}")

    st.markdown("<hr/>", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Approval Distribution")
        st.bar_chart(df["Loan_Approved_num"].value_counts())

    with right:
        st.markdown("#### Credit Score vs Loan Amount (sample)")
        sample = df[["Credit_Score", "Loan_Amount"]].head(80)
        st.line_chart(sample)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Dataset Tab
# ----------------------------
with tab3:
    st.markdown("### Dataset Preview")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.caption("© CreditWise • SaaS-ready UI • Light/Dark Theme • Streamlit Cloud")
