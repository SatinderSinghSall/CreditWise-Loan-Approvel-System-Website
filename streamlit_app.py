import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="CreditWise Loan System", layout="centered")

st.title("💳 CreditWise Loan Approval System")
st.write("Predict whether a loan will be approved based on applicant details.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

# Drop Applicant_ID
if "Applicant_ID" in df.columns:
    df = df.drop(columns=["Applicant_ID"])

# Features & Target
X = df.drop(columns=["Loan_Approved"])
y = df["Loan_Approved"]

# Identify categorical & numeric columns
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
model = RandomForestClassifier(n_estimators=200, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.info(f"✅ Model trained successfully | Accuracy: {acc:.2f}")

st.subheader("📌 Enter Applicant Details")

# Input form
with st.form("loan_form"):
    Applicant_Income = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    Employment_Status = st.selectbox("Employment Status", sorted(df["Employment_Status"].unique()))
    Age = st.number_input("Age", min_value=18.0, max_value=100.0, value=30.0)
    Marital_Status = st.selectbox("Marital Status", sorted(df["Marital_Status"].unique()))
    Dependents = st.number_input("Dependents", min_value=0.0, max_value=10.0, value=0.0)
    Credit_Score = st.number_input("Credit Score", min_value=300.0, max_value=900.0, value=650.0)
    Existing_Loans = st.number_input("Existing Loans", min_value=0.0, max_value=10.0, value=0.0)
    DTI_Ratio = st.number_input("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3)
    Savings = st.number_input("Savings", min_value=0.0, value=10000.0)
    Collateral_Value = st.number_input("Collateral Value", min_value=0.0, value=50000.0)
    Loan_Amount = st.number_input("Loan Amount", min_value=0.0, value=20000.0)
    Loan_Term = st.number_input("Loan Term (months)", min_value=1.0, value=60.0)
    Loan_Purpose = st.selectbox("Loan Purpose", sorted(df["Loan_Purpose"].unique()))
    Property_Area = st.selectbox("Property Area", sorted(df["Property_Area"].unique()))
    Education_Level = st.selectbox("Education Level", sorted(df["Education_Level"].unique()))
    Gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
    Employer_Category = st.selectbox("Employer Category", sorted(df["Employer_Category"].unique()))

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = pd.DataFrame([{
        "Applicant_Income": Applicant_Income,
        "Coapplicant_Income": Coapplicant_Income,
        "Employment_Status": Employment_Status,
        "Age": Age,
        "Marital_Status": Marital_Status,
        "Dependents": Dependents,
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

    prediction = pipeline.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
