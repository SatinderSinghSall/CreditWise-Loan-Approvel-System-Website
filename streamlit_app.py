import streamlit as st
import pandas as pd

st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("loan_approval_data.csv")

df = load_data()

st.title("🏦 Loan Approval Dashboard")
st.caption("Basic Streamlit app for exploring the loan approval dataset")

# Sidebar filters
st.sidebar.header("Filters")

# Filter: Loan Approved
if "Loan_Approved" in df.columns:
    approval_options = ["All"] + sorted(df["Loan_Approved"].dropna().unique().tolist())
    selected_approval = st.sidebar.selectbox("Loan Approved", approval_options)

    if selected_approval != "All":
        df = df[df["Loan_Approved"] == selected_approval]

# Filter: Employment Status
if "Employment_Status" in df.columns:
    emp_options = ["All"] + sorted(df["Employment_Status"].dropna().unique().tolist())
    selected_emp = st.sidebar.selectbox("Employment Status", emp_options)

    if selected_emp != "All":
        df = df[df["Employment_Status"] == selected_emp]

# Filter: Property Area
if "Property_Area" in df.columns:
    prop_options = ["All"] + sorted(df["Property_Area"].dropna().unique().tolist())
    selected_prop = st.sidebar.selectbox("Property Area", prop_options)

    if selected_prop != "All":
        df = df[df["Property_Area"] == selected_prop]

# Main layout
tab1, tab2, tab3 = st.tabs(["📄 Data Preview", "📊 Summary", "📈 Charts"])

with tab1:
    st.subheader("Dataset Preview")
    st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
    st.dataframe(df, use_container_width=True)

with tab2:
    st.subheader("Summary Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Numeric Summary")
        numeric_df = df.select_dtypes(include=["number"])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with col2:
        st.markdown("### Missing Values")
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.dataframe(missing.to_frame("Missing Count"), use_container_width=True)
        else:
            st.success("No missing values 🎉")

with tab3:
    st.subheader("Charts")

    # Loan approval distribution
    if "Loan_Approved" in df.columns:
        st.markdown("### Loan Approval Distribution")
        approval_counts = df["Loan_Approved"].value_counts(dropna=False)
        st.bar_chart(approval_counts)

    # Credit Score vs Loan Approved (simple group mean)
    if "Credit_Score" in df.columns and "Loan_Approved" in df.columns:
        st.markdown("### Avg Credit Score by Loan Approval")
        grouped = df.groupby("Loan_Approved")["Credit_Score"].mean().sort_values(ascending=False)
        st.bar_chart(grouped)

    # Loan Amount distribution
    if "Loan_Amount" in df.columns:
        st.markdown("### Loan Amount Histogram")
        st.bar_chart(df["Loan_Amount"].dropna().value_counts().head(30))
