import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="LoanOS • Loan Approval AI", layout="wide")

DATA_PATH = "loan_approval_data.csv"
TARGET_COL = "Loan_Approved"


# ----------------------------
# UI HELPERS
# ----------------------------
def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 16px;
            background: #111827;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        ">
            <div style="font-size: 13px; opacity: 0.75;">{title}</div>
            <div style="font-size: 28px; font-weight: 700; margin-top: 6px;">{value}</div>
            <div style="font-size: 12px; opacity: 0.65; margin-top: 4px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def result_card(label, probability_text="", positive=True):
    bg = "#052e16" if positive else "#3f1d1d"
    border = "rgba(34,197,94,0.35)" if positive else "rgba(239,68,68,0.35)"
    icon = "✅" if positive else "⚠️"

    st.markdown(
        f"""
        <div style="
            padding: 18px;
            border-radius: 16px;
            background: {bg};
            border: 1px solid {border};
        ">
            <div style="font-size: 14px; opacity: 0.85;">Prediction</div>
            <div style="font-size: 26px; font-weight: 800; margin-top: 6px;">
                {icon} {label}
            </div>
            <div style="font-size: 13px; opacity: 0.8; margin-top: 6px;">
                {probability_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ----------------------------
# DATA + MODEL
# ----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str).str.strip()

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Approval rate
    approval_rate = (y == "Yes").mean() if (y == "Yes").any() else None

    return clf, acc, numeric_cols, categorical_cols, X, approval_rate


def build_input_form(X_template: pd.DataFrame, numeric_cols, categorical_cols):
    st.markdown("### 🧾 Applicant Input Form")
    st.caption("Fill in all fields. Defaults are set to typical values from your dataset.")

    user_data = {}
    cols = X_template.columns.tolist()

    # Nice UX: split into expandable sections
    with st.expander("📌 Basic Details", expanded=True):
        c1, c2 = st.columns(2)
        for col in cols[: len(cols)//3]:
            container = c1 if col in cols[: len(cols)//6] else c2
            with container:
                user_data[col] = render_field(col, X_template, numeric_cols, categorical_cols)

    with st.expander("💳 Financial & Credit", expanded=True):
        c1, c2 = st.columns(2)
        for col in cols[len(cols)//3: 2*len(cols)//3]:
            container = c1 if col in cols[len(cols)//3: len(cols)//3 + (len(cols)//6)] else c2
            with container:
                user_data[col] = render_field(col, X_template, numeric_cols, categorical_cols)

    with st.expander("🏠 Loan & Other Info", expanded=True):
        c1, c2 = st.columns(2)
        for col in cols[2*len(cols)//3:]:
            container = c1 if col in cols[2*len(cols)//3: 2*len(cols)//3 + (len(cols)//6)] else c2
            with container:
                user_data[col] = render_field(col, X_template, numeric_cols, categorical_cols)

    return pd.DataFrame([user_data], columns=cols)


def render_field(col, X_template, numeric_cols, categorical_cols):
    if col in numeric_cols:
        series = X_template[col]
        col_min = float(np.nanmin(series.values)) if series.notna().any() else 0.0
        col_max = float(np.nanmax(series.values)) if series.notna().any() else 100.0
        col_median = float(np.nanmedian(series.values)) if series.notna().any() else 0.0

        # If very wide range, number input is better
        if (col_max - col_min) > 10000:
            return st.number_input(col, value=float(col_median), step=1.0)
        else:
            return st.slider(col, min_value=float(col_min), max_value=float(col_max), value=float(col_median))
    else:
        options = sorted(X_template[col].dropna().astype(str).unique().tolist())
        if not options:
            options = ["Unknown"]
        default_index = 0
        return st.selectbox(col, options, index=default_index)


# ----------------------------
# APP LAYOUT
# ----------------------------
df = load_data(DATA_PATH)
clf, acc, numeric_cols, categorical_cols, X_template, approval_rate = train_model(df)

st.markdown(
    """
    <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div>
            <div style="font-size:34px; font-weight:800;">LoanOS</div>
            <div style="opacity:0.75; margin-top:-6px;">Loan Approval Intelligence • SaaS-style Streamlit App</div>
        </div>
        <div style="opacity:0.65; font-size:12px;">v1.0</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Sidebar navigation
st.sidebar.title("LoanOS")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict", "🗂 Data Explorer"], index=1)

st.sidebar.markdown("---")
st.sidebar.caption("Model: Logistic Regression")
st.sidebar.caption("Preprocessing: Impute + OneHot + Scale")

# Dashboard
if page == "📊 Dashboard":
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Model Accuracy", f"{acc:.2%}", "Test split performance")
    with c2:
        kpi_card("Total Records", f"{df.shape[0]}", "Rows in dataset")
    with c3:
        kpi_card("Total Features", f"{df.shape[1]-1}", "All input fields used")
    with c4:
        if approval_rate is not None:
            kpi_card("Approval Rate", f"{approval_rate:.2%}", "Share of Yes labels")
        else:
            kpi_card("Approval Rate", "N/A", "No 'Yes' class found")

    st.write("")
    st.markdown("### Quick Insights")

    left, right = st.columns([1, 1])
    with left:
        if TARGET_COL in df.columns:
            st.markdown("**Loan Approved Distribution**")
            st.bar_chart(df[TARGET_COL].value_counts(dropna=False))

    with right:
        # show top missing columns
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].head(10)
        st.markdown("**Top Missing Fields**")
        if len(missing) > 0:
            st.dataframe(missing.to_frame("Missing Count"), use_container_width=True)
        else:
            st.success("No missing values found 🎉")

# Predict
elif page == "🔮 Predict":
    st.subheader("🔮 Predict Loan Approval")
    st.caption("Uses all input fields from your dataset")

    input_df = build_input_form(X_template, numeric_cols, categorical_cols)

    st.write("")
    colA, colB = st.columns([1, 1])

    with colA:
        predict_btn = st.button("🚀 Run Prediction", type="primary")

    with colB:
        st.download_button(
            "⬇️ Download Input JSON",
            data=input_df.to_json(orient="records", indent=2),
            file_name="loan_input.json",
            mime="application/json"
        )

    if predict_btn:
        pred = clf.predict(input_df)[0]

        probability_text = ""
        positive = str(pred).strip().lower() == "yes"

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(input_df)[0]
            classes = clf.named_steps["model"].classes_
            if "Yes" in classes:
                yes_idx = list(classes).index("Yes")
                probability_text = f"Probability of Approval (Yes): {probs[yes_idx]:.2%}"
            else:
                probability_text = f"Class probabilities: {dict(zip(classes, probs.round(3)))}"

        st.write("")
        result_card(f"Loan Approved: {pred}", probability_text=probability_text, positive=positive)

        with st.expander("🔍 Submitted Input (1 row)"):
            st.dataframe(input_df, use_container_width=True)

# Data explorer
else:
    st.subheader("🗂 Data Explorer")
    st.caption("Browse the raw dataset with quick filters")

    st.dataframe(df, use_container_width=True)

    st.write("")
    st.markdown("### Column Types")
    col_types = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(col_types, use_container_width=True)
