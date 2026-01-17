import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

DATA_PATH = "loan_approval_data.csv"
TARGET_COL = "Loan_Approved"


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    # Basic cleaning
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # Split X/y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str).str.strip()

    # Identify numeric/categorical columns
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # Preprocessing
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

    # Model
    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return clf, acc, numeric_cols, categorical_cols, X


def build_input_form(X_template: pd.DataFrame, numeric_cols, categorical_cols):
    """
    Creates Streamlit inputs for ALL fields.
    Returns a 1-row DataFrame matching training columns.
    """
    st.subheader("🧾 Enter Applicant Details (All Fields)")

    user_data = {}

    # Use columns in same order as training
    cols = X_template.columns.tolist()

    # Split into 2 columns layout for nicer UI
    left, right = st.columns(2)

    for i, col in enumerate(cols):
        container = left if i % 2 == 0 else right

        with container:
            if col in numeric_cols:
                # Use min/max from dataset to make slider sensible
                col_series = X_template[col]
                col_min = float(np.nanmin(col_series.values)) if col_series.notna().any() else 0.0
                col_max = float(np.nanmax(col_series.values)) if col_series.notna().any() else 100.0
                col_median = float(np.nanmedian(col_series.values)) if col_series.notna().any() else 0.0

                # If range is huge, use number_input instead of slider
                if (col_max - col_min) > 10000:
                    user_data[col] = st.number_input(
                        f"{col}",
                        value=float(col_median),
                        step=1.0
                    )
                else:
                    user_data[col] = st.slider(
                        f"{col}",
                        min_value=float(col_min),
                        max_value=float(col_max),
                        value=float(col_median)
                    )
            else:
                # categorical
                options = sorted(X_template[col].dropna().astype(str).unique().tolist())
                if len(options) == 0:
                    options = ["Unknown"]

                user_data[col] = st.selectbox(f"{col}", options)

    return pd.DataFrame([user_data], columns=cols)


# ----------------------------
# APP
# ----------------------------
st.title("🏦 Loan Approval Prediction App")
st.caption("This app trains a model using ALL input fields in loan_approval_data.csv")

df = load_data(DATA_PATH)

with st.expander("📄 Dataset Preview", expanded=False):
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head(20), use_container_width=True)

# Train model
clf, acc, numeric_cols, categorical_cols, X_template = train_model(df)

st.success(f"✅ Model trained successfully | Test Accuracy: **{acc:.2%}**")

# Build input form (all fields)
input_df = build_input_form(X_template, numeric_cols, categorical_cols)

st.divider()

# Predict
if st.button("🔮 Predict Loan Approval", type="primary"):
    pred = clf.predict(input_df)[0]

    # Try to show probability for "Yes" if available
    proba_text = ""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(input_df)[0]
        classes = clf.named_steps["model"].classes_
        if "Yes" in classes:
            yes_index = list(classes).index("Yes")
            proba_text = f" | Probability(Yes): **{probs[yes_index]:.2%}**"
        else:
            proba_text = f" | Probabilities: {dict(zip(classes, probs.round(3)))}"

    if str(pred).strip().lower() == "yes":
        st.success(f"🎉 Prediction: **{pred}**{proba_text}")
    else:
        st.error(f"⚠️ Prediction: **{pred}**{proba_text}")

    with st.expander("🔍 Your Input Data (1 row)"):
        st.dataframe(input_df, use_container_width=True)
