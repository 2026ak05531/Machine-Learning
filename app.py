import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report)

TARGET = "Number of Doctors Visited"
MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}

st.title("NPHA Doctor Visits – Classification Models")
st.write("Predict the number of doctor visits category (1 = 0-1, 2 = 2-3, 3 = 4+) for older adults.")

choice = st.selectbox("Select model", list(MODELS))
file = st.file_uploader("Upload test CSV (same columns as test_data.csv)", type="csv")

if file:
    df = pd.read_csv(file)
    prep = joblib.load("model/preprocess.joblib")
    model = joblib.load(MODELS[choice])
    X = df[prep["features"]].fillna(prep["modes"])
    pred = model.predict(X)
    st.subheader("Predictions (first 20 rows)")
    st.dataframe(pd.DataFrame({"Predicted": pred}).head(20))

    if TARGET in df.columns:
        y, proba = df[TARGET], model.predict_proba(X)
        c = st.columns(3)
        c[0].metric("Accuracy", f"{accuracy_score(y, pred):.4f}")
        c[1].metric("AUC", f"{roc_auc_score(y, proba, multi_class='ovr', average='macro'):.4f}")
        c[2].metric("MCC", f"{matthews_corrcoef(y, pred):.4f}")
        c = st.columns(3)
        c[0].metric("Precision", f"{precision_score(y, pred, average='macro', zero_division=0):.4f}")
        c[1].metric("Recall", f"{recall_score(y, pred, average='macro', zero_division=0):.4f}")
        c[2].metric("F1", f"{f1_score(y, pred, average='macro', zero_division=0):.4f}")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.text(classification_report(y, pred, zero_division=0))
