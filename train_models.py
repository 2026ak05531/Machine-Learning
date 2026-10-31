"""Train 5 classifiers on NPHA doctor-visits data and save models + metrics."""
import json
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "Number of Doctors Visited"
print("Current Working Directory:", os.getcwd())
df = pd.read_csv("NPHA-doctor-visits.csv")

# --- preprocessing -------------------------------------------------------
df = df.drop(columns=["Age"])            # constant column (all = 2), no information
df = df.replace(-1, np.nan)              # -1 = "refused" in the survey -> missing
X, y = df.drop(columns=[TARGET]), df[TARGET]
modes = X.mode().iloc[0]                 # impute with mode (all features are categorical codes)
X = X.fillna(modes)
joblib.dump({"features": list(X.columns), "modes": modes.to_dict()}, "model/preprocess.joblib")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
    "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf", probability=True, random_state=42))]),
}

rows = []
for name, mdl in models.items():
    mdl.fit(X_tr, y_tr)
    pred, proba = mdl.predict(X_te), mdl.predict_proba(X_te)
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba, multi_class="ovr", average="macro"),
        "Precision": precision_score(y_te, pred, average="macro", zero_division=0),
        "Recall": recall_score(y_te, pred, average="macro", zero_division=0),
        "F1": f1_score(y_te, pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_te, pred),
    })
    joblib.dump(mdl, f"model/{name.lower().replace(' ', '_')}.joblib")

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics.csv", index=False)
print(res.to_string(index=False))
print(y_tr.value_counts().to_dict(), y_te.value_counts().to_dict())
