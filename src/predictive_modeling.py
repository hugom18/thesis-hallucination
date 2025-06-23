# src/predictive_modeling.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 1) Load feature matrix
df = pd.read_csv("data/feature_matrix.csv")

# 2) Preprocess
df["ambiguity_flag"] = df["ambiguity_flag"].fillna(0)
X_num = df[["bert_f1", "cos_sim", "nli_contradiction", "token_len", "entity_count"]]

# One-hot encode question_type
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
q_enc = ohe.fit_transform(df[["question_type"]].fillna("none"))
q_cols = [name.replace("question_type", "qtype") for name in ohe.get_feature_names_out()]
X_q = pd.DataFrame(q_enc, columns=q_cols, index=df.index)

# Combine features & labels
X = pd.concat([X_num, X_q], axis=1)
y = df["hallucination"].astype(int)

# 3) Define classifiers
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# 4) Stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {"accuracy": [], "f1": []} for name in models}

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name]["accuracy"].append(accuracy_score(y_test, preds))
        results[name]["f1"].append(f1_score(y_test, preds))

# 5) Summarize and print
summary = []
for name, scores in results.items():
    summary.append({
        "Model": name,
        "Accuracy": round(np.mean(scores["accuracy"]), 3),
        "Macro-F1": round(np.mean(scores["f1"]), 3),
    })
df_summary = pd.DataFrame(summary)

print("\nCross-Validated Model Performance:")
print(df_summary.to_string(index=False))
