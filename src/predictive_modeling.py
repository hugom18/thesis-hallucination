import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

# 1) Load feature matrix from final folder
feature_matrix_path = "data/final/feature_matrix.csv"
df = pd.read_csv(feature_matrix_path)

# 2) Preprocess
# Fill ambiguity_flag and ensure integer type
df["ambiguity_flag"] = df["ambiguity_flag"].fillna(0).astype(int)

# Numeric features: include all available numeric prompt-level features
numeric_features = [
    "bert_f1", "cos_sim", "nli_contradiction",
    "token_len", "char_len", "entity_count",
    "pronoun_ratio", "imperative_flag",
    "readability_score", "punct_density"
]
X_num = df[numeric_features]

# One-hot encode question_type and ambiguity_flag as categorical
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_features = ["question_type"]
X_cat = pd.DataFrame(
    ohe.fit_transform(df[cat_features].fillna("none")),
    columns=[f"{col}_{val}" for col in cat_features for val in ohe.categories_[0]],
    index=df.index
)

# Combine features & labels
X = pd.concat([X_num, X_cat], axis=1)
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
        results[name]["f1"].append(f1_score(y_test, preds, average='macro'))

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

# Optional: save summary
os.makedirs("result/tables", exist_ok=True)
df_summary.to_csv("result/tables/model_performance.csv", index=False)
print("Saved model performance summary to results/tables/model_performance.csv")
