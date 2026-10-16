
"""
Task-03: Decision Tree Classifier
Goal: Predict whether a customer will purchase a product/service
using Bank Marketing data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

# ---------- 1. Load data ----------
csv_path = "bank.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, sep=None, engine="python")
else:
    print(f"'{csv_path}' not found — generating a synthetic bank-marketing sample.")

    rng = np.random.default_rng(7)
    n = 800

    df = pd.DataFrame({
        "age": rng.integers(18, 75, n),
        "job": rng.choice([
            "admin.", "technician", "blue-collar",
            "management", "retired", "student", "unemployed"
        ], n),
        "marital": rng.choice(
            ["married", "single", "divorced"],
            n, p=[0.6, 0.28, 0.12]
        ),
        "education": rng.choice(
            ["primary", "secondary", "tertiary"],
            n, p=[0.15, 0.5, 0.35]
        ),
        "housing": rng.choice(["yes", "no"], n),
        "loan": rng.choice(["yes", "no"], n, p=[0.15, 0.85]),
        "balance": rng.normal(1300, 2500, n).round(2),
        "duration": rng.exponential(220, n).round().astype(int),
        "campaign": rng.integers(1, 8, n),
    })

    # Create target variable
    score = (
        (df["balance"] > 1000).astype(int)
        + (df["duration"] > 250).astype(int)
        + (df["education"] == "tertiary").astype(int)
        + (df["loan"] == "no").astype(int)
        - (df["campaign"] > 4).astype(int)
    )

    prob_yes = 1 / (1 + np.exp(-(score - 1.5)))

    df["y"] = (
        rng.random(n) < prob_yes
    ).map({True: "yes", False: "no"})

print(df.head())
print("\nTarget distribution:\n", df["y"].value_counts())

# ---------- 2. Preprocess ----------
target_col = "y"

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical columns
encoders = {}

for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

y_enc = LabelEncoder().fit(y)
y = y_enc.transform(y)

# ---------- 3. Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------- 4. Train Decision Tree ----------
clf = DecisionTreeClassifier(
    max_depth=4,
    criterion="entropy",
    random_state=42
)

clf.fit(X_train, y_train)

# ---------- 5. Evaluate ----------
y_pred = clf.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification report:\n",
      classification_report(
          y_test, y_pred,
          target_names=y_enc.classes_
      ))

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# ---------- 6. Visualize the tree ----------
plt.figure(figsize=(18, 8))

plot_tree(
    clf,
    feature_names=X.columns,
    class_names=y_enc.classes_,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree — Predicting Customer Purchase")

out_path = "task3_decision_tree.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")

print(f"\nSaved tree diagram to {out_path}")

# ---------- 7. Feature importance ----------
importance = pd.Series(
    clf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature importances:\n", importance)

plt.show()
