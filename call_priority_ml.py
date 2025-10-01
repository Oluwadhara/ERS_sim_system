# call_priority_ml.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# -------------
# Load your synthetic data (the CSV you generated earlier)
df = pd.read_csv("synthetic_calls.csv")  # adapt path if needed

# If Timestamp exists convert to hour/day features
if "Timestamp" in df.columns:
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["hour"] = df["Timestamp"].dt.hour.fillna(0).astype(int)
    df["dayofweek"] = df["Timestamp"].dt.dayofweek.fillna(0).astype(int)
else:
    # if no timestamp, create random hours for training variety
    df["hour"] = np.random.randint(0,24,size=len(df))
    df["dayofweek"] = np.random.randint(0,7,size=len(df))

# target: Severity (ensure it's categorical)
df["Severity"] = df["Severity"].astype(str)

# basic features
X = df[["X","Y","hour","dayofweek"]].copy()

# one-hot incident types
if "IncidentType" in df.columns:
    X = pd.concat([X, pd.get_dummies(df["IncidentType"], prefix="type")], axis=1)

y = df["Severity"]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# train decision tree
clf = DecisionTreeClassifier(max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# predict & evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=clf.classes_, yticklabels=clf.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# SHAP explainability
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
# summary plot (for multiclass shap values, picks class 0..n)
plt.figure()
shap.summary_plot(shap_values, X_test, show=True)
