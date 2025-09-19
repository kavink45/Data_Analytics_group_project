import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Step 1: Synthetic Churn Data (replace with your dataset) ---
np.random.seed(42)
data = {
    "CustomerID": range(1, 501),
    "Tenure": np.random.randint(1, 60, 500),
    "Purchases": np.random.randint(1, 20, 500),
    "AvgSpend": np.random.randint(5000, 50000, 500),
    "Complaints": np.random.randint(0, 5, 500),
    "Churn": np.random.choice([0, 1], size=500, p=[0.7, 0.3])
}
df = pd.DataFrame(data)

X = df[["Tenure", "Purchases", "AvgSpend", "Complaints"]]
y = df["Churn"]

# --- Step 2: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Step 4: Model Training with GridSearchCV ---
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["saga", "lbfgs"]
}
grid = GridSearchCV(LogisticRegression(max_iter=2000), param_grid, cv=5, n_jobs=-1, scoring="accuracy")
grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# --- Step 5: Collect Metrics ---
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred)

# Convert metrics to DataFrame
metrics_df = pd.DataFrame(report).transpose()
metrics_df["accuracy"] = accuracy

# Predictions DataFrame
predictions_df = pd.DataFrame({
    "CustomerID": X_test.index,
    "Tenure": X_test["Tenure"].values,
    "Purchases": X_test["Purchases"].values,
    "AvgSpend": X_test["AvgSpend"].values,
    "Complaints": X_test["Complaints"].values,
    "Actual Churn": y_test.values,
    "Predicted Churn": y_pred
})

# --- Step 6: Export to Excel ---
output_file = "Churn_Prediction_Results.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
    metrics_df.to_excel(writer, sheet_name="Metrics")
    pd.DataFrame(conf_matrix).to_excel(writer, sheet_name="Confusion_Matrix", index=False)

print(f"✅ Results exported successfully to {output_file}")
