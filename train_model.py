import pandas as pd
import numpy as np
import json

# Load data
data = pd.read_csv('students.csv')
print("Dataset preview:")
print(data.head())

# Features (inputs) and target (what we're predicting)
X = data[['study_hours', 'attendance', 'previous_score']].values
y = data['final_score'].values

# Add a column of 1s for the intercept term
X_with_bias = np.column_stack([np.ones(len(X)), X])

# Split into training (80%) and testing (20%) sets
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(len(X) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X_with_bias[train_idx], X_with_bias[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train using the Normal Equation (closed-form solution for Linear Regression)
# This solves: weights = (X^T X)^-1 X^T y
weights = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

# Evaluate on test data
predictions = X_test @ weights
mae = np.mean(np.abs(predictions - y_test))
ss_res = np.sum((y_test - predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"\nModel trained!")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model weights as JSON (simple, no extra libraries needed)
model = {
    "intercept": float(weights[0]),
    "study_hours_coef": float(weights[1]),
    "attendance_coef": float(weights[2]),
    "previous_score_coef": float(weights[3]),
}

with open('model.json', 'w') as f:
    json.dump(model, f, indent=2)

print("\nModel saved as model.json")
print(model)
