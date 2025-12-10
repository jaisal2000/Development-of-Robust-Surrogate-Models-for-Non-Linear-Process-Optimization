

# Cell 1: Setup & Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# For scaling/encoding
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# For ignoring scientific notation if needed
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

# Cell 2: Load dataset
data_path = "distillation_sim_results.csv"  # update if path differs
df = pd.read_csv(data_path)

# Quick check
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()

# Cell 3: Clean and check unit consistency

# 1. Check for missing values
print("Missing values:\n", df.isnull().sum())

# 2. Remove any rows with NaNs
df_clean = df.dropna().copy()

# 3. Check ranges for physical realism
# R should be 0.8 - 5.0
df_clean = df_clean[(df_clean["R"] >= 0.8) & (df_clean["R"] <= 5.0)]

# xF_Ethanol between 0.2 and 0.95
df_clean = df_clean[(df_clean["xF_Ethanol"] >= 0.2) & (df_clean["xF_Ethanol"] <= 0.95)]

# xD_Ethanol must be between 0 and 1
df_clean = df_clean[(df_clean["xD_Ethanol"] > 0) & (df_clean["xD_Ethanol"] <= 1)]

# Reset index
df_clean.reset_index(drop=True, inplace=True)

print("Cleaned shape:", df_clean.shape)
df_clean.describe()

# Cell 4: Split into train/validation/test by operating space

# Define blocks based on R
test_block = df_clean[df_clean["R"] >= 4.2]   # held-out region
train_val_block = df_clean[df_clean["R"] < 4.2]

# Split train/validation (80/20)
train_block, val_block = train_test_split(train_val_block, test_size=0.2, random_state=42)

print("Train size:", train_block.shape)
print("Validation size:", val_block.shape)
print("Test size:", test_block.shape)

# Cell 5: Preprocessing setup

# Separate inputs/outputs
X_train = train_block[["R", "B", "xF_Ethanol", "F", "N"]]
y_train = train_block[["xD_Ethanol", "QR_kW"]]

X_val = val_block[["R", "B", "xF_Ethanol", "F", "N"]]
y_val = val_block[["xD_Ethanol", "QR_kW"]]

X_test = test_block[["R", "B", "xF_Ethanol", "F", "N"]]
y_test = test_block[["xD_Ethanol", "QR_kW"]]

# Define preprocessing: scale continuous + one-hot encode N
numeric_features = ["R", "B", "xF_Ethanol", "F"]
categorical_features = ["N"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, drop="if_binary"), categorical_features)
    ]
)

# Fit only on train, then transform all
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)
X_test_prep = preprocessor.transform(X_test)

print("Transformed shapes:")
print("Train:", X_train_prep.shape)
print("Validation:", X_val_prep.shape)
print("Test:", X_test_prep.shape)

df_clean

df_clean.to_csv("cleaned_data.csv",index=False)

# Cell M1: Modeling — imports, data loading, preprocessing & splits
# Run this cell first. When ready, say "next" and I'll give the next modeling cell.

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 1) Load data (use df_clean if present in the notebook; otherwise read the CSV)
try:
    df_clean  # if this exists from earlier cells, we'll use it
except NameError:
    csv_path = "mock_distillation_dataset_320.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found in working directory. Put your CSV in the notebook folder or update the path.")
    df = pd.read_csv(csv_path)
    # basic cleaning (same checks used previously)
    df = df.dropna().copy()
    df = df[(df["R"] >= 0.8) & (df["R"] <= 5.0)]
    df = df[(df["xF_Ethanol"] >= 0.2) & (df["xF_Ethanol"] <= 0.95)]
    df = df[(df["xD_Ethanol"] > 0) & (df["xD_Ethanol"] <= 1)]
    df.reset_index(drop=True, inplace=True)
else:
    df = df_clean.copy()

# 2) Features and targets
X = df[["R", "B", "xF_Ethanol", "F", "N"]].copy()
y = df[["xD_Ethanol", "QR_kW"]].copy()

# 3) Create train/validation/test splits by operating block (hold out high-R region for extrapolation)
test_mask = (df["R"] >= 4.2)  # change threshold if you prefer a different holdout region

if test_mask.sum() == 0:
    # fallback if no rows meet the block criterion: use random test split (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
else:
    X_train_val = X.loc[~test_mask]
    y_train_val = y.loc[~test_mask]
    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

# split train/validation from the remaining (80/20)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_STATE)

print(f"Rows: total={len(df)}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# Cell M1 (fix for OneHotEncoder)

numeric_features = ["R", "B", "xF_Ethanol", "F"]
categorical_features = ["N"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categorical_features),
    ]
)

# Fit preprocessor on training data and transform train/val/test
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)
X_test_prep = preprocessor.transform(X_test)

# Save preprocessor for reuse
import joblib
joblib.dump(preprocessor, "preprocessor_modeling.joblib")
print("Preprocessor saved to: preprocessor_modeling.joblib")

print("Prepared feature shapes:")
print("  X_train_prep:", X_train_prep.shape)
print("  X_val_prep:  ", X_val_prep.shape)
print("  X_test_prep: ", X_test_prep.shape)

# Cell M2: Baseline Polynomial Regression (degree 2 and 3)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# We'll try degree 2 and 3 and compare
results_poly = {}

for deg in [2, 3]:
    model_poly = make_pipeline(
        PolynomialFeatures(degree=deg, include_bias=False),
        LinearRegression()
    )

    # Fit on training
    model_poly.fit(X_train_prep, y_train)

    # Validate
    y_val_pred = model_poly.predict(X_val_prep)

    # Metrics
    mse = mean_squared_error(y_val, y_val_pred, multioutput="raw_values")
    r2 = r2_score(y_val, y_val_pred, multioutput="raw_values")

    results_poly[deg] = {
        "MSE_xD": mse[0],
        "MSE_QR": mse[1],
        "R2_xD": r2[0],
        "R2_QR": r2[1],
        "Model": model_poly,
    }

# Show results
for deg, res in results_poly.items():
    print(f"\nPolynomial Regression (degree={deg})")
    print(f"  MSE_xD: {res['MSE_xD']:.6f}, R2_xD: {res['R2_xD']:.4f}")
    print(f"  MSE_QR: {res['MSE_QR']:.6f}, R2_QR: {res['R2_QR']:.4f}")

# Cell: Random Forest with Hyperparameter Tuning
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV

# Define Random Forest regressor (multi-output)
rf = RandomForestRegressor(random_state=42)

# Wrap for multi-output regression
multi_rf = MultiOutputRegressor(rf)

# Define parameter grid for tuning
param_grid = {
    "estimator__n_estimators": [100, 200],
    "estimator__max_depth": [None, 10, 20],
    "estimator__min_samples_split": [2, 5]
}

# Grid search with 5-fold CV
grid_search = GridSearchCV(
    estimator=multi_rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

# Fit on training data
grid_search.fit(X_train_prep, y_train)

# Get best model
best_rf = grid_search.best_estimator_

# Predictions
y_pred_rf_train = best_rf.predict(X_train_prep)
y_pred_rf_val = best_rf.predict(X_val_prep)

# Convert y to numpy arrays for easier slicing
y_train_np = y_train.to_numpy()
y_val_np = y_val.to_numpy()

# Predictions are already numpy arrays
y_pred_rf_train = best_rf.predict(X_train_prep)
y_pred_rf_val = best_rf.predict(X_val_prep)

# Evaluate
mse_xD_rf = mean_squared_error(y_val_np[:, 0], y_pred_rf_val[:, 0])
r2_xD_rf = r2_score(y_val_np[:, 0], y_pred_rf_val[:, 0])
mse_QR_rf = mean_squared_error(y_val_np[:, 1], y_pred_rf_val[:, 1])
r2_QR_rf = r2_score(y_val_np[:, 1], y_pred_rf_val[:, 1])

print("Random Forest (best params):", grid_search.best_params_)
print(f"  MSE_xD: {mse_xD_rf:.6f}, R2_xD: {r2_xD_rf:.4f}")
print(f"  MSE_QR: {mse_QR_rf:.6f}, R2_QR: {r2_QR_rf:.4f}")

# Cell: Gradient Boosting Regressor with Hyperparameter Tuning
from sklearn.ensemble import GradientBoostingRegressor

# Define GBR model
gbr = GradientBoostingRegressor(random_state=42)

# Multi-output wrapper
multi_gbr = MultiOutputRegressor(gbr)

# Define parameter grid
param_grid_gbr = {
    "estimator__n_estimators": [100, 200],
    "estimator__learning_rate": [0.05, 0.1],
    "estimator__max_depth": [3, 5],
    "estimator__min_samples_split": [2, 5]
}

# Grid Search CV
grid_search_gbr = GridSearchCV(
    estimator=multi_gbr,
    param_grid=param_grid_gbr,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

# Fit
grid_search_gbr.fit(X_train_prep, y_train)

# Get best model
best_gbr = grid_search_gbr.best_estimator_

# Predictions
y_pred_gbr_val = best_gbr.predict(X_val_prep)

# Convert y_val to numpy if needed
y_val_np = y_val.to_numpy()

# Evaluate
mse_xD_gbr = mean_squared_error(y_val_np[:, 0], y_pred_gbr_val[:, 0])
r2_xD_gbr = r2_score(y_val_np[:, 0], y_pred_gbr_val[:, 0])
mse_QR_gbr = mean_squared_error(y_val_np[:, 1], y_pred_gbr_val[:, 1])
r2_QR_gbr = r2_score(y_val_np[:, 1], y_pred_gbr_val[:, 1])

print("Gradient Boosting (best params):", grid_search_gbr.best_params_)
print(f"  MSE_xD: {mse_xD_gbr:.6f}, R2_xD: {r2_xD_gbr:.4f}")
print(f"  MSE_QR: {mse_QR_gbr:.6f}, R2_QR: {r2_QR_gbr:.4f}")

# Cell D1: Physical consistency — enforce bounds and sanity check
import numpy as np
import pandas as pd

# Assuming best model is Random Forest (best_rf) and preprocessor is already fitted
# Transform test features
X_test_prep = preprocessor.transform(X_test)

# Predict
y_test_pred = best_rf.predict(X_test_prep)

# Convert y_test to numpy array
y_test_np = y_test.to_numpy()

# 1) Enforce bounds for xD: 0 <= xD <= 1
y_test_pred[:, 0] = np.clip(y_test_pred[:, 0], 0, 1)

# 2) Quick sanity check
df_diagnostics = pd.DataFrame({
    "R": X_test["R"].values,
    "xF_Ethanol": X_test["xF_Ethanol"].values,
    "N": X_test["N"].values,
    "xD_true": y_test_np[:, 0],
    "xD_pred": y_test_pred[:, 0],
    "QR_true": y_test_np[:, 1],
    "QR_pred": y_test_pred[:, 1]
})

print("Preview of test set predictions with bounds enforced:")
df_diagnostics.head(10)

# 3) Check for any violations outside [0,1] (should be none after clipping)
violations = np.sum((y_test_pred[:,0] < 0) | (y_test_pred[:,0] > 1))
print(f"Number of xD predictions outside [0,1]: {violations}")

# Cell D2: Monotonic sanity check for xD vs R

# Define a small tolerance for noise
tolerance = 1e-3  # allowable small decrease

violations_list = []

# Group by fixed xF and N
grouped = df_diagnostics.groupby(["xF_Ethanol", "N"])

for (xF_val, N_val), group in grouped:
    # Sort by increasing R
    group_sorted = group.sort_values("R")

    # Compute differences in predicted xD
    dxD = np.diff(group_sorted["xD_pred"].values)

    # Check for decreases beyond tolerance
    if np.any(dxD < -tolerance):
        idx_violations = np.where(dxD < -tolerance)[0]
        for idx in idx_violations:
            violations_list.append({
                "xF_Ethanol": xF_val,
                "N": N_val,
                "R_prev": group_sorted["R"].values[idx],
                "R_next": group_sorted["R"].values[idx+1],
                "xD_prev": group_sorted["xD_pred"].values[idx],
                "xD_next": group_sorted["xD_pred"].values[idx+1],
                "dxD": dxD[idx]
            })

# Convert violations to DataFrame for reporting
df_violations = pd.DataFrame(violations_list)

print(f"Number of monotonicity violations: {len(df_violations)}")
df_violations.head(10)

# Cell D3 (fixed): Sensitivity / Partial Dependence Plots for xD
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# 1) xD vs R (for a feed and N close to median)
xF_median_val = df_diagnostics["xF_Ethanol"].median()
N_median_val = df_diagnostics["N"].median()

# Select the closest available xF and N
xF_closest = df_diagnostics["xF_Ethanol"].iloc[(df_diagnostics["xF_Ethanol"] - xF_median_val).abs().argsort()[:1]].values[0]
N_closest = df_diagnostics["N"].iloc[(df_diagnostics["N"] - N_median_val).abs().argsort()[:1]].values[0]

subset_R = df_diagnostics[(df_diagnostics["xF_Ethanol"] == xF_closest) &
                          (df_diagnostics["N"] == N_closest)].sort_values("R")

plt.figure(figsize=(8,5))
plt.plot(subset_R["R"], subset_R["xD_pred"], marker='o', label="Predicted xD")
plt.xlabel("Reflux Ratio (R)")
plt.ylabel("Distillate Mole Fraction xD")
plt.title(f"Sensitivity: xD vs R (xF≈{xF_closest}, N={N_closest})")
plt.grid(True)
plt.legend()
plt.show()

# 2) xD vs xF (for a fixed R and N)
R_median_val = df_diagnostics["R"].median()
R_closest = df_diagnostics["R"].iloc[(df_diagnostics["R"] - R_median_val).abs().argsort()[:1]].values[0]

subset_xF = df_diagnostics[(df_diagnostics["R"] == R_closest) &
                           (df_diagnostics["N"] == N_closest)].sort_values("xF_Ethanol")

plt.figure(figsize=(8,5))
plt.plot(subset_xF["xF_Ethanol"], subset_xF["xD_pred"], marker='o', color='orange', label="Predicted xD")
plt.xlabel("Feed Mole Fraction xF")
plt.ylabel("Distillate Mole Fraction xD")
plt.title(f"Sensitivity: xD vs xF (R≈{R_closest}, N={N_closest})")
plt.grid(True)
plt.legend()
plt.show()

# Cell D4: Error slices for high-purity region (xD >= 0.95)

# Select high-purity rows
high_purity = df_diagnostics[df_diagnostics["xD_true"] >= 0.95]

# Check if there are any high-purity points
if len(high_purity) == 0:
    print("No high-purity points (xD >= 0.95) in the test set.")
else:
    from sklearn.metrics import mean_squared_error, r2_score

    # True and predicted values
    xD_true_hp = high_purity["xD_true"].values
    xD_pred_hp = high_purity["xD_pred"].values
    QR_true_hp = high_purity["QR_true"].values
    QR_pred_hp = high_purity["QR_pred"].values

    # Compute metrics
    mse_xD_hp = mean_squared_error(xD_true_hp, xD_pred_hp)
    r2_xD_hp = r2_score(xD_true_hp, xD_pred_hp)
    mse_QR_hp = mean_squared_error(QR_true_hp, QR_pred_hp)
    r2_QR_hp = r2_score(QR_true_hp, QR_pred_hp)

    print(f"High-purity region (xD >= 0.95) metrics:")
    print(f"  xD: MSE = {mse_xD_hp:.6f}, R2 = {r2_xD_hp:.4f}")
    print(f"  QR: MSE = {mse_QR_hp:.6f}, R2 = {r2_QR_hp:.4f}")

# Cell D5: Model-based sensitivity plots and high-purity metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 1) Define grid for R and xF
R_vals = np.linspace(df_clean["R"].min(), df_clean["R"].max(), 50)
xF_vals = np.linspace(df_clean["xF_Ethanol"].min(), df_clean["xF_Ethanol"].max(), 50)
N_fixed = df_clean["N"].median()  # pick a representative N

# 2) Sweep R (xF fixed) for xD vs R
xF_fixed = df_clean["xF_Ethanol"].median()
df_sweep_R = pd.DataFrame({
    "R": R_vals,
    "xF_Ethanol": [xF_fixed]*len(R_vals),
    "F": [df_clean["F"].median()]*len(R_vals),
    "B": [df_clean["B"].median()]*len(R_vals),
    "N": [N_fixed]*len(R_vals)
})

# Preprocess
X_sweep_R = preprocessor.transform(df_sweep_R)
y_sweep_R_pred = best_rf.predict(X_sweep_R)
y_sweep_R_pred[:,0] = np.clip(y_sweep_R_pred[:,0], 0, 1)

# Plot xD vs R
plt.figure(figsize=(8,5))
plt.plot(R_vals, y_sweep_R_pred[:,0], marker='o', label=f"Predicted xD (xF≈{xF_fixed})")
plt.xlabel("Reflux Ratio R")
plt.ylabel("Distillate Mole Fraction xD")
plt.title("Partial Dependence: xD vs R")
plt.grid(True)
plt.legend()
plt.show()

# 3) Sweep xF (R fixed) for xD vs xF
R_fixed = df_clean["R"].median()
df_sweep_xF = pd.DataFrame({
    "R": [R_fixed]*len(xF_vals),
    "xF_Ethanol": xF_vals,
    "F": [df_clean["F"].median()]*len(xF_vals),
    "B": [df_clean["B"].median()]*len(xF_vals),
    "N": [N_fixed]*len(xF_vals)
})

X_sweep_xF = preprocessor.transform(df_sweep_xF)
y_sweep_xF_pred = best_rf.predict(X_sweep_xF)
y_sweep_xF_pred[:,0] = np.clip(y_sweep_xF_pred[:,0], 0, 1)

# Plot xD vs xF
plt.figure(figsize=(8,5))
plt.plot(xF_vals, y_sweep_xF_pred[:,0], marker='o', color='orange', label=f"Predicted xD (R≈{R_fixed})")
plt.xlabel("Feed Mole Fraction xF")
plt.ylabel("Distillate Mole Fraction xD")
plt.title("Partial Dependence: xD vs xF")
plt.grid(True)
plt.legend()
plt.show()

# 4) High-purity error metrics (xD >= 0.95)
# Compare predictions on the test set
y_test_pred = best_rf.predict(preprocessor.transform(X_test))
y_test_pred[:,0] = np.clip(y_test_pred[:,0], 0, 1)

high_purity_idx = y_test["xD_Ethanol"] >= 0.95
xD_true_hp = y_test["xD_Ethanol"][high_purity_idx].to_numpy()
xD_pred_hp = y_test_pred[high_purity_idx, 0]
QR_true_hp = y_test["QR_kW"][high_purity_idx].to_numpy()
QR_pred_hp = y_test_pred[high_purity_idx, 1]

if len(xD_true_hp) > 0:
    mse_xD_hp = mean_squared_error(xD_true_hp, xD_pred_hp)
    r2_xD_hp = r2_score(xD_true_hp, xD_pred_hp)
    mse_QR_hp = mean_squared_error(QR_true_hp, QR_pred_hp)
    r2_QR_hp = r2_score(QR_true_hp, QR_pred_hp)

    print("High-purity region (xD >= 0.95) metrics:")
    print(f"  xD: MSE = {mse_xD_hp:.6f}, R2 = {r2_xD_hp:.4f}")
    print(f"  QR: MSE = {mse_QR_hp:.6f}, R2 = {r2_QR_hp:.4f}")
else:
    print("No high-purity points (xD >= 0.95) in test set.")

# Cell E1: Standard evaluation metrics for Random Forest (fixed for older sklearn)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predictions on validation set
y_val_pred = best_rf.predict(X_val_prep)
y_val_pred[:,0] = np.clip(y_val_pred[:,0], 0, 1)

# Predictions on test set
y_test_pred = best_rf.predict(X_test_prep)
y_test_pred[:,0] = np.clip(y_test_pred[:,0], 0, 1)

# Function to compute metrics
def evaluate_metrics(y_true, y_pred, labels=["xD","QR"]):
    metrics = {}
    for i, label in enumerate(labels):
        mae = mean_absolute_error(y_true[:,i], y_pred[:,i])
        rmse = np.sqrt(mean_squared_error(y_true[:,i], y_pred[:,i]))  # compute RMSE manually
        r2 = r2_score(y_true[:,i], y_pred[:,i])
        metrics[label] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return metrics

# Convert y_val and y_test to numpy arrays if not already
y_val_np = y_val.to_numpy()
y_test_np = y_test.to_numpy()

# Evaluate
val_metrics = evaluate_metrics(y_val_np, y_val_pred)
test_metrics = evaluate_metrics(y_test_np, y_test_pred)

print("Validation Set Metrics:")
for k,v in val_metrics.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

print("\nTest Set Metrics:")
for k,v in test_metrics.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

# Cell E2: Parity plots for xD and QR
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Helper function to plot parity
def parity_plot(y_true, y_pred, title, xlabel="Simulated", ylabel="Predicted"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Validation set
parity_plot(y_val_np[:,0], y_val_pred[:,0], title="Parity Plot: xD (Validation)")
parity_plot(y_val_np[:,1], y_val_pred[:,1], title="Parity Plot: QR (Validation)")

# Test set
parity_plot(y_test_np[:,0], y_test_pred[:,0], title="Parity Plot: xD (Test)")
parity_plot(y_test_np[:,1], y_test_pred[:,1], title="Parity Plot: QR (Test)")

# Cell E3 (fixed): Residual plots vs each input for test set
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Compute residuals for test set
residuals_xD = y_test_pred[:,0] - y_test_np[:,0]
residuals_QR = y_test_pred[:,1] - y_test_np[:,1]

inputs = ["R", "xF_Ethanol", "F", "N"]
residuals = {"xD": residuals_xD, "QR": residuals_QR}

# Use test set inputs (not df_clean)
X_test_inputs = X_test.copy()  # make sure X_test has the original column names

# Plot residuals
for output in ["xD", "QR"]:
    plt.figure(figsize=(12,8))
    for i, feature in enumerate(inputs):
        plt.subplot(2,2,i+1)
        plt.scatter(X_test_inputs[feature], residuals[output], alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(feature)
        plt.ylabel(f"{output} Residual")
        plt.title(f"{output} Residual vs {feature}")
    plt.tight_layout()
    plt.show()

# Cell E4: Generalization test (leave out R ∈ [3.5, 4.5])
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define the excluded R region
R_lower, R_upper = 3.5, 4.5

# Split training data: exclude R ∈ [3.5,4.5]
train_mask = ~((df_clean["R"] >= R_lower) & (df_clean["R"] <= R_upper))
df_train_gen = df_clean[train_mask]
df_test_gen = df_clean[~train_mask]  # only the excluded region

# Inputs and outputs
X_train_gen = df_train_gen[["R","B","xF_Ethanol","F","N"]]
y_train_gen = df_train_gen[["xD_Ethanol","QR_kW"]]
X_test_gen = df_test_gen[["R","B","xF_Ethanol","F","N"]]
y_test_gen = df_test_gen[["xD_Ethanol","QR_kW"]]

# Preprocess
numeric_features = ["R","B","xF_Ethanol","F"]
categorical_features = ["N"]

preprocessor_gen = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

X_train_gen_prep = preprocessor_gen.fit_transform(X_train_gen)
X_test_gen_prep = preprocessor_gen.transform(X_test_gen)

# Train Random Forest
rf_gen = RandomForestRegressor(n_estimators=100, random_state=42)
rf_gen.fit(X_train_gen_prep, y_train_gen)

# Predict on excluded region
y_test_gen_pred = rf_gen.predict(X_test_gen_prep)
y_test_gen_pred[:,0] = np.clip(y_test_gen_pred[:,0], 0, 1)

# Evaluate metrics
def evaluate_metrics(y_true, y_pred, labels=["xD","QR"]):
    metrics = {}
    for i, label in enumerate(labels):
        mae = mean_absolute_error(y_true[:,i], y_pred[:,i])
        rmse = np.sqrt(mean_squared_error(y_true[:,i], y_pred[:,i]))
        r2 = r2_score(y_true[:,i], y_pred[:,i])
        metrics[label] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return metrics

y_test_gen_np = y_test_gen.to_numpy()
gen_metrics = evaluate_metrics(y_test_gen_np, y_test_gen_pred)

print(f"Generalization Test Metrics (R ∈ [{R_lower},{R_upper}] excluded during training):")
for k,v in gen_metrics.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

# Cell G1: Generalization test for Polynomial Regression (degree=3)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Excluded region
R_lower, R_upper = 3.5, 4.5
train_mask = ~((df_clean["R"] >= R_lower) & (df_clean["R"] <= R_upper))
df_train_gen = df_clean[train_mask]
df_test_gen = df_clean[~train_mask]  # excluded region

# Inputs and outputs
X_train_gen = df_train_gen[["R","B","xF_Ethanol","F","N"]]
y_train_gen = df_train_gen[["xD_Ethanol","QR_kW"]]
X_test_gen = df_test_gen[["R","B","xF_Ethanol","F","N"]]
y_test_gen = df_test_gen[["xD_Ethanol","QR_kW"]]

# Preprocess: one-hot for N, scale numeric
numeric_features = ["R","B","xF_Ethanol","F"]
categorical_features = ["N"]

preprocessor_gen = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

# Polynomial regression degree 3, multi-output
X_train_prep = preprocessor_gen.fit_transform(X_train_gen)
X_test_prep = preprocessor_gen.transform(X_test_gen)

poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_prep)
X_test_poly = poly.transform(X_test_prep)

# Train multi-output linear regression
from sklearn.multioutput import MultiOutputRegressor
poly_model = MultiOutputRegressor(LinearRegression())
poly_model.fit(X_train_poly, y_train_gen)

# Predict
y_test_pred = poly_model.predict(X_test_poly)
y_test_pred[:,0] = np.clip(y_test_pred[:,0], 0, 1)

# Evaluate
y_test_np = y_test_gen.to_numpy()
gen_metrics_poly = evaluate_metrics(y_test_np, y_test_pred)

print(f"Generalization Test Metrics for Polynomial Regression (degree=3):")
for k,v in gen_metrics_poly.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

# Cell G2: Generalization test for Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor

# Excluded region
R_lower, R_upper = 3.5, 4.5
train_mask = ~((df_clean["R"] >= R_lower) & (df_clean["R"] <= R_upper))
df_train_gen = df_clean[train_mask]
df_test_gen = df_clean[~train_mask]  # excluded region

# Inputs and outputs
X_train_gen = df_train_gen[["R","B","xF_Ethanol","F","N"]]
y_train_gen = df_train_gen[["xD_Ethanol","QR_kW"]]
X_test_gen = df_test_gen[["R","B","xF_Ethanol","F","N"]]
y_test_gen = df_test_gen[["xD_Ethanol","QR_kW"]]

# Preprocess: one-hot for N, scale numeric
numeric_features = ["R","B","xF_Ethanol","F"]
categorical_features = ["N"]

preprocessor_gen = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ]
)

X_train_prep = preprocessor_gen.fit_transform(X_train_gen)
X_test_prep = preprocessor_gen.transform(X_test_gen)

# Train multi-output Gradient Boosting
gb_model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
)
gb_model.fit(X_train_prep, y_train_gen)

# Predict
y_test_pred = gb_model.predict(X_test_prep)
y_test_pred[:,0] = np.clip(y_test_pred[:,0], 0, 1)

# Evaluate
y_test_np = y_test_gen.to_numpy()
gen_metrics_gb = evaluate_metrics(y_test_np, y_test_pred)

print(f"Generalization Test Metrics for Gradient Boosting:")
for k,v in gen_metrics_gb.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

# Cell F1: Polynomial Regression Parity Plots + Metrics

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

# Prepare validation and test sets
X_val_poly_prep = preprocessor.transform(X_val)
X_val_poly = poly.transform(X_val_poly_prep)

X_test_poly_prep = preprocessor.transform(X_test)
X_test_poly = poly.transform(X_test_poly_prep)

# Predict
y_val_pred_poly = poly_model.predict(X_val_poly)
y_test_pred_poly = poly_model.predict(X_test_poly)

# Clip xD predictions
y_val_pred_poly[:,0] = np.clip(y_val_pred_poly[:,0], 0, 1)
y_test_pred_poly[:,0] = np.clip(y_test_pred_poly[:,0], 0, 1)


# Fixed evaluate_metrics function for older sklearn versions
def evaluate_metrics(y_true, y_pred, labels=["xD","QR"]):
    metrics = {}
    for i, label in enumerate(labels):
        mae = mean_absolute_error(y_true[:,i], y_pred[:,i])
        rmse = np.sqrt(mean_squared_error(y_true[:,i], y_pred[:,i]))  # use sqrt manually
        r2 = r2_score(y_true[:,i], y_pred[:,i])
        metrics[label] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    return metrics

y_val_np = y_val.to_numpy()
y_test_np = y_test.to_numpy()

val_metrics_poly = evaluate_metrics(y_val_np, y_val_pred_poly)
test_metrics_poly = evaluate_metrics(y_test_np, y_test_pred_poly)

print("Validation Set Metrics (Polynomial Regression):")
for k,v in val_metrics_poly.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

print("\nTest Set Metrics (Polynomial Regression):")
for k,v in test_metrics_poly.items():
    print(f"  {k}: MAE={v['MAE']:.6f}, RMSE={v['RMSE']:.6f}, R2={v['R2']:.4f}")

# Parity plots function
def parity_plot(y_true, y_pred, title, xlabel="Simulated", ylabel="Predicted"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

# Parity plots
parity_plot(y_val_np[:,0], y_val_pred_poly[:,0], "xD (Validation)")
parity_plot(y_val_np[:,1], y_val_pred_poly[:,1], "QR (Validation)")
parity_plot(y_test_np[:,0], y_test_pred_poly[:,0], "xD (Test)")
parity_plot(y_test_np[:,1], y_test_pred_poly[:,1], "QR (Test)")

import matplotlib.pyplot as plt
import seaborn as sns

# List of features to visualize
features = ["R", "B", "xF_Ethanol", "F", "N", "xD_Ethanol", "QR_kW"]

# Set up the plotting grid
plt.figure(figsize=(18, 12))

for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(df_clean[feature], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {feature}', fontsize=12)
    plt.xlabel(feature)
    plt.ylabel('Count')

plt.tight_layout()

# Save the figure as PNG with high resolution
plt.savefig("df_clean_distribution.png", dpi=300)

# Show the plot
plt.show()

