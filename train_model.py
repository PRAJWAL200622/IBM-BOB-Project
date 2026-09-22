# =============================================================================
# train_model.py
# Used Car Price Prediction - Machine Learning Model Training
# =============================================================================
# What this file does (step by step):
#   1. Loads the preprocessed training and testing data
#   2. Trains 4 regression models:
#        - Linear Regression      (simple baseline)
#        - Decision Tree          (tree-based, easy to understand)
#        - Random Forest          (many trees combined - usually very accurate)
#        - Gradient Boosting      (trees built one after another to fix errors)
#   3. Evaluates every model using:
#        - MAE  (Mean Absolute Error)   - average error in price prediction
#        - RMSE (Root Mean Squared Error) - penalises large errors more
#        - R2   (R-squared Score)       - how much of the price variation
#                                         the model explains (1.0 = perfect)
#   4. Picks the best model by highest R2 score
#   5. Saves the best model + feature column list to the models/ folder
# =============================================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# =============================================================================
# STEP 1 - Load the preprocessed data
# =============================================================================
print("=" * 65)
print("STEP 1: Loading preprocessed data ...")
print("=" * 65)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()  # convert 1-col DataFrame to Series
y_test  = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print(f"  [OK] X_train : {X_train.shape[0]} rows x {X_train.shape[1]} features")
print(f"  [OK] X_test  : {X_test.shape[0]}  rows x {X_test.shape[1]} features")
print(f"  [OK] Feature columns: {X_train.columns.tolist()}\n")

# Save feature column order - the web app will need this to build input arrays
feature_cols = X_train.columns.tolist()
joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
print(f"  [OK] Feature column list saved -> models/feature_columns.pkl\n")

# =============================================================================
# STEP 2 - Define the models to train and compare
# =============================================================================
# Each entry is:  "display name" : model object
#
# LINEAR REGRESSION
#   The simplest model. Draws a straight line through the data.
#   Fast but may miss complex patterns.
#
# DECISION TREE
#   Asks a series of yes/no questions about the car (like a flowchart).
#   Can overfit (memorise training data) if not controlled.
#
# RANDOM FOREST
#   Builds hundreds of Decision Trees on random subsets of data,
#   then averages their predictions. Much more accurate than a single tree.
#
# GRADIENT BOOSTING
#   Builds trees one by one; each new tree focuses on the mistakes made
#   by the previous trees. Very accurate but a bit slower to train.
# =============================================================================
print("=" * 65)
print("STEP 2: Defining models to train ...")
print("=" * 65)

models = {
    "Linear Regression"   : LinearRegression(),
    "Decision Tree"       : DecisionTreeRegressor(random_state=42),
    "Random Forest"       : RandomForestRegressor(
                                n_estimators=100,   # 100 trees
                                random_state=42,
                                n_jobs=-1           # use all CPU cores
                            ),
    "Gradient Boosting"   : GradientBoostingRegressor(
                                n_estimators=200,   # 200 boosting stages
                                learning_rate=0.1,  # how much each tree corrects
                                max_depth=5,        # max depth per tree
                                random_state=42
                            ),
}

print(f"  [OK] {len(models)} models defined: {list(models.keys())}\n")

# =============================================================================
# STEP 3 - Train and evaluate every model
# =============================================================================
# NOTE: selling_price was log-transformed during preprocessing.
# We use np.expm1() (the reverse of log1p) to convert predictions back to
# actual rupee values before calculating MAE and RMSE, so the errors are
# readable in real rupees.  R2 is calculated on the log scale (standard practice).
# =============================================================================
print("=" * 65)
print("STEP 3: Training and evaluating all models ...")
print("=" * 65)

results = {}   # stores evaluation metrics for every model

for name, model in models.items():
    print(f"\n  --- {name} ---")

    # ── Train ────────────────────────────────────────────────────────────────
    model.fit(X_train, y_train)
    print(f"    [OK] Training complete.")

    # ── Predict on the test set ───────────────────────────────────────────────
    y_pred_log = model.predict(X_test)   # predictions in log scale

    # ── Convert back to actual rupees for MAE and RMSE ───────────────────────
    y_pred_actual = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # ── Calculate metrics ────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2   = r2_score(y_test, y_pred_log)   # R2 on log scale

    results[name] = {"model": model, "MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"    MAE  (avg error in Rs) : Rs {mae:,.0f}")
    print(f"    RMSE (penalised error) : Rs {rmse:,.0f}")
    print(f"    R2   (accuracy score)  : {r2:.4f}  ({r2*100:.2f}%)")

# =============================================================================
# STEP 4 - Print a side-by-side comparison table
# =============================================================================
print("\n")
print("=" * 65)
print("STEP 4: Model Comparison Summary")
print("=" * 65)
print(f"  {'Model':<25} {'MAE (Rs)':>14} {'RMSE (Rs)':>14} {'R2 Score':>10}")
print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*10}")

for name, m in results.items():
    print(f"  {name:<25} {m['MAE']:>14,.0f} {m['RMSE']:>14,.0f} {m['R2']:>10.4f}")

# =============================================================================
# STEP 5 - Select the best model (highest R2 score)
# =============================================================================
print("\n")
print("=" * 65)
print("STEP 5: Selecting the best model ...")
print("=" * 65)

best_name  = max(results, key=lambda k: results[k]["R2"])
best_model = results[best_name]["model"]
best_r2    = results[best_name]["R2"]
best_mae   = results[best_name]["MAE"]
best_rmse  = results[best_name]["RMSE"]

print(f"  [WINNER] {best_name}")
print(f"    MAE  : Rs {best_mae:,.0f}")
print(f"    RMSE : Rs {best_rmse:,.0f}")
print(f"    R2   : {best_r2:.4f}  ({best_r2*100:.2f}%)")

# =============================================================================
# STEP 6 - Save the best model and all individual models
# =============================================================================
print("\n")
print("=" * 65)
print("STEP 6: Saving models ...")
print("=" * 65)

# Save the best model under a clear filename the API will import
best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
joblib.dump(best_model, best_model_path)
print(f"  [OK] Best model ({best_name}) saved -> models/best_model.pkl")

# Save the name of the best model so the API knows which one was chosen
joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))
print(f"  [OK] Best model name saved           -> models/best_model_name.pkl")

# Save each individual model too (useful for the analytics dashboard later)
safe_names = {
    "Linear Regression" : "model_linear_regression.pkl",
    "Decision Tree"     : "model_decision_tree.pkl",
    "Random Forest"     : "model_random_forest.pkl",
    "Gradient Boosting" : "model_gradient_boosting.pkl",
}
for name, filename in safe_names.items():
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(results[name]["model"], path)
    print(f"  [OK] {name:<25} saved -> models/{filename}")

# Save the comparison results as a CSV so the dashboard can display them
results_rows = []
for name, m in results.items():
    results_rows.append({
        "Model"   : name,
        "MAE"     : round(m["MAE"], 2),
        "RMSE"    : round(m["RMSE"], 2),
        "R2_Score": round(m["R2"], 4),
        "Selected": "YES" if name == best_name else "NO"
    })

results_df = pd.DataFrame(results_rows)
results_csv_path = os.path.join(DATA_DIR, "model_comparison.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"  [OK] Comparison table saved          -> data/model_comparison.csv")

# =============================================================================
# Done
# =============================================================================
print()
print("=" * 65)
print("  ALL STEPS COMPLETE - Model training finished successfully!")
print(f"  Best model : {best_name}")
print(f"  R2 Score   : {best_r2*100:.2f}%")
print("=" * 65)
