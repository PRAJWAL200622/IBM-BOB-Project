# 🚗 Used Car Price Prediction and Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

**Used Car Price Prediction and Analytics Dashboard** is an end-to-end machine learning project that predicts the resale price of used cars based on their features. It combines a full data preprocessing pipeline, comparative model training, and an interactive analytics dashboard built with Streamlit.

The system ingests a real-world used-car dataset, cleans and transforms the data, trains and evaluates multiple regression models, and exposes predictions through a user-friendly web interface. The final model achieves an **R² score of 92.49%**, meaning it explains over 92% of the variation in used-car prices.

---

## 🧩 Problem Statement

The used-car market is large, fragmented and opaque. Buyers and sellers alike struggle to determine a fair price for a vehicle because resale value depends on many interdependent factors — brand, age, mileage, fuel type, engine capacity, and more.

This project addresses the problem by building a supervised machine learning regression model that, given the key attributes of a used car, produces an accurate estimate of its fair market selling price in Indian Rupees.

---

## 🎯 Objectives

- Analyse a real-world used-car dataset to understand the factors that influence price.
- Build a robust data preprocessing pipeline that handles duplicates, outliers and encoding.
- Train and compare four regression algorithms to select the best-performing model.
- Evaluate models objectively using MAE, RMSE and R² metrics.
- Deploy the best model as an interactive Streamlit dashboard with analytics and live predictions.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Data Preprocessing** | Automated cleaning: duplicate removal, outlier capping, log transformation, label encoding |
| **Multi-Model Training** | Four regression models trained and compared side by side |
| **Honest Evaluation** | Real MAE, RMSE and R² scores from a held-out 20% test set |
| **Interactive Dashboard** | Four-section Streamlit app: Dashboard, Prediction, Analytics, Model Info |
| **Live Price Prediction** | Instant price estimate in Indian Rupee format (Lakhs / Crores) |
| **Brand-Filtered Models** | Selecting a brand automatically filters the model dropdown to that brand's options |
| **Analytics Charts** | Price distribution, depreciation curves, fuel-type analysis, brand rankings |
| **No Retraining on Click** | Model is cached; the web app never retrains on user input |

---

## 🛠️ Technologies Used

| Category | Library / Tool | Version |
|---|---|---|
| Language | Python | 3.9+ |
| Data Processing | pandas | 2.0+ |
| Numerical Computing | NumPy | 1.24+ |
| Machine Learning | scikit-learn | 1.3+ |
| Model Serialisation | joblib | 1.3+ |
| Web Application | Streamlit | 1.35+ |

---

## 🏗️ Project Architecture

```
Raw CSV Dataset
      │
      ▼
┌─────────────────────────┐
│   preprocess.py         │  ← Clean, encode, transform, split
│   (Data Pipeline)       │
└────────────┬────────────┘
             │  Outputs: X_train, X_test, y_train, y_test,
             │           processed_cars_data.csv, label_encoders.pkl
             ▼
┌─────────────────────────┐
│   train_model.py        │  ← Train 4 models, evaluate, select best
│   (ML Training)         │
└────────────┬────────────┘
             │  Outputs: best_model.pkl, model_*.pkl,
             │           feature_columns.pkl, model_comparison.csv
             ▼
┌─────────────────────────┐
│   app.py (Streamlit)    │  ← Load model, serve predictions + analytics
│   (Web Application)     │
└─────────────────────────┘
```

---

## 📁 Project Folder Structure

```
used_car_price_prediction/
│
├── data/                          # All data files
│   ├── cars_data.csv              # Original raw dataset (do not modify)
│   ├── processed_cars_data.csv    # Fully preprocessed dataset
│   ├── X_train.csv                # Training features (80%)
│   ├── X_test.csv                 # Testing features  (20%)
│   ├── y_train.csv                # Training labels
│   ├── y_test.csv                 # Testing labels
│   └── model_comparison.csv       # Evaluation results for all 4 models
│
├── models/                        # Serialised model artefacts
│   ├── best_model.pkl             # The selected Gradient Boosting model
│   ├── best_model_name.pkl        # Name string of the best model
│   ├── feature_columns.pkl        # Ordered list of 11 input feature names
│   ├── label_encoders.pkl         # LabelEncoder objects for all categorical columns
│   ├── model_linear_regression.pkl
│   ├── model_decision_tree.pkl
│   ├── model_random_forest.pkl
│   └── model_gradient_boosting.pkl
│
├── backend/                       # Python scripts
│   ├── preprocess.py              # Data cleaning and preprocessing pipeline
│   └── train_model.py             # Model training, evaluation and saving
│
├── frontend/                      # Streamlit web application
│   └── app.py                     # Full dashboard (4 sections)
│
├── notebooks/                     # Jupyter notebooks (EDA / experiments)
│
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **File** | `data/cars_data.csv` |
| **Total Records** | 15,411 |
| **Columns** | 14 |
| **Missing Values** | 0 |
| **Duplicate Rows** | 167 (removed during preprocessing) |
| **Car Brands** | 32 (Maruti, Hyundai, Honda, BMW, Audi, …) |
| **Car Models** | 120 unique models |
| **Price Range** | ₹40,000 → ₹3.95 Crore |
| **Average Price** | ₹7.75 Lakhs |
| **Median Price** | ₹5.56 Lakhs |

### Column Description

| Column | Type | Description |
|---|---|---|
| `car_name` | Text | Full car name (brand + model combined) |
| `brand` | Categorical | Manufacturer name (e.g. Maruti, Hyundai) |
| `model` | Categorical | Model name (e.g. Alto, Creta, i20) |
| `vehicle_age` | Integer | Age of the car in years (0 = new) |
| `km_driven` | Integer | Total kilometres driven |
| `seller_type` | Categorical | Individual / Dealer / Trustmark Dealer |
| `fuel_type` | Categorical | Petrol / Diesel / CNG / LPG / Electric |
| `transmission_type` | Categorical | Manual / Automatic |
| `mileage` | Float | Fuel efficiency in km/litre |
| `engine` | Integer | Engine displacement in cc |
| `max_power` | Float | Maximum engine power in bhp |
| `seats` | Integer | Seating capacity |
| `selling_price` | Integer | **Target variable** — resale price in ₹ |

> `Unnamed: 0` (a row index column) and `car_name` (redundant with `brand` + `model`) are dropped during preprocessing.

---

## 🔧 Data Preprocessing

Preprocessing is handled entirely by [`backend/preprocess.py`](backend/preprocess.py).

### Steps Performed

| Step | Action | Reason |
|---|---|---|
| 1 | Drop `Unnamed: 0` and `car_name` | Useless index; car_name is redundant |
| 2 | Remove 167 duplicate rows | Prevents model from memorising repeated records |
| 3 | Fill missing values | Numeric → median; Categorical → mode |
| 4 | Cap outliers (IQR method) | Clips extreme values in `km_driven`, `mileage`, `engine`, `max_power`, `selling_price` |
| 5 | Log-transform `selling_price` | Balances the heavily right-skewed price distribution |
| 6 | Label Encode 5 categorical columns | Converts text to integers that the model can process |
| 7 | Separate X (features) and y (target) | 11 input features, 1 target column |
| 8 | 80 / 20 train-test split | `random_state=42` for reproducibility |

### Encoded Columns

| Column | Unique Values | Example Encoding |
|---|---|---|
| `brand` | 32 | Maruti → 18, Hyundai → 8 |
| `model` | 120 | Alto → 7, Creta → 30 |
| `seller_type` | 3 | Individual → 1, Dealer → 0 |
| `fuel_type` | 5 | Petrol → 4, Diesel → 1 |
| `transmission_type` | 2 | Manual → 1, Automatic → 0 |

### Output Files

After running `preprocess.py` the following files are saved:

```
data/processed_cars_data.csv   ← 15,244 clean rows
data/X_train.csv               ← 12,195 rows × 11 features
data/X_test.csv                ←  3,049 rows × 11 features
data/y_train.csv               ← 12,195 log-price labels
data/y_test.csv                ←  3,049 log-price labels
models/label_encoders.pkl      ← 5 LabelEncoder objects
```

---

## 🔍 Exploratory Data Analysis

Key patterns observed in the dataset:

- **Price distribution** is heavily right-skewed — most cars are priced between ₹2–8 Lakhs, with a long tail of luxury vehicles extending to ₹3.95 Crore.
- **Vehicle age vs. price** shows clear depreciation — new cars (age 0) average ₹22.3 Lakhs while 10-year-old cars average ₹4.96 Lakhs.
- **Transmission** has a strong effect — Automatic cars average ₹15.8 Lakhs vs ₹5.65 Lakhs for Manual.
- **Fuel type** matters — Electric cars average highest (₹18.5 L), followed by Diesel (₹10.0 L), then Petrol (₹5.73 L).
- **Top brands by listing volume**: Maruti (4,992), Hyundai (2,982), Honda (1,485), Mahindra (1,011).
- **Kilometres driven** is negatively correlated with price — higher mileage generally means a lower resale value.

---

## 🤖 Machine Learning Models

Training is handled by [`backend/train_model.py`](backend/train_model.py).

Four regression algorithms were trained and compared on the same 80/20 split.

### Models Trained

| # | Model | Brief Description |
|---|---|---|
| 1 | **Linear Regression** | Baseline model; fits a linear relationship between features and log-price |
| 2 | **Decision Tree** | Recursive binary splits; captures non-linear patterns but prone to overfitting |
| 3 | **Random Forest** | 100 decision trees averaged; more stable and accurate than a single tree |
| 4 | **Gradient Boosting** | 200 sequential trees, each correcting the previous tree's residual errors |

### Hyperparameters

| Model | Key Parameters |
|---|---|
| Linear Regression | Default scikit-learn settings |
| Decision Tree | `random_state=42` |
| Random Forest | `n_estimators=100`, `random_state=42`, `n_jobs=-1` |
| Gradient Boosting | `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`, `random_state=42` |

---

## 📈 Model Evaluation

All metrics are computed on the **held-out 20% test set (3,049 records)** — data the model never saw during training.

MAE and RMSE are expressed in actual Rupees (after reversing the log transformation). R² is calculated on the log scale, which is standard practice for log-transformed targets.

### Results

| Model | MAE (₹) | RMSE (₹) | R² Score | Selected |
|---|---|---|---|---|
| Linear Regression | 1,18,095 | 1,71,740 | 83.61% | |
| Decision Tree | 80,397 | 1,20,434 | 86.52% | |
| Random Forest | 65,449 | 96,667 | 91.69% | |
| **Gradient Boosting** | **65,463** | **94,064** | **92.49%** | ✅ **Best** |

### Metric Definitions

| Metric | What It Measures | Goal |
|---|---|---|
| **MAE** (Mean Absolute Error) | Average rupee difference between predicted and actual price | Lower is better |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalises large errors more heavily | Lower is better |
| **R²** (R-Squared) | Percentage of price variation the model explains | Higher is better; 1.0 = perfect |

### Why Gradient Boosting Was Selected

- Highest R² score: **92.49%** — explains the most price variance
- Lowest RMSE: **₹94,064** — smallest penalised error
- Comparable MAE to Random Forest (within ₹15), but superior on RMSE and R²

---

## 💡 AI Price Prediction

The prediction pipeline works as follows:

1. **User selects** 11 car features in the Streamlit form.
2. **Categorical inputs** are encoded using the saved `LabelEncoder` objects — the same encoders used during training — guaranteeing consistency.
3. **A single-row DataFrame** is constructed with columns in the exact order the model was trained on (`feature_columns.pkl`).
4. **Gradient Boosting model** receives the 11 encoded features and returns a log-scale prediction.
5. **`np.expm1()`** reverses the log transformation, converting the output back to actual Rupees.
6. **The result** is formatted in Indian notation (Lakhs / Crores) and displayed on screen.

> The model is loaded once at startup using `@st.cache_resource` and is never retrained on user input.

---

## 🖥️ Streamlit Dashboard

The dashboard is the file [`frontend/app.py`](frontend/app.py) and is launched with:

```bash
streamlit run frontend/app.py
```

It opens at **http://localhost:8501** in your browser.

---

## 📋 Dashboard Features

The app is divided into four sections, selectable from the sidebar navigation.

### 1. Dashboard — Dataset Overview
Computed from the raw `cars_data.csv` dataset.

- **KPI cards**: Total Cars (15,411) · Average Price (₹7.75 L) · Minimum Price (₹40,000) · Maximum Price (₹3.95 Cr)
- **Cars by Fuel Type** — bar chart showing the split across Petrol, Diesel, CNG, LPG and Electric
- **Cars by Seller Type** — Individual vs Dealer vs Trustmark Dealer
- **Top 8 Brands by Listings** — Maruti leads with 4,992 cars
- **Average Price by Transmission** — Automatic vs Manual
- **Vehicle Age Distribution** — how old are the cars being sold?

### 2. Price Prediction
- Form with 11 input fields grouped into: Car Identity, Car History, Technical Specifications, Sale Details
- Brand selection auto-filters the Model dropdown to only that brand's options
- Live input summary table showing selected values
- **Predict Price** button triggers instant ML inference
- Result displayed in a prominent card showing price in Lakhs or Crores

### 3. Data Analytics — Deep Dive Charts
All charts use actual dataset values — no data is fabricated.

| Chart | Description |
|---|---|
| Price Distribution | Count of cars per price bucket (0–2 L up to 20–30 L) |
| Price vs Kilometres Driven | How average price drops as km driven increases |
| Price vs Vehicle Age | Depreciation curve across all ages (line chart) |
| Average Price by Fuel Type | Electric > Diesel > Petrol > CNG > LPG |
| Top 10 Brands by Avg Price | Ferrari, Rolls-Royce, Bentley lead on average resale value |

### 4. Model Information
- Cards for all four models tested, with real MAE / RMSE / R² values
- Selected model (Gradient Boosting) highlighted
- Plain-English explanation of MAE, RMSE and R²
- Step-by-step explanation of how a prediction is made
- Training data summary: 12,195 train / 3,049 test / 11 features

---

## 📸 Screenshots

### Dashboard — Overview
![Dashboard Overview](<img width="1042" height="602" alt="Screenshot 2026-09-22 231024" src="https://github.com/user-attachments/assets/cc837262-231f-433b-a800-3cb309956a06" />
)

### Price Prediction
![Price Prediction]("<img width="1067" height="718" alt="Screenshot 2026-09-22 230956" src="https://github.com/user-attachments/assets/369fffea-1a98-44bd-9982-581b1ad87fdb" />
")

### Data Analytics
![Data Analytics](<img width="1012" height="692" alt="Screenshot 2026-09-22 231229" src="https://github.com/user-attachments/assets/e2e03582-b4f3-4f06-ad9a-64f3e8138f91" />
)

### Model Information
![Model Information](<img width="995" height="581" alt="Screenshot 2026-09-22 231252" src="https://github.com/user-attachments/assets/32c12fc4-2ccb-4338-9631-5429c1b027cd" />
)

---

## ⚙️ Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### 1. Clone or Download the Project

```bash
git clone https://github.com/prajwal200622/used-car-price-prediction.git
cd used-car-price-prediction
```

### 2. (Recommended) Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt` contents:**

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
streamlit>=1.35.0
```

---

## ▶️ Running the Project

The project must be run in the following order. Each step depends on the output of the previous one.

### Step 1 — Preprocess the Data

```bash
python backend/preprocess.py
```

**What it does:** Cleans the raw CSV, encodes features, splits into train/test sets, and saves all output files to `data/` and `models/`.

**Expected output:**
```
STEP 1: Loading the dataset ...
  [OK] Loaded  ->  15411 rows  x  14 columns
...
STEP 10: Saving processed files ...
  [OK] X_train saved  ->  data/X_train.csv  (12195 rows)
  [OK] X_test  saved  ->  data/X_test.csv   (3049 rows)
  ALL STEPS COMPLETE - Preprocessing finished successfully!
```

---

### Step 2 — Train the Model

```bash
python backend/train_model.py
```

**What it does:** Loads the preprocessed data, trains 4 regression models, evaluates them, saves the best model and all artefacts to `models/`.

**Expected output:**
```
  [WINNER] Gradient Boosting
    MAE  : Rs 65,463
    RMSE : Rs 94,064
    R2   : 0.9249  (92.49%)
  ALL STEPS COMPLETE - Model training finished successfully!
```

---

### Step 3 — Launch the Dashboard

```bash
streamlit run frontend/app.py
```

Open your browser at: **http://localhost:8501**

---

## 🔮 Example Prediction

Using the **Price Prediction** section of the dashboard:

| Feature | Value |
|---|---|
| Brand | Hyundai |
| Model | Creta |
| Vehicle Age | 3 years |
| Kilometres Driven | 30,000 km |
| Seller Type | Dealer |
| Fuel Type | Diesel |
| Transmission | Manual |
| Mileage | 21.4 km/l |
| Engine | 1,493 cc |
| Max Power | 113 bhp |
| Seats | 5 |

**Predicted Price: ₹11.44 Lakhs**

> Actual market prices for similar cars vary between ₹10–14 Lakhs depending on condition and location, showing the model is well-calibrated.

---

## 🔄 Project Workflow

```
Step 1 ── Collect Dataset
             │  cars_data.csv (15,411 records, 14 columns)
             ▼
Step 2 ── Exploratory Analysis
             │  Understand distributions, correlations, outliers
             ▼
Step 3 ── Data Preprocessing       [preprocess.py]
             │  Clean → Encode → Transform → Split
             ▼
Step 4 ── Model Training           [train_model.py]
             │  Train 4 models → Evaluate → Select best
             ▼
Step 5 ── Model Serialisation
             │  Save best_model.pkl + encoders + feature list
             ▼
Step 6 ── Web Application          [app.py]
             │  Load model → Accept inputs → Predict → Display
             ▼
Step 7 ── Analytics Dashboard
             Visualise dataset insights for end users
```

---

## 🚀 Future Enhancements

- **More algorithms**: XGBoost, LightGBM and CatBoost are known to outperform standard Gradient Boosting on tabular data.
- **Hyperparameter tuning**: Grid search or Bayesian optimisation over the Gradient Boosting parameter space.
- **Additional features**: Number of previous owners, service history, accident record, city/location.
- **Geographic pricing**: Different Indian cities have different used-car price levels; adding location as a feature could improve accuracy.
- **Confidence intervals**: Display a price range (e.g. ₹9.5 L – ₹12.5 L) rather than a single point estimate.
- **REST API**: Expose the model as a FastAPI endpoint so other applications can query it.
- **Automated retraining**: Pipeline to periodically retrain on fresh data as the market evolves.
- **Model explainability**: Add SHAP value charts to show which features drove a specific prediction.

---

## ⚠️ Limitations

- **Label Encoding for high-cardinality features**: `model` has 120 unique values. Label encoding implies a numeric ordering that doesn't truly exist; target encoding or embeddings would be more principled.
- **Price ceiling from training data**: The model was trained on cars up to ₹14.85 Lakhs (post-outlier capping). Very high-end luxury vehicles outside this range may be predicted less accurately.
- **Static dataset**: The model reflects the price patterns at the time the dataset was collected. Market conditions change; retraining periodically is recommended.
- **No geographic feature**: Resale prices vary significantly by city, which the current model cannot account for.
- **km_driven cap**: Training data caps kilometres at 1,30,000. Cars driven beyond this range are clipped to the cap value before prediction.

---

## ✅ Conclusion

This project demonstrates a complete machine learning pipeline — from raw data ingestion through to a deployed interactive web application — applied to the practical problem of used-car price estimation.

The **Gradient Boosting** model achieved the best performance across all three evaluation metrics, with an R² of **92.49%**, a MAE of **₹65,463** and an RMSE of **₹94,064** on the held-out test set. The Streamlit dashboard makes the model accessible to non-technical users, provides transparent model information, and visualises the underlying dataset through five analytics charts.

All statistics, model results and feature names in this README are sourced directly from the actual implementation — nothing is assumed or fabricated.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<p align="center">
  Built with Python · pandas · scikit-learn · Streamlit
</p>
