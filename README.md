# ML Assignment 1 — Bike Sharing Demand Prediction

M.Tech (AIML/DSE) Machine Learning course assignment. Predicts hourly bike rental demand using weather, time, and seasonal features.

## Repository Contents

| File | Description |
|------|-------------|
| `ML_Assignment_1_Bike_Sharing_Demand.ipynb` | Complete notebook with code, outputs, and answers to all 12 questions |
| `submission.csv` | Predictions on test data (datetime, count_predicted) |
| `data/bike_train.csv` | Training dataset (10,449 rows) |
| `data/bike_test.csv` | Test dataset (2,612 rows) |
| `data/sampleSubmission.csv` | Example submission format |
| `*.png` | EDA and model comparison plots |

## Approach

1. **EDA** — Analyzed distributions, missing values, correlations, and hourly demand patterns (Q1–Q3)
2. **Feature Engineering** — Hour dummies, cyclical encoding, rush hour indicators, polynomial/interaction terms (Q4)
3. **Models Built** (Q5–Q6):
   - Simple Linear Regression (baseline)
   - Linear Regression with engineered features
   - Polynomial Regression (degree 2)
   - Ridge Regression on polynomial features
   - Lasso Regression on polynomial features
   - Linear Regression with log-transformed target
   - **Ridge Regression + log target + engineered features (best)**
4. **Model Comparison & Interpretation** — RMSLE comparison table, residual analysis (Q7–Q9)
5. **Reflection** — RMSLE properties, bias-variance trade-off, time-of-day limitations of linear models (Q10–Q12)

## Results

| Model | Validation RMSLE |
|-------|------------------|
| Simple Linear Regression | ~1.38 |
| LR + Engineered Features | ~1.13 |
| Polynomial (deg 2) | ~0.99 |
| Ridge + Log Target + Eng Features | **~0.55** |

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook ML_Assignment_1_Bike_Sharing_Demand.ipynb
```

Run all cells sequentially. The notebook generates `submission.csv` and all plot files automatically.

## Evaluation Metric

RMSLE = sqrt( (1/n) * Σ (log(pred+1) - log(actual+1))² )
