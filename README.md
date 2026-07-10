
# Project 1: Advanced EDA & Feature Engineering

E-commerce order data pipeline covering missing-value imputation, outlier detection, 
and feature engineering — built as part of the DecodeLabs Data Science Industrial 
Training Kit (Batch 2026).

## 📌 Overview

This project transforms a raw, messy e-commerce orders dataset (1,200 records) into 
a clean, model-ready dataset through rigorous statistical methods — not just running 
algorithms on autopilot, but understanding *why* each cleaning decision was made.

## 📂 Repository Contents

| File | Description |
|---|---|
| `pipeline.py` | Full reproducible Python pipeline: EDA → imputation → outlier handling → feature engineering → correlation check |
| `orders_raw.csv` | Original raw dataset before any cleaning |
| `orders_cleaned.csv` | Cleaned dataset with engineered features, human-readable |
| `orders_cleaned_encoded.csv` | One-hot encoded, model-ready version |
| `Project1_EDA_Feature_Engineering_Report.docx` | Full write-up with findings, tables, and charts |

## 🛠 What Was Done

**1. Missing Data Treatment**
- Identified 25.75% missingness in `CouponCode` — treated as a genuine category (`NoCoupon`) rather than guessed, since a blank value structurally means no coupon was applied
- Benchmarked Mean vs. Median vs. KNN imputation on a simulated missing-value scenario — KNN outperformed both (MAE ≈ 95 vs. ≈ 182)

**2. Outlier Detection**
- Applied both IQR (1.5× bounds) and Z-score (|z| > 3) methods across all numeric columns
- Found and neutralized 8 outliers in `TotalPrice` via winsorization (`numpy.clip`) instead of deleting rows

**3. Feature Engineering**
- `OrderMonth`, `OrderYear`, `OrderDayOfWeek`, `IsWeekendOrder` — temporal features
- `CartConversionRate` — purchase intent proxy (Quantity / ItemsInCart)
- `RevenuePerCartItem` — basket value density
- `HasCoupon`, `DiscountedRevenueFlag` — discount/margin signals
- `OrderValueTier` — quartile-based order value segment
- `IsFailedOrder` — binary flag for cancelled/returned orders

**4. Multicollinearity Check**
- Correlation matrix across numeric + engineered features — no predictor pairs exceeded the 0.80 collinearity threshold

## 🧰 Tech Stack
Python · Pandas · NumPy · scikit-learn · SciPy · Matplotlib

## ▶️ How to Run

```bash
pip install pandas numpy scipy scikit-learn matplotlib
python pipeline.py
