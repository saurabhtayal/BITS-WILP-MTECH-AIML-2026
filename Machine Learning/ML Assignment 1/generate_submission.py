import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Load datasets
df_train = pd.read_csv("bike_train.csv")
df_test = pd.read_csv("bike_test.csv")


def preprocess_and_engineer(df):
    """Advanced feature engineering"""
    df_copy = df.copy()

    # Parse datetime
    df_copy["datetime_dt"] = pd.to_datetime(
        df_copy["datetime"], format="mixed", dayfirst=False
    )

    # Extract temporal features
    df_copy["hour"] = df_copy["datetime_dt"].dt.hour
    df_copy["month"] = df_copy["datetime_dt"].dt.month
    df_copy["year"] = df_copy["datetime_dt"].dt.year - 2011
    df_copy["day_of_week"] = df_copy["datetime_dt"].dt.dayofweek
    df_copy["day"] = df_copy["datetime_dt"].dt.day

    # Cyclical encoding
    df_copy["hour_sin"] = np.sin(2 * np.pi * df_copy["hour"] / 24)
    df_copy["hour_cos"] = np.cos(2 * np.pi * df_copy["hour"] / 24)
    df_copy["month_sin"] = np.sin(2 * np.pi * df_copy["month"] / 12)
    df_copy["month_cos"] = np.cos(2 * np.pi * df_copy["month"] / 12)
    df_copy["dayofweek_sin"] = np.sin(2 * np.pi * df_copy["day_of_week"] / 7)
    df_copy["dayofweek_cos"] = np.cos(2 * np.pi * df_copy["day_of_week"] / 7)

    # Peak hour indicators
    df_copy["is_peak_hour"] = (
        ((df_copy["hour"] == 8) | (df_copy["hour"] == 17) | (df_copy["hour"] == 18))
        & (df_copy["workingday"] == 1)
    ).astype(int)

    df_copy["is_night"] = ((df_copy["hour"] >= 22) | (df_copy["hour"] <= 5)).astype(int)
    df_copy["is_afternoon"] = (
        (df_copy["hour"] >= 12) & (df_copy["hour"] <= 16)
    ).astype(int)

    # Weather interactions
    df_copy["temp_humidity"] = df_copy["temp"] * df_copy["humidity"]
    df_copy["temp_atemp"] = df_copy["temp"] * df_copy["atemp"]
    df_copy["temp_windspeed"] = df_copy["temp"] * df_copy["windspeed"]
    df_copy["humidity_windspeed"] = df_copy["humidity"] * df_copy["windspeed"]

    # Polynomial weather features
    df_copy["temp_sq"] = df_copy["temp"] ** 2
    df_copy["humidity_sq"] = df_copy["humidity"] ** 2
    df_copy["windspeed_sq"] = df_copy["windspeed"] ** 2

    df_copy = df_copy.drop(columns=["datetime_dt"])
    return df_copy


# Preprocess
df_train_eng = preprocess_and_engineer(df_train)
df_test_eng = preprocess_and_engineer(df_test)

# Prepare training data
# Drop columns that are not in test set
cols_to_drop = ["datetime", "count", "casual", "registered"]
X = df_train_eng.drop(columns=[c for c in cols_to_drop if c in df_train_eng.columns])
y = df_train_eng["count"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Define RMSLE
def get_rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


# Create polynomial features
weather_cols = ["temp", "humidity", "windspeed"]
poly = PolynomialFeatures(degree=2, include_bias=False)

poly_tr = poly.fit_transform(X_train[weather_cols])
poly_val = poly.transform(X_val[weather_cols])

poly_tr_df = pd.DataFrame(
    poly_tr, columns=poly.get_feature_names_out(weather_cols), index=X_train.index
)
poly_val_df = pd.DataFrame(
    poly_val, columns=poly.get_feature_names_out(weather_cols), index=X_val.index
)

# Get non-weather columns from X_train
non_weather_cols = [col for col in X_train.columns if col not in weather_cols]

X_train_poly = pd.concat([X_train[non_weather_cols], poly_tr_df], axis=1)
X_val_poly = pd.concat([X_val[non_weather_cols], poly_val_df], axis=1)

# Scale
scaler = StandardScaler()
X_train_poly_sc = scaler.fit_transform(X_train_poly)
X_val_poly_sc = scaler.transform(X_val_poly)

# Tune Lasso
print("Tuning Lasso Regression...")
best_lasso_alpha = None
best_lasso_val_rmsle = float("inf")

for alpha in [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
    lasso = Lasso(alpha=alpha, max_iter=10000, warm_start=False)
    lasso.fit(X_train_poly_sc, np.log1p(y_train))
    pred_val = np.clip(np.expm1(lasso.predict(X_val_poly_sc)), 0, None)
    val_rmsle = get_rmsle(y_val, pred_val)

    if val_rmsle < best_lasso_val_rmsle:
        best_lasso_val_rmsle = val_rmsle
        best_lasso_alpha = alpha

print(f"Best Lasso alpha: {best_lasso_alpha}, Val RMSLE: {best_lasso_val_rmsle:.6f}")

# Train final model
lasso_final = Lasso(alpha=best_lasso_alpha, max_iter=10000, warm_start=False)
lasso_final.fit(X_train_poly_sc, np.log1p(y_train))

# Prepare test data
X_test = df_test_eng.drop(columns=["datetime"], errors="ignore")

poly_test = poly.transform(X_test[weather_cols])
poly_test_df = pd.DataFrame(
    poly_test, columns=poly.get_feature_names_out(weather_cols), index=X_test.index
)
X_test_poly = pd.concat([X_test[non_weather_cols], poly_test_df], axis=1)

# Ensure columns match training set and in same order
X_test_poly = X_test_poly[X_train_poly.columns]
X_test_poly_sc = scaler.transform(X_test_poly)

# Make predictions
pred_test = np.clip(np.expm1(lasso_final.predict(X_test_poly_sc)), 0, None)

# Create submission
submission = pd.DataFrame(
    {"datetime": df_test["datetime"], "count_predicted": pred_test}
)

# Validate
print("\nSubmission validation:")
print(f"- Rows: {len(submission)} (expected: {len(df_test)})")
print(f"- Columns: {submission.columns.tolist()}")
print(f"- Missing values: {submission.isna().sum().sum()}")
print(f"- Negative predictions: {(submission['count_predicted'] < 0).sum()}")
print(
    f"- Prediction range: [{submission['count_predicted'].min():.2f}, {submission['count_predicted'].max():.2f}]"
)

# Save
submission.to_csv("submission.csv", index=False)
print("\n✓ submission.csv created successfully!")
print(f"✓ File size: {os.path.getsize('submission.csv') / 1024:.2f} KB")
print("✓ First 5 rows:")
print(submission.head())
