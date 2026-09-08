import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Set plotting style
sns.set_theme(style="whitegrid")

# Define local paths
local_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(local_dir, "output")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load datasets - Update these paths if your CSV files are in a different location
df_train = pd.read_csv(os.path.join(local_dir, "bike_train.csv"))
df_test = pd.read_csv(os.path.join(local_dir, "bike_test.csv"))


# Preprocessing / Feature Engineering function
def preprocess_and_engineer(df):
    df_copy = df.copy()
    # Parse datetime strings - use format='mixed' to handle both ISO and DD-MM-YYYY formats
    df_copy["datetime_dt"] = pd.to_datetime(
        df_copy["datetime"], format="mixed", dayfirst=False
    )
    df_copy["hour"] = df_copy["datetime_dt"].dt.hour
    df_copy["month"] = df_copy["datetime_dt"].dt.month
    df_copy["year"] = df_copy["datetime_dt"].dt.year - 2011  # 0 for 2011, 1 for 2012
    df_copy["day_of_week"] = df_copy["datetime_dt"].dt.dayofweek
    df_copy["temp_humidity"] = df_copy["temp"] * df_copy["humidity"]
    df_copy["is_peak_hour"] = (
        ((df_copy["hour"] == 8) | (df_copy["hour"] == 17) | (df_copy["hour"] == 18))
        & (df_copy["workingday"] == 1)
    ).astype(int)
    df_copy = df_copy.drop(columns=["datetime_dt"])
    return df_copy


df_train_eng = preprocess_and_engineer(df_train)
df_test_eng = preprocess_and_engineer(df_test)

# --- Generate EDA Plots ---
print("Generating EDA plots...")

# 1. Hourly profile
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=df_train_eng,
    x="hour",
    y="count",
    hue="workingday",
    palette={0: "royalblue", 1: "crimson"},
    marker="o",
)
plt.title(
    "Hourly Bike Rental Demand: Working Day vs. Weekend/Holiday",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Hour of the Day", fontsize=12)
plt.ylabel("Average Rental Count", fontsize=12)
plt.legend(title="Day Type", labels=["Weekend/Holiday", "Working Day"])
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "plot_hourly_profile.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# 2. Temperature vs. count
plt.figure(figsize=(10, 5))
sns.regplot(
    data=df_train_eng,
    x="temp",
    y="count",
    scatter_kws={"alpha": 0.3, "color": "darkorange"},
    line_kws={"color": "darkblue", "linewidth": 2},
    order=2,
)
plt.title(
    "Influence of Temperature on Rental Demand (Quadratic Fit)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Temperature (°C)", fontsize=12)
plt.ylabel("Rental Count", fontsize=12)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "plot_temp_vs_demand.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# 3. Humidity vs. count
plt.figure(figsize=(10, 5))
sns.regplot(
    data=df_train_eng,
    x="humidity",
    y="count",
    scatter_kws={"alpha": 0.3, "color": "mediumseagreen"},
    line_kws={"color": "purple", "linewidth": 2},
)
plt.title(
    "Influence of Relative Humidity on Rental Demand", fontsize=14, fontweight="bold"
)
plt.xlabel("Relative Humidity (%)", fontsize=12)
plt.ylabel("Rental Count", fontsize=12)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "plot_humidity_vs_demand.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# 4. Season and Weather combinations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.barplot(
    data=df_train_eng,
    x="season",
    y="count",
    ax=axes[0],
    palette="pastel",
    errorbar=None,
)
axes[0].set_title("Average Demand by Season", fontsize=12, fontweight="bold")
axes[0].set_xticklabels(["Spring", "Summer", "Fall", "Winter"])
axes[0].set_xlabel("Season")
axes[0].set_ylabel("Average Count")

sns.barplot(
    data=df_train_eng, x="weather", y="count", ax=axes[1], palette="vlag", errorbar=None
)
axes[1].set_title("Average Demand by Weather Type", fontsize=12, fontweight="bold")
axes[1].set_xticklabels(["Clear", "Misty/Cloudy", "Rain/Snow"])
axes[1].set_xlabel("Weather Category")
axes[1].set_ylabel("Average Count")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "plot_season_weather.png"), dpi=150, bbox_inches="tight"
)
plt.close()

# --- Regression Modeling ---
X = df_train_eng.drop(columns=["datetime", "count"])
y = df_train_eng["count"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def get_rmsle(y_true, y_pred):
    return np.sqrt(
        np.mean((np.log1p(np.clip(y_pred, 0, None)) - np.log1p(y_true)) ** 2)
    )


# 1. Baseline Model (temp, humidity, windspeed)
base_cols = ["temp", "humidity", "windspeed"]
lr_base = LinearRegression()
lr_base.fit(X_train[base_cols], y_train)
pred_tr_base = lr_base.predict(X_train[base_cols])
pred_val_base = lr_base.predict(X_val[base_cols])
r_tr_b = get_rmsle(y_train, pred_tr_base)
r_va_b = get_rmsle(y_val, pred_val_base)

# 2. Engineered OLS (Log target)
lr_eng = LinearRegression()
lr_eng.fit(X_train, np.log1p(y_train))
pred_tr_eng = np.expm1(lr_eng.predict(X_train))
pred_val_eng = np.expm1(lr_eng.predict(X_val))
r_tr_e = get_rmsle(y_train, pred_tr_eng)
r_va_e = get_rmsle(y_val, pred_val_eng)

# 3. Polynomial Expansion (Degree 2) + Scaling
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_tr = poly.fit_transform(X_train[base_cols])
poly_val = poly.transform(X_val[base_cols])
poly_tr_df = pd.DataFrame(
    poly_tr, columns=poly.get_feature_names_out(base_cols), index=X_train.index
)
poly_val_df = pd.DataFrame(
    poly_val, columns=poly.get_feature_names_out(base_cols), index=X_val.index
)
X_train_poly = pd.concat([X_train.drop(columns=base_cols), poly_tr_df], axis=1)
X_val_poly = pd.concat([X_val.drop(columns=base_cols), poly_val_df], axis=1)

scaler = StandardScaler()
X_train_poly_sc = scaler.fit_transform(X_train_poly)
X_val_poly_sc = scaler.transform(X_val_poly)

lr_poly = LinearRegression()
lr_poly.fit(X_train_poly_sc, np.log1p(y_train))
pred_tr_poly = np.expm1(lr_poly.predict(X_train_poly_sc))
pred_val_poly = np.expm1(lr_poly.predict(X_val_poly_sc))
r_tr_p = get_rmsle(y_train, pred_tr_poly)
r_va_p = get_rmsle(y_val, pred_val_poly)

# 4. Ridge Regression
r_model = Ridge(alpha=1.0)
r_model.fit(X_train_poly_sc, np.log1p(y_train))
pred_tr_ridge = np.expm1(r_model.predict(X_train_poly_sc))
pred_val_ridge = np.expm1(r_model.predict(X_val_poly_sc))
r_tr_r = get_rmsle(y_train, pred_tr_ridge)
r_va_r = get_rmsle(y_val, pred_val_ridge)

# 5. Lasso Regression
l_model = Lasso(alpha=0.001, max_iter=10000)
l_model.fit(X_train_poly_sc, np.log1p(y_train))
pred_tr_lasso = np.expm1(l_model.predict(X_train_poly_sc))
pred_val_lasso = np.expm1(l_model.predict(X_val_poly_sc))
r_tr_l = get_rmsle(y_train, pred_tr_lasso)
r_va_l = get_rmsle(y_val, pred_val_lasso)

# Compile results
results = [
    {
        "Model": "Baseline OLS Regression",
        "Features": "temp, humidity, windspeed",
        "Train RMSLE": r_tr_b,
        "Val RMSLE": r_va_b,
        "Observations": "High bias. Cannot capture hour or season effects.",
    },
    {
        "Model": "Engineered OLS (Log target)",
        "Features": "All features including hour, year, peak-hour indicator",
        "Train RMSLE": r_tr_e,
        "Val RMSLE": r_va_e,
        "Observations": "Dramatic performance boost. Captures time-of-day and growth trends.",
    },
    {
        "Model": "Polynomial OLS (Degree 2, Log target)",
        "Features": "Degree 2 weather features + time features",
        "Train RMSLE": r_tr_p,
        "Val RMSLE": r_va_p,
        "Observations": "Captures temperature curvature well, further reducing error.",
    },
    {
        "Model": "Ridge Regression (alpha=1.0)",
        "Features": "Scaled Polynomial features + time features",
        "Train RMSLE": r_tr_r,
        "Val RMSLE": r_va_r,
        "Observations": "Regularization controls model complexity. High stability.",
    },
    {
        "Model": "Lasso Regression (alpha=0.001)",
        "Features": "Scaled Polynomial features + time features",
        "Train RMSLE": r_tr_l,
        "Val RMSLE": r_va_l,
        "Observations": "Performs L1 penalty, zeroing out uninformative polynomial terms.",
    },
]
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)

print(f"Analysis complete. Results stored in {output_dir}")

# --- Residual Analysis Plots for Lasso Model ---
print("Generating residual plots...")
residuals = y_val - pred_val_lasso

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual vs predicted
sns.scatterplot(
    x=pred_val_lasso, y=residuals, ax=axes[0], alpha=0.5, color="darkviolet"
)
axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
axes[0].set_title(
    "Residuals vs. Predicted Values (Lasso Model)", fontsize=12, fontweight="bold"
)
axes[0].set_xlabel("Predicted Counts")
axes[0].set_ylabel("Residuals (Actual - Predicted)")

# Residual distribution
sns.histplot(residuals, kde=True, ax=axes[1], color="teal")
axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2)
axes[1].set_title("Distribution of Error Residuals", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Residual Value")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "plot_residuals.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print("Residual plots saved successfully.")
