"""
Baseline Model Comparison for Fashion-Economy Forecasting (FULLY DEBUGGED)
Tests whether fashion improves economic forecasts beyond standard baselines
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed_features", "fashion_economy_aligned.csv"
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# DATA PREPARATION
# =============================================================================


def load_and_prepare_data():
    """Load data and check stationarity"""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found:\n{DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Keep only required columns and drop NA
    required_cols = ["date", "vibrancy", "VIX", "SP500"]
    available_cols = [c for c in required_cols if c in df.columns]

    if "SP500" not in df.columns and "SP500_Return" in df.columns:
        print("⚠️  SP500 not found, using SP500_Return instead")
        available_cols.append("SP500_Return")
        df = df[available_cols].dropna()
    else:
        df = df[available_cols].dropna()

    print(f"✅ Loaded {len(df)} months")
    print(f"   Date range: {df['date'].min():%Y-%m} → {df['date'].max():%Y-%m}")

    # Stationarity test
    adf_stat, adf_p = adfuller(df["VIX"])[:2]
    print(f"\n📊 ADF test for VIX: stat={adf_stat:.3f}, p={adf_p:.3f}")

    use_diff = adf_p > 0.05
    if use_diff:
        print("⚠️  VIX is non-stationary → using first differences")
    else:
        print("✅ VIX is stationary → using levels")

    return df, use_diff


def train_test_split(df, train_ratio=0.75):
    """Split into train/test sets"""
    n = len(df)
    split_idx = int(n * train_ratio)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT")
    print("=" * 70)
    print(
        f"Train: {train['date'].min():%Y-%m} to {train['date'].max():%Y-%m} (n={len(train)})"
    )
    print(
        f"Test:  {test['date'].min():%Y-%m} to {test['date'].max():%Y-%m} (n={len(test)})"
    )

    return train, test


# =============================================================================
# MODEL 1: NAIVE (LAST VALUE)
# =============================================================================


def model1_naive(train, test):
    """Naive forecast: predict last observed value"""
    print("\n" + "=" * 70)
    print("MODEL 1: NAIVE (Last Value)")
    print("=" * 70)

    last_vix = train["VIX"].iloc[-1]
    predictions = np.full(len(test), last_vix)

    rmse = np.sqrt(mean_squared_error(test["VIX"], predictions))
    mae = mean_absolute_error(test["VIX"], predictions)

    print(f"   Forecast: Always predict {last_vix:.2f}")
    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAE  = {mae:.2f}")

    return {
        "model": "Naive",
        "predictions": predictions,
        "rmse": rmse,
        "mae": mae,
    }


# =============================================================================
# MODEL 2: AR(1)
# =============================================================================


def model2_ar1(train, test):
    """AR(1) model - univariate autoregression"""
    print("\n" + "=" * 70)
    print("MODEL 2: AR(1)")
    print("=" * 70)

    predictions = []
    history = train["VIX"].tolist()

    for i, actual in enumerate(test["VIX"]):
        try:
            model = ARIMA(history, order=(1, 0, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)[0]
            predictions.append(forecast)
        except Exception as e:
            # Fallback to last value if ARIMA fails
            print(f"   Warning at step {i}: {e}")
            predictions.append(history[-1])

        history.append(actual)  # Add actual value for next iteration

    predictions = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(test["VIX"], predictions))
    mae = mean_absolute_error(test["VIX"], predictions)

    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAE  = {mae:.2f}")

    return {
        "model": "AR(1)",
        "predictions": predictions,
        "rmse": rmse,
        "mae": mae,
    }


# =============================================================================
# MODEL 3: VAR WITHOUT FASHION (VIX + SP500)
# =============================================================================


def model3_var_no_fashion(train, test, use_diff):
    """VAR model with VIX and SP500 only (no fashion)"""
    print("\n" + "=" * 70)
    print("MODEL 3: VAR (VIX + SP500, No Fashion)")
    print("=" * 70)

    # Check if SP500 is available
    if "SP500" not in train.columns:
        print("   ⚠️  SP500 not available, skipping this model")
        return {
            "model": "VAR (No Fashion)",
            "predictions": None,
            "rmse": np.inf,
            "mae": np.inf,
        }

    cols = ["VIX", "SP500"]

    # Prepare training data
    if use_diff:
        train_data = train[cols].diff().dropna()
        print("   Using first differences")
    else:
        train_data = train[cols].copy()
        print("   Using levels")

    predictions = []
    history = train_data.copy()

    for i in range(len(test)):
        try:
            # Fit VAR model
            model = VAR(history)
            fitted = model.fit(maxlags=min(4, len(history) // 10), ic="aic")

            # CRITICAL FIX: Handle k_ar = 0 case
            if fitted.k_ar == 0:
                # No dynamics detected, use last value
                if use_diff:
                    vix_pred_diff = 0.0
                    if i == 0:
                        vix_pred_level = train["VIX"].iloc[-1] + vix_pred_diff
                    else:
                        vix_pred_level = predictions[-1] + vix_pred_diff
                else:
                    vix_pred_level = history["VIX"].iloc[-1]
            else:
                # Normal forecast
                last_obs = history.values[-fitted.k_ar :]
                forecast = fitted.forecast(last_obs, steps=1)
                vix_pred_value = forecast[0, 0]  # VIX is first column

                if use_diff:
                    # Convert diff to level
                    if i == 0:
                        vix_pred_level = train["VIX"].iloc[-1] + vix_pred_value
                    else:
                        vix_pred_level = predictions[-1] + vix_pred_value
                else:
                    # Already in levels
                    vix_pred_level = vix_pred_value

            predictions.append(vix_pred_level)

            # Add actual observation to history
            if use_diff:
                if i == 0:
                    actual_vix_diff = test["VIX"].iloc[i] - train["VIX"].iloc[-1]
                    actual_sp_diff = test["SP500"].iloc[i] - train["SP500"].iloc[-1]
                else:
                    actual_vix_diff = test["VIX"].iloc[i] - test["VIX"].iloc[i - 1]
                    actual_sp_diff = test["SP500"].iloc[i] - test["SP500"].iloc[i - 1]

                new_row = pd.DataFrame(
                    [[actual_vix_diff, actual_sp_diff]], columns=cols
                )
            else:
                new_row = test[cols].iloc[[i]]

            history = pd.concat([history, new_row], ignore_index=True)

        except Exception as e:
            print(f"   Warning at step {i}: {e}")
            # Fallback to last prediction or naive
            if predictions:
                predictions.append(predictions[-1])
            else:
                predictions.append(train["VIX"].iloc[-1])

    predictions = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(test["VIX"], predictions))
    mae = mean_absolute_error(test["VIX"], predictions)

    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAE  = {mae:.2f}")

    return {
        "model": "VAR (No Fashion)",
        "predictions": predictions,
        "rmse": rmse,
        "mae": mae,
    }


# =============================================================================
# MODEL 4: VAR WITH FASHION (VIX + VIBRANCY)
# =============================================================================


def model4_var_with_fashion(train, test, use_diff):
    """VAR model including fashion vibrancy"""
    print("\n" + "=" * 70)
    print("MODEL 4: VAR (VIX + Fashion Vibrancy)")
    print("=" * 70)

    cols = ["VIX", "vibrancy"]

    # Prepare training data
    if use_diff:
        train_data = train[cols].diff().dropna()
        print("   Using first differences")
    else:
        train_data = train[cols].copy()
        print("   Using levels")

    predictions = []
    history = train_data.copy()

    for i in range(len(test)):
        try:
            # Fit VAR model
            model = VAR(history)
            fitted = model.fit(maxlags=min(4, len(history) // 10), ic="aic")

            # CRITICAL FIX: Handle k_ar = 0 case
            if fitted.k_ar == 0:
                # No dynamics detected
                if use_diff:
                    vix_pred_diff = 0.0
                    if i == 0:
                        vix_pred_level = train["VIX"].iloc[-1] + vix_pred_diff
                    else:
                        vix_pred_level = predictions[-1] + vix_pred_diff
                else:
                    vix_pred_level = history["VIX"].iloc[-1]
            else:
                # Normal forecast
                last_obs = history.values[-fitted.k_ar :]
                forecast = fitted.forecast(last_obs, steps=1)
                vix_pred_value = forecast[0, 0]  # VIX is first column

                if use_diff:
                    # Convert diff to level
                    if i == 0:
                        vix_pred_level = train["VIX"].iloc[-1] + vix_pred_value
                    else:
                        vix_pred_level = predictions[-1] + vix_pred_value
                else:
                    # Already in levels
                    vix_pred_level = vix_pred_value

            predictions.append(vix_pred_level)

            # Add actual observation to history
            if use_diff:
                if i == 0:
                    actual_vix_diff = test["VIX"].iloc[i] - train["VIX"].iloc[-1]
                    actual_vib_diff = (
                        test["vibrancy"].iloc[i] - train["vibrancy"].iloc[-1]
                    )
                else:
                    actual_vix_diff = test["VIX"].iloc[i] - test["VIX"].iloc[i - 1]
                    actual_vib_diff = (
                        test["vibrancy"].iloc[i] - test["vibrancy"].iloc[i - 1]
                    )

                new_row = pd.DataFrame(
                    [[actual_vix_diff, actual_vib_diff]], columns=cols
                )
            else:
                new_row = test[cols].iloc[[i]]

            history = pd.concat([history, new_row], ignore_index=True)

        except Exception as e:
            print(f"   Warning at step {i}: {e}")
            # Fallback
            if predictions:
                predictions.append(predictions[-1])
            else:
                predictions.append(train["VIX"].iloc[-1])

    predictions = np.array(predictions)

    rmse = np.sqrt(mean_squared_error(test["VIX"], predictions))
    mae = mean_absolute_error(test["VIX"], predictions)

    print(f"   RMSE = {rmse:.2f}")
    print(f"   MAE  = {mae:.2f}")

    return {
        "model": "VAR (Fashion)",
        "predictions": predictions,
        "rmse": rmse,
        "mae": mae,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_forecasts(test, results):
    """Create forecast comparison plot"""

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: Time series comparison
    ax1.plot(
        test["date"],
        test["VIX"],
        label="Actual VIX",
        linewidth=2.5,
        color="black",
        marker="o",
        markersize=5,
    )

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    markers = ["x", "s", "^", "d"]

    for i, r in enumerate(results):
        if r["predictions"] is not None and not np.isinf(r["rmse"]):
            ax1.plot(
                test["date"],
                r["predictions"],
                linestyle="--",
                marker=markers[i],
                markersize=4,
                alpha=0.7,
                color=colors[i],
                linewidth=1.5,
                label=f"{r['model']} (RMSE={r['rmse']:.2f})",
            )

    ax1.set_title(
        "Out-of-Sample VIX Forecasts: All Models", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("VIX", fontweight="bold")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Panel 2: Forecast errors
    for i, r in enumerate(results):
        if r["predictions"] is not None and not np.isinf(r["rmse"]):
            errors = test["VIX"].values - r["predictions"]
            ax2.plot(
                test["date"],
                errors,
                label=f"{r['model']}",
                marker=markers[i],
                markersize=4,
                alpha=0.7,
                color=colors[i],
                linewidth=1.5,
            )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Forecast Errors", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date", fontweight="bold")
    ax2.set_ylabel("Error (Actual - Predicted)", fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, "forecast_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Visualization saved: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def compare_all_models():
    """Run all baseline models and compare results"""

    print("=" * 70)
    print("BASELINE MODEL COMPARISON FOR VIX FORECASTING")
    print("=" * 70)
    print("Question: Does fashion improve VIX forecasts?")
    print("=" * 70)

    # Load data
    df, use_diff = load_and_prepare_data()
    train, test = train_test_split(df)

    # Run all models
    results = [
        model1_naive(train, test),
        model2_ar1(train, test),
        model3_var_no_fashion(train, test, use_diff),
        model4_var_with_fashion(train, test, use_diff),
    ]

    # Filter out models that failed
    valid_results = [
        r for r in results if r["predictions"] is not None and not np.isinf(r["rmse"])
    ]

    # Create comparison table
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    comparison = pd.DataFrame(valid_results)[["model", "rmse", "mae"]]
    comparison = comparison.sort_values("rmse").reset_index(drop=True)
    comparison.insert(0, "Rank", comparison.index + 1)

    print("\n" + comparison.to_string(index=False))

    # Determine if fashion helps
    fashion_models = comparison[comparison["model"] == "VAR (Fashion)"]
    no_fashion_models = comparison[comparison["model"] == "VAR (No Fashion)"]

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if len(fashion_models) > 0 and len(no_fashion_models) > 0:
        fashion_rmse = fashion_models["rmse"].values[0]
        no_fashion_rmse = no_fashion_models["rmse"].values[0]
        improvement = ((no_fashion_rmse - fashion_rmse) / no_fashion_rmse) * 100

        if improvement > 5:
            print(f"✅ Fashion IMPROVES forecasts by {improvement:.1f}%")
            print(f"   RMSE: {no_fashion_rmse:.2f} → {fashion_rmse:.2f}")
        elif improvement > -5:
            print(f"≈ Fashion has MINIMAL impact ({improvement:+.1f}%)")
            print(f"   RMSE: {no_fashion_rmse:.2f} vs {fashion_rmse:.2f}")
        else:
            print(f"❌ Fashion WORSENS forecasts by {abs(improvement):.1f}%")
            print(f"   RMSE: {no_fashion_rmse:.2f} → {fashion_rmse:.2f}")
    else:
        print("⚠️  Unable to compare VAR models (one or both failed)")

    # Best overall model
    if len(valid_results) > 0:
        best_model = comparison.iloc[0]
        print(f"\n🏆 Best model: {best_model['model']}")
        print(f"   RMSE: {best_model['rmse']:.2f}")
        print(f"   MAE:  {best_model['mae']:.2f}")

    # Save results
    comparison.to_csv(os.path.join(RESULTS_DIR, "baseline_comparison.csv"), index=False)
    print(f"\n✅ Results saved: {RESULTS_DIR}/baseline_comparison.csv")

    # Create visualization
    plot_forecasts(test, results)

    return comparison


if __name__ == "__main__":
    compare_all_models()
