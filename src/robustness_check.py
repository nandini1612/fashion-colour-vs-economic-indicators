"""
Robustness Checks for Fashion-Economy Analysis (FIXED VERSION)
Run this after main econometric analysis to validate findings
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mstats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# PATH SETUP (ROBUST)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# HELPER FUNCTION: GRANGER CAUSALITY
# =============================================================================


def run_granger_test(df, predictor, target, max_lag=8, verbose=False):
    """
    Run Granger causality test on differenced series
    Returns: dict or None
    """

    df = df.sort_values("date").copy()

    df[f"{predictor}_diff"] = df[predictor].diff()
    df[f"{target}_diff"] = df[target].diff()

    df_clean = df[[f"{predictor}_diff", f"{target}_diff"]].dropna()

    if len(df_clean) < max_lag + 10:
        print(f"   ⚠️ Insufficient data: {len(df_clean)} observations")
        return None

    data = df_clean[[f"{target}_diff", f"{predictor}_diff"]].values

    try:
        results = grangercausalitytests(data, maxlag=max_lag, verbose=False)

        lag_results = []
        for lag in range(1, max_lag + 1):
            f_p = results[lag][0]["ssr_ftest"][1]
            chi2_p = results[lag][0]["ssr_chi2test"][1]
            lag_results.append({"lag": lag, "f_p": f_p, "chi2_p": chi2_p})

        df_lags = pd.DataFrame(lag_results)
        best = df_lags.loc[df_lags["f_p"].idxmin()]

        if verbose:
            print(f"   Best lag: {best['lag']}, p={best['f_p']:.3f}")

        return {
            "predictor": predictor,
            "target": target,
            "best_lag": int(best["lag"]),
            "f_p": float(best["f_p"]),
            "chi2_p": float(best["chi2_p"]),
            "significant": "Yes" if best["f_p"] < 0.05 else "No",
            "n_obs": len(df_clean),
        }

    except Exception as e:
        print(f"   ❌ Granger test failed: {e}")
        return None


# =============================================================================
# CHECK 1: EXCLUDE COVID PERIOD
# =============================================================================


def check1_exclude_covid(df):
    print("\n" + "=" * 70)
    print("CHECK 1: EXCLUDE COVID PERIOD (Mar–Jun 2020)")
    print("=" * 70)

    covid_mask = (df["date"] >= "2020-03-01") & (df["date"] <= "2020-06-30")
    df_nc = df.loc[~covid_mask].copy()

    print(f"Original n: {len(df)}")
    print(f"Excluding COVID: {len(df_nc)}")

    # Test both VIX and SP500
    result_vix = run_granger_test(df_nc, "vibrancy", "VIX", verbose=True)
    result_sp = run_granger_test(df_nc, "vibrancy", "SP500", verbose=True)

    if result_sp is None:
        result_sp = {"f_p": 999, "significant": "N/A"}

    corr_vix = df_nc[["vibrancy", "VIX"]].corr().iloc[0, 1]
    corr_sp = df_nc[["vibrancy", "SP500"]].corr().iloc[0, 1]

    print("\n📊 Results:")
    print(
        f"   VIX: p={result_vix['f_p'] if result_vix else 'N/A':.3f}, r={corr_vix:.3f}"
    )
    print(f"   SP500: p={result_sp['f_p']:.3f}, r={corr_sp:.3f}")

    return {
        "check": "Exclude COVID",
        "p_value": result_sp["f_p"] if result_sp else 999,
        "correlation": corr_sp,
        "significant": result_sp["significant"] if result_sp else "N/A",
    }


# =============================================================================
# CHECK 2: ALTERNATIVE METRICS
# =============================================================================


def check2_alternative_metrics(df):
    print("\n" + "=" * 70)
    print("CHECK 2: ALTERNATIVE FASHION METRICS")
    print("=" * 70)

    metrics = ["darkness", "weighted_vibrancy", "brightness"]
    results = []

    for m in metrics:
        if m not in df.columns:
            print(f"⚠️ Skipping {m} (not in dataset)")
            continue

        print(f"\nTesting {m} → SP500...")
        r = run_granger_test(df, m, "SP500", verbose=True)
        if r:
            corr = df[[m, "SP500"]].corr().iloc[0, 1]
            results.append(
                {
                    "metric": m,
                    "p_value": r["f_p"],
                    "correlation": corr,
                    "best_lag": r["best_lag"],
                    "significant": r["significant"],
                }
            )

    return (
        pd.DataFrame(results)
        if results
        else pd.DataFrame(
            columns=["metric", "p_value", "correlation", "best_lag", "significant"]
        )
    )


# =============================================================================
# CHECK 3: MULTIPLE TARGETS
# =============================================================================


def check3_multiple_targets(df):
    print("\n" + "=" * 70)
    print("CHECK 3: MULTIPLE ECONOMIC TARGETS")
    print("=" * 70)

    # Test both differenced and level targets
    targets = []
    if "SP500" in df.columns:
        targets.append("SP500")
    if "SP500_Return" in df.columns:
        targets.append("SP500_Return")
    if "Unemployment" in df.columns:
        targets.append("Unemployment")

    results = []

    for t in targets:
        print(f"\nTesting vibrancy → {t}")
        r = run_granger_test(df, "vibrancy", t, verbose=True)
        if r:
            corr = df[["vibrancy", t]].corr().iloc[0, 1]
            results.append(
                {
                    "target": t,
                    "p_value": r["f_p"],
                    "correlation": corr,
                    "best_lag": r["best_lag"],
                    "significant": r["significant"],
                }
            )

    return (
        pd.DataFrame(results)
        if results
        else pd.DataFrame(
            columns=["target", "p_value", "correlation", "best_lag", "significant"]
        )
    )


# =============================================================================
# CHECK 4: SUBPERIOD STABILITY
# =============================================================================


def check4_subperiod_stability(df):
    print("\n" + "=" * 70)
    print("CHECK 4: SUBPERIOD STABILITY")
    print("=" * 70)

    periods = [
        ("2015-08-01", "2019-12-31", "Pre-COVID"),
        ("2020-03-01", "2021-12-31", "COVID Era"),
        ("2022-01-01", "2024-07-31", "Post-COVID"),
    ]

    rows = []

    for start, end, label in periods:
        sub = df[(df["date"] >= start) & (df["date"] <= end)]
        if len(sub) < 6:
            continue

        # Test vibrancy → SP500 for each period
        r_granger = run_granger_test(
            sub, "vibrancy", "SP500", max_lag=min(4, len(sub) // 8), verbose=False
        )
        r_corr = sub[["vibrancy", "SP500"]].corr().iloc[0, 1]
        n = len(sub)

        # Correlation significance test
        if n >= 3:
            t = r_corr * np.sqrt(n - 2) / np.sqrt(1 - r_corr**2)
            p_corr = 2 * (1 - stats.t.cdf(abs(t), n - 2))
        else:
            p_corr = 1.0

        rows.append(
            {
                "period": label,
                "start": start,
                "end": end,
                "n_months": n,
                "correlation": r_corr,
                "corr_p_value": p_corr,
                "granger_p": r_granger["f_p"] if r_granger else None,
                "granger_sig": r_granger["significant"] if r_granger else "N/A",
            }
        )

        print(
            f"{label}: n={n}, r={r_corr:.3f}, Granger p={r_granger['f_p'] if r_granger else 'N/A'}"
        )

    return pd.DataFrame(rows)


# =============================================================================
# CHECK 5: SEASONALITY CONTROL
# =============================================================================


def check5_seasonality_control(df):
    print("\n" + "=" * 70)
    print("CHECK 5: SEASONALITY CONTROL")
    print("=" * 70)

    df = df.copy()
    df["season"] = df["date"].dt.month.map(
        lambda m: "Fall" if m in [9, 10, 11, 12, 1, 2] else "Spring"
    )

    X = pd.get_dummies(df["season"], drop_first=True)
    y = df["vibrancy"]

    model = LinearRegression().fit(X, y)
    df["vibrancy_deseas"] = y - model.predict(X)

    r = run_granger_test(df, "vibrancy_deseas", "SP500", verbose=True)
    if r is None:
        return None

    return {
        "check": "Deseasonalized",
        "p_value": r["f_p"],
        "significant": r["significant"],
    }


# =============================================================================
# CHECK 6: OUTLIER SENSITIVITY
# =============================================================================


def check6_outlier_sensitivity(df):
    print("\n" + "=" * 70)
    print("CHECK 6: OUTLIER SENSITIVITY (Winsorization)")
    print("=" * 70)

    df = df.copy()
    df["vibrancy_w"] = pd.Series(
        mstats.winsorize(df["vibrancy"], limits=[0.05, 0.05]), index=df.index
    )
    df["SP500_w"] = pd.Series(
        mstats.winsorize(df["SP500"], limits=[0.05, 0.05]), index=df.index
    )

    r = run_granger_test(df, "vibrancy_w", "SP500_w", verbose=True)
    if r is None:
        return None

    return {
        "check": "Winsorized",
        "p_value": r["f_p"],
        "significant": r["significant"],
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def run_all_robustness_checks():
    print("=" * 70)
    print("ROBUSTNESS CHECKS FOR FASHION–ECONOMY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(
        os.path.join(
            PROJECT_ROOT, "data/processed_features/fashion_economy_aligned.csv"
        )
    )
    df["date"] = pd.to_datetime(df["date"])

    print(f"\nLoaded {len(df)} months of data")
    print(f"Date range: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")

    # Initialize results with baseline from your actual econometric analysis
    results = [
        {
            "Check": "0. Baseline (Full Sample)",
            "P-value": 0.0186,  # FIXED: Your actual vibrancy→SP500 result
            "Correlation": -0.022,
            "Significant": "Yes",
            "Notes": "vibrancy → SP500 at lag 7",
        }
    ]

    # Run all checks
    c1 = check1_exclude_covid(df)
    if c1:
        results.append(
            {
                "Check": "1. Exclude COVID",
                "P-value": c1["p_value"],
                "Correlation": c1["correlation"],
                "Significant": c1["significant"],
                "Notes": "Remove Mar–Jun 2020",
            }
        )

    c2 = check2_alternative_metrics(df)
    if len(c2) > 0:
        results.append(
            {
                "Check": "2. Alt Metrics (avg)",
                "P-value": c2["p_value"].mean(),
                "Correlation": c2["correlation"].mean(),
                "Significant": f"{(c2['significant'] == 'Yes').sum()}/{len(c2)} sig",
                "Notes": "Darkness, brightness, weighted",
            }
        )

    c3 = check3_multiple_targets(df)
    if len(c3) > 0:
        for _, row in c3.iterrows():
            results.append(
                {
                    "Check": f"3. Target: {row['target']}",
                    "P-value": row["p_value"],
                    "Correlation": row["correlation"],
                    "Significant": row["significant"],
                    "Notes": f"vibrancy → {row['target']}",
                }
            )

    c4 = check4_subperiod_stability(df)

    c5 = check5_seasonality_control(df)
    if c5:
        results.append(
            {
                "Check": "5. Deseasonalized",
                "P-value": c5["p_value"],
                "Correlation": None,
                "Significant": c5["significant"],
                "Notes": "Season control",
            }
        )

    c6 = check6_outlier_sensitivity(df)
    if c6:
        results.append(
            {
                "Check": "6. Winsorized",
                "P-value": c6["p_value"],
                "Correlation": None,
                "Significant": c6["significant"],
                "Notes": "5/95 winsorization",
            }
        )

    # Create summary
    summary = pd.DataFrame(results)

    # Interpret
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    robustness_only = summary[summary["Check"] != "0. Baseline (Full Sample)"]
    sig_count = robustness_only["Significant"].str.contains("Yes", na=False).sum()
    total = len(robustness_only)

    print("\n" + "=" * 70)
    print("OVERALL INTERPRETATION")
    print("=" * 70)
    print(f"Baseline finding: vibrancy → SP500 (p=0.0186) ✅ SIGNIFICANT")
    print(f"Robustness checks: {sig_count}/{total} confirm the relationship")

    if sig_count >= total * 0.5:
        print("\n✅ ROBUST: Finding holds across multiple specifications")
    else:
        print("\n⚠️  FRAGILE: Finding sensitive to specification")

    # Save results
    summary.to_csv(os.path.join(RESULTS_DIR, "robustness_summary.csv"), index=False)

    if len(c2) > 0:
        c2.to_csv(
            os.path.join(RESULTS_DIR, "alternative_metrics_detail.csv"), index=False
        )

    if len(c4) > 0:
        c4.to_csv(os.path.join(RESULTS_DIR, "subperiod_analysis.csv"), index=False)

    print(f"\n✅ Results saved to {RESULTS_DIR}")

    return summary


if __name__ == "__main__":
    run_all_robustness_checks()
