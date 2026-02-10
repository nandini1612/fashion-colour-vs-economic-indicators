"""
Econometric Analysis for Fashion-Economy Causality (SIMPLIFIED)
================================================================
Uses only 3 economic variables: S&P 500, VIX, Unemployment Rate

FIXED: Correct column mapping to avoid matching lagged variables
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =====================================================
# STATIONARITY TEST
# =====================================================
def test_stationarity(series, name, alpha=0.05):
    """Augmented Dickey-Fuller test for stationarity"""
    result = adfuller(series.dropna(), autolag="AIC")

    output = {
        "variable": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_observations": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < alpha,
    }

    print(f"\n{'=' * 60}")
    print(f"ADF Test: {name}")
    print(f"{'=' * 60}")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-value: {result[1]:.4f}")
    print(f"Used Lag: {result[2]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")

    if result[1] < alpha:
        print("✅ Reject null hypothesis → Series IS stationary")
    else:
        print("⚠️  Cannot reject null → Series is NON-stationary (needs differencing)")

    return output


# =====================================================
# DIFFERENCING
# =====================================================
def difference_series(df, columns):
    """Apply first-order differencing"""
    df_diff = df.copy()
    for col in columns:
        df_diff[f"{col}_diff"] = df[col].diff()
    return df_diff.dropna()


# =====================================================
# GRANGER CAUSALITY (SIMPLIFIED FOR 3 VARIABLES)
# =====================================================
def granger_causality_test(df, target, predictor, max_lag=6):
    """
    Test if predictor Granger-causes target
    Returns both F-test and chi-squared test for robustness
    """
    print(f"\n{'=' * 60}")
    print("GRANGER CAUSALITY TEST")
    print(f"Does '{predictor}' predict '{target}'?")
    print(f"{'=' * 60}")

    test_data = df[[target, predictor]].dropna()

    # Adaptive max_lag based on sample size (rule of thumb: n/8)
    recommended_lag = min(max_lag, len(test_data) // 8)
    print(f"Sample size: {len(test_data)} → Using max_lag = {recommended_lag}")

    try:
        results = grangercausalitytests(
            test_data, maxlag=recommended_lag, verbose=False
        )

        # Extract both F-test and chi-squared test p-values
        p_values_f = {}
        p_values_chi2 = {}

        for lag in range(1, recommended_lag + 1):
            p_values_f[lag] = results[lag][0]["ssr_ftest"][1]
            p_values_chi2[lag] = results[lag][0]["ssr_chi2test"][1]

        best_lag = min(p_values_f, key=p_values_f.get)
        best_p_f = p_values_f[best_lag]
        best_p_chi2 = p_values_chi2[best_lag]

        print("\n📊 Results by Lag:")
        print(f"{'Lag':<6} {'F-test p':<12} {'χ² p':<12} {'Conclusion'}")
        print("-" * 50)

        for lag in range(1, recommended_lag + 1):
            p_f = p_values_f[lag]
            p_chi2 = p_values_chi2[lag]
            sig = "✓ SIGNIFICANT" if p_f < 0.05 else "✗ Not sig"
            print(f"{lag:<6} {p_f:<12.4f} {p_chi2:<12.4f} {sig}")

        print(f"\n🎯 Best Lag: {best_lag}")
        print(f"   F-test p-value: {best_p_f:.4f}")
        print(f"   χ² test p-value: {best_p_chi2:.4f}")

        # Conservative: both tests must agree for significance
        is_significant = (best_p_f < 0.05) and (best_p_chi2 < 0.05)

        if is_significant:
            print(f"✅ STRONG: '{predictor}' DOES Granger-cause '{target}'")
        elif best_p_f < 0.10 and best_p_chi2 < 0.10:
            print(f"⚠️  WEAK: Marginal evidence (p < 0.10)")
        else:
            print("❌ NO: No significant Granger causality")

        return {
            "predictor": predictor,
            "target": target,
            "p_values_f": p_values_f,
            "p_values_chi2": p_values_chi2,
            "best_lag": best_lag,
            "best_p_f": best_p_f,
            "best_p_chi2": best_p_chi2,
            "is_significant": is_significant,
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


# =====================================================
# VAR MODEL (SIMPLIFIED FOR 3 VARIABLES)
# =====================================================
def fit_var_model(df, variables, max_lag=6):
    """Fit Vector Autoregression model"""
    print(f"\n{'=' * 60}")
    print("VAR MODEL ESTIMATION")
    print(f"Variables: {', '.join(variables)}")
    print(f"{'=' * 60}")

    model_data = df[variables].dropna()

    if len(model_data) < max_lag + 10:
        print(f"⚠️  WARNING: Only {len(model_data)} observations. Reducing max_lag.")
        max_lag = max(1, len(model_data) // 10)

    model = VAR(model_data)

    lag_order = model.select_order(maxlags=max_lag)
    print("\n📊 Lag Order Selection:")
    print(lag_order.summary())

    optimal_lag = lag_order.aic
    print(f"\n🎯 Using AIC-optimal lag: {optimal_lag}")

    results = model.fit(optimal_lag)
    print(f"\n{'=' * 60}")
    print("VAR MODEL SUMMARY")
    print(f"{'=' * 60}")
    print(results.summary())

    return results


# =====================================================
# STRUCTURAL BREAK ANALYSIS (SIMPLIFIED)
# =====================================================
def test_structural_break(df, breakpoint_date="2020-03-01"):
    """
    Test if COVID caused a structural break
    SIMPLIFIED: Only tests vibrancy-VIX correlation
    """
    print(f"\n{'=' * 60}")
    print("STRUCTURAL BREAK ANALYSIS")
    print(f"Breakpoint: {breakpoint_date}")
    print(f"{'=' * 60}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["post_covid"] = (df["date"] >= breakpoint_date).astype(int)

    # Split data
    pre_covid = df[df["post_covid"] == 0]
    post_covid = df[df["post_covid"] == 1]

    print(
        f"\nPre-COVID:  {len(pre_covid)} months ({pre_covid['date'].min():%Y-%m} to {pre_covid['date'].max():%Y-%m})"
    )
    print(
        f"Post-COVID: {len(post_covid)} months ({post_covid['date'].min():%Y-%m} to {post_covid['date'].max():%Y-%m})"
    )

    # Compare correlations for all three variables
    results = {}

    for var in ["VIX", "SP500", "Unemployment"]:
        if var in df.columns:
            corr_pre = pre_covid[["vibrancy", var]].corr().iloc[0, 1]
            corr_post = post_covid[["vibrancy", var]].corr().iloc[0, 1]
            corr_full = df[["vibrancy", var]].corr().iloc[0, 1]

            print(f"\n📊 Vibrancy-{var} Correlation:")
            print(f"   Pre-COVID:  {corr_pre:+.3f}")
            print(f"   Post-COVID: {corr_post:+.3f}")
            print(f"   Full period: {corr_full:+.3f}")
            print(f"   Change: {corr_post - corr_pre:+.3f}")

            results[var] = {
                "corr_pre": corr_pre,
                "corr_post": corr_post,
                "corr_full": corr_full,
                "change": corr_post - corr_pre,
            }

    return results


# =====================================================
# RESULTS TABLE
# =====================================================
def create_results_table(granger_results, output_path):
    """Create results table compatible with new Granger output"""
    rows = []

    for result in granger_results:
        if result:
            rows.append(
                {
                    "Predictor": result["predictor"],
                    "Target": result["target"],
                    "Optimal Lag": result["best_lag"],
                    "F-test p": f"{result['best_p_f']:.4f}",
                    "χ² p": f"{result['best_p_chi2']:.4f}",
                    "Significant": "✓" if result["is_significant"] else "✗",
                }
            )

    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(output_path / "granger_results.csv", index=False, encoding="utf-8")

    # Save LaTeX
    latex_table = df.to_latex(index=False, escape=False)
    with open(output_path / "granger_results.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"\n✅ Results saved to {output_path}")
    return df


# =====================================================
# VISUALIZATION (SIMPLIFIED FOR 3 VARIABLES)
# =====================================================
def create_visualization(df, output_dir):
    """Create comprehensive visualization for 3 economic variables"""

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Vibrancy vs VIX
    ax1 = fig.add_subplot(gs[0, :])
    ax1_twin = ax1.twinx()

    ax1.plot(
        df["date"],
        df["vibrancy"],
        color="#2E86AB",
        linewidth=2.5,
        label="Fashion Vibrancy",
        marker="o",
        markersize=4,
    )
    ax1_twin.plot(
        df["date"],
        df["VIX"],
        color="#A23B72",
        linewidth=2,
        label="VIX (Market Fear)",
        linestyle="--",
        marker="s",
        markersize=3,
    )

    ax1.set_ylabel("Fashion Vibrancy Score", fontsize=11, fontweight="bold")
    ax1_twin.set_ylabel("VIX (Volatility Index)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Fashion Vibrancy vs Market Volatility", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Highlight COVID
    covid_start = pd.to_datetime("2020-03-01")
    covid_end = pd.to_datetime("2020-06-30")
    if df["date"].min() <= covid_start <= df["date"].max():
        ax1.axvspan(covid_start, covid_end, alpha=0.2, color="red", label="COVID-19")

    # Panel 2: Vibrancy vs S&P 500
    ax2 = fig.add_subplot(gs[1, :])
    ax2_twin = ax2.twinx()

    ax2.plot(
        df["date"],
        df["vibrancy"],
        color="#2E86AB",
        linewidth=2.5,
        label="Fashion Vibrancy",
        marker="o",
        markersize=4,
    )
    ax2_twin.plot(
        df["date"],
        df["SP500"],
        color="#6A994E",
        linewidth=2,
        label="S&P 500",
        marker="^",
        markersize=3,
    )

    ax2.set_ylabel("Fashion Vibrancy Score", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("S&P 500 Index", fontsize=11, fontweight="bold")
    ax2.set_title("Fashion Vibrancy vs Stock Market", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    if df["date"].min() <= covid_start <= df["date"].max():
        ax2.axvspan(covid_start, covid_end, alpha=0.2, color="red")

    # Panel 3: Vibrancy vs Unemployment
    ax3 = fig.add_subplot(gs[2, :])
    ax3_twin = ax3.twinx()

    ax3.plot(
        df["date"],
        df["vibrancy"],
        color="#2E86AB",
        linewidth=2.5,
        label="Fashion Vibrancy",
        marker="o",
        markersize=4,
    )
    ax3_twin.plot(
        df["date"],
        df["Unemployment"],
        color="#F18F01",
        linewidth=2,
        label="Unemployment Rate",
        marker="d",
        markersize=3,
    )

    ax3.set_ylabel("Fashion Vibrancy Score", fontsize=11, fontweight="bold")
    ax3_twin.set_ylabel("Unemployment Rate (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Fashion Vibrancy vs Labor Market", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(alpha=0.3)

    if df["date"].min() <= covid_start <= df["date"].max():
        ax3.axvspan(covid_start, covid_end, alpha=0.2, color="red")

    # Panel 4: Correlation Matrix
    ax4 = fig.add_subplot(gs[3, 0])
    corr = df[["vibrancy", "VIX", "SP500", "Unemployment"]].corr()
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        ax=ax4,
        cbar_kws={"shrink": 0.8},
        linewidths=1,
        linecolor="white",
    )
    ax4.set_title("Correlation Matrix", fontsize=12, fontweight="bold")

    # Panel 5: Summary Statistics
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis("off")

    summary_text = f"""
    Summary Statistics
    ==================
    
    Sample Size: {len(df)} months
    Date Range: {df["date"].min():%Y-%m} to {df["date"].max():%Y-%m}
    
    Fashion Vibrancy:
      Mean: {df["vibrancy"].mean():.2f}
      Std:  {df["vibrancy"].std():.2f}
    
    VIX:
      Mean: {df["VIX"].mean():.2f}
      Std:  {df["VIX"].std():.2f}
    
    S&P 500:
      Mean: {df["SP500"].mean():.2f}
      Std:  {df["SP500"].std():.2f}
    
    Unemployment:
      Mean: {df["Unemployment"].mean():.2f}%
      Std:  {df["Unemployment"].std():.2f}%
    """

    ax5.text(
        0.1,
        0.5,
        summary_text,
        fontsize=9,
        family="monospace",
        verticalalignment="center",
        transform=ax5.transAxes,
    )

    plt.savefig(
        output_dir / "fashion_economy_comprehensive.png", dpi=300, bbox_inches="tight"
    )
    print(
        f"\n✅ Visualization saved to {output_dir / 'fashion_economy_comprehensive.png'}"
    )
    plt.close()


# =====================================================
# MAIN PIPELINE (SIMPLIFIED) - FIXED COLUMN MAPPING
# =====================================================
def main():
    print("=" * 70)
    print("ECONOMETRIC ANALYSIS PIPELINE (3 VARIABLES)")
    print("Variables: Fashion Vibrancy, VIX, S&P 500, Unemployment Rate")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    data_path = (
        project_root / "data" / "processed_features" / "fashion_economy_aligned.csv"
    )
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"\n📂 Loading: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    print(f"✅ Loaded {len(df)} months of data")
    print(f"   Date range: {df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")

    # Check available columns
    print(f"\n📊 Available columns: {df.columns.tolist()}")

    # FIXED: Better column detection that avoids lagged columns
    required = ["vibrancy", "VIX", "SP500", "Unemployment"]

    # First check if columns exist exactly as named
    all_exist = all(col in df.columns for col in required)

    if not all_exist:
        print("\n⚠️  Required columns not found with exact names.")
        print("Attempting smart mapping...")

        column_map = {}
        for req_col in required:
            # Find columns that match but aren't lagged versions
            matches = [
                col
                for col in df.columns
                if req_col.lower() in col.lower() and "lag" not in col.lower()
            ]

            if matches:
                # Prefer exact match, otherwise take first
                if req_col in matches:
                    column_map[req_col] = req_col
                else:
                    column_map[req_col] = matches[0]

        print(f"📋 Column mapping: {column_map}")

        # Verify we found all required columns
        if len(column_map) != len(required):
            missing = [col for col in required if col not in column_map]
            print(f"\n❌ ERROR: Could not find columns for: {missing}")
            print(
                f"Available non-lagged columns: {[c for c in df.columns if 'lag' not in c.lower()]}"
            )
            return None

        # Create working dataframe with standardized names
        df_work = df.copy()
        for standard_name, actual_name in column_map.items():
            if standard_name != actual_name:
                df_work[standard_name] = df[actual_name]
    else:
        df_work = df.copy()
        print("✅ All required columns found with exact names")

    # -----------------------------
    # STEP 1: STATIONARITY
    # -----------------------------
    print("\n" + "=" * 70)
    print("STEP 1: STATIONARITY TESTS")
    print("=" * 70)

    variables = ["vibrancy", "VIX", "SP500", "Unemployment"]
    adf_results = [test_stationarity(df_work[v], v) for v in variables]

    needs_diff = [r["variable"] for r in adf_results if not r["is_stationary"]]

    if needs_diff:
        print(f"\n⚠️  Variables needing differencing: {needs_diff}")
        df_stationary = difference_series(df_work, needs_diff)
        print(f"   After differencing: {len(df_stationary)} observations")
    else:
        print(f"\n✅ All variables are stationary")
        df_stationary = df_work

    var_map = {v: f"{v}_diff" if v in needs_diff else v for v in variables}

    # -----------------------------
    # STEP 2: STRUCTURAL BREAK
    # -----------------------------
    print("\n" + "=" * 70)
    print("STEP 2: STRUCTURAL BREAK ANALYSIS")
    print("=" * 70)

    break_results = test_structural_break(df_work)

    # -----------------------------
    # STEP 3: GRANGER CAUSALITY (ALL COMBINATIONS)
    # -----------------------------
    print("\n" + "=" * 70)
    print("STEP 3: GRANGER CAUSALITY TESTS (ALL COMBINATIONS)")
    print("=" * 70)

    # Adaptive max_lag
    max_lag_to_use = min(8, len(df_stationary) // 8)
    print(f"\n📊 Using max_lag = {max_lag_to_use} (based on sample size)")

    # Test all meaningful combinations
    granger_tests = [
        # Fashion → Economic indicators
        (var_map["VIX"], var_map["vibrancy"], "vibrancy → VIX"),
        (var_map["SP500"], var_map["vibrancy"], "vibrancy → SP500"),
        (var_map["Unemployment"], var_map["vibrancy"], "vibrancy → Unemployment"),
        # Economic indicators → Fashion (reverse causality check)
        (var_map["vibrancy"], var_map["VIX"], "VIX → vibrancy"),
        (var_map["vibrancy"], var_map["SP500"], "SP500 → vibrancy"),
        (var_map["vibrancy"], var_map["Unemployment"], "Unemployment → vibrancy"),
    ]

    granger_results = []
    for target, predictor, description in granger_tests:
        print(f"\n{'=' * 60}")
        print(f"Testing: {description}")
        result = granger_causality_test(
            df_stationary, target, predictor, max_lag_to_use
        )
        if result:
            granger_results.append(result)

    results_df = create_results_table(granger_results, output_dir)
    print("\n📊 Granger Causality Summary:")
    print(results_df.to_string(index=False))

    # -----------------------------
    # STEP 4: VAR MODEL (TRIVARIATE)
    # -----------------------------
    if len(df_stationary) >= 20:
        print("\n" + "=" * 70)
        print("STEP 4: TRIVARIATE VAR MODEL")
        print("=" * 70)

        # Full model with all 3 economic variables
        var_results = fit_var_model(
            df_stationary,
            [var_map["vibrancy"], var_map["VIX"], var_map["SP500"]],
            max_lag=max_lag_to_use,
        )

        print("\n" + "=" * 70)
        print("STEP 5: VAR WITH UNEMPLOYMENT")
        print("=" * 70)

        # Alternative specification with unemployment
        var_results_unemp = fit_var_model(
            df_stationary,
            [var_map["vibrancy"], var_map["VIX"], var_map["Unemployment"]],
            max_lag=max_lag_to_use,
        )

    else:
        print("\n⚠️  Insufficient data for VAR (need 20+)")

    # -----------------------------
    # STEP 6: VISUALIZATION
    # -----------------------------
    print("\n" + "=" * 70)
    print("STEP 6: CREATING VISUALIZATIONS")
    print("=" * 70)

    create_visualization(df_work, output_dir)

    # -----------------------------
    # FINAL SUMMARY
    # -----------------------------
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\n📁 Results saved to: {output_dir}")
    print("   - granger_results.csv")
    print("   - granger_results.tex")
    print("   - fashion_economy_comprehensive.png")

    print(f"\n📊 Key Findings:")
    print(f"   Sample size: {len(df_work)} months")

    for var in ["VIX", "SP500", "Unemployment"]:
        corr = df_work[["vibrancy", var]].corr().iloc[0, 1]
        print(f"   Vibrancy-{var} correlation: {corr:+.3f}")

        if var in break_results:
            print(f"     Pre-COVID:  {break_results[var]['corr_pre']:+.3f}")
            print(f"     Post-COVID: {break_results[var]['corr_post']:+.3f}")

    # Determine if significant Granger causality found
    sig_results = [r for r in granger_results if r and r["is_significant"]]
    if sig_results:
        print(
            f"\n✅ Found {len(sig_results)} significant Granger causality relationships!"
        )
        for r in sig_results:
            print(f"   {r['predictor']} → {r['target']} (p={r['best_p_f']:.4f})")
    else:
        print(f"\n⚠️  No statistically significant Granger causality at p<0.05 level")
        print("   → This is where regime-dependent analysis becomes crucial")
        print(
            "   → Run regime_analysis.py to test if relationship varies by crisis period"
        )


if __name__ == "__main__":
    main()
