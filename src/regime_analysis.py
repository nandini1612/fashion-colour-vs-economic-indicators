"""
Regime-Dependent Analysis for Fashion-Economy Research (3 VARIABLES)
=====================================================================
Tests if fashion correlations with VIX, S&P 500, and Unemployment
vary by market regime (calm vs crisis)
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import seaborn as sns


class RegimeDependentAnalyzer:
    """
    Test if fashion-economy relationships vary by market regime
    SIMPLIFIED: Works with 3 economic variables only
    """

    def __init__(self, df):
        """
        Args:
            df: DataFrame with columns ['date', 'vibrancy', 'VIX', 'SP500', 'Unemployment']
        """
        self.df = df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        self.df = self.df.sort_values("date").reset_index(drop=True)

        # Verify required columns
        required = ["vibrancy", "VIX", "SP500", "Unemployment"]
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def define_crisis_periods(self):
        """
        Define volatility regimes based on VIX thresholds and known events
        """
        crises = {
            "2015-2016 China Slowdown": ("2015-08-01", "2016-02-29"),
            "2018 Q4 Correction": ("2018-10-01", "2018-12-31"),
            "2020 COVID-19": ("2020-02-15", "2020-06-30"),
            "2022 Inflation Spike": ("2022-01-01", "2022-06-30"),
            "2022 Ukraine Crisis": ("2022-02-20", "2022-04-30"),
        }

        # VIX-based regime definition
        self.df["regime"] = "Calm"
        self.df.loc[self.df["VIX"] > 25, "regime"] = "High Volatility"
        self.df.loc[self.df["VIX"] > 35, "regime"] = "Crisis"

        # Mark known crisis periods
        for crisis_name, (start, end) in crises.items():
            mask = (self.df["date"] >= start) & (self.df["date"] <= end)
            self.df.loc[mask, "crisis_period"] = crisis_name

        return crises

    def threshold_analysis(self, vix_threshold=25):
        """
        Compare correlations in high vs low volatility regimes
        FOR ALL 3 ECONOMIC VARIABLES
        """
        print("=" * 70)
        print("THRESHOLD ANALYSIS: Testing Regime-Dependent Correlations")
        print("=" * 70)

        low_vol = self.df[self.df["VIX"] <= vix_threshold].copy()
        high_vol = self.df[self.df["VIX"] > vix_threshold].copy()

        print(f"\nSample sizes:")
        print(f"  Low volatility (VIX ≤ {vix_threshold}): n={len(low_vol)}")
        print(f"  High volatility (VIX > {vix_threshold}): n={len(high_vol)}")

        results = {}

        # Test for each economic variable
        for econ_var in ["VIX", "SP500", "Unemployment"]:
            print(f"\n--- {econ_var} ---")

            corr_low = low_vol[["vibrancy", econ_var]].corr().iloc[0, 1]
            corr_high = high_vol[["vibrancy", econ_var]].corr().iloc[0, 1]

            print(f"  Low volatility:  r = {corr_low:+.3f}")
            print(f"  High volatility: r = {corr_high:+.3f}")
            print(f"  Difference:      Δr = {abs(corr_high - corr_low):.3f}")

            # Fisher Z-test
            z_low = np.arctanh(corr_low)
            z_high = np.arctanh(corr_high)

            se = np.sqrt(1 / (len(low_vol) - 3) + 1 / (len(high_vol) - 3))
            z_stat = (z_high - z_low) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            print(f"  Fisher Z-test p-value: {p_value:.4f}")

            if p_value < 0.05:
                print(f"  ✅ SIGNIFICANT: Correlations differ by regime")
            elif p_value < 0.10:
                print(f"  ⚠️  MARGINAL: Weak evidence (p < 0.10)")
            else:
                print(f"  ❌ NOT SIGNIFICANT: No regime dependence")

            results[econ_var] = {
                "corr_low": corr_low,
                "corr_high": corr_high,
                "p_value": p_value,
                "z_stat": z_stat,
            }

        return results

    def crisis_specific_granger(self, vix_threshold=25, maxlag=3):
        """
        Run Granger causality ONLY on high-volatility periods
        FOR ALL 3 ECONOMIC VARIABLES
        """
        print("\n" + "=" * 70)
        print("CRISIS-SPECIFIC GRANGER CAUSALITY")
        print("=" * 70)

        crisis_df = self.df[self.df["VIX"] > vix_threshold].copy()

        print(f"\nCrisis sample: {len(crisis_df)} months (VIX > {vix_threshold})")

        if len(crisis_df) < maxlag + 5:
            print("⚠️  WARNING: Sample too small for reliable Granger test")
            return None

        # Prepare differenced data
        crisis_df["vibrancy_diff"] = crisis_df["vibrancy"].diff()
        crisis_df["VIX_diff"] = crisis_df["VIX"].diff()
        crisis_df["SP500_diff"] = crisis_df["SP500"].diff()
        crisis_df["Unemployment_diff"] = crisis_df["Unemployment"].diff()
        crisis_df = crisis_df.dropna()

        results = {}

        # Test vibrancy → each economic variable
        for econ_var in ["VIX", "SP500", "Unemployment"]:
            print(f"\n--- Testing: Vibrancy → {econ_var} (crisis periods only) ---")

            try:
                result = grangercausalitytests(
                    crisis_df[[f"{econ_var}_diff", "vibrancy_diff"]],
                    maxlag=maxlag,
                    verbose=False,
                )

                for lag in range(1, maxlag + 1):
                    p_ftest = result[lag][0]["ssr_ftest"][1]
                    p_chi2 = result[lag][0]["ssr_chi2test"][1]

                    print(f"  Lag {lag}: F-test p={p_ftest:.4f}, χ² p={p_chi2:.4f}")

                    if p_ftest < 0.05 or p_chi2 < 0.05:
                        print(f"    ✅ Significant at lag {lag}")

                results[econ_var] = result

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results[econ_var] = None

        return results

    def multi_regime_comparison(self):
        """
        Compare correlations across all defined crisis periods
        FOR ALL 3 ECONOMIC VARIABLES
        """
        print("\n" + "=" * 70)
        print("MULTI-REGIME CORRELATION ANALYSIS")
        print("=" * 70)

        crises = self.define_crisis_periods()

        results = []

        for crisis_name, (start, end) in crises.items():
            mask = (self.df["date"] >= start) & (self.df["date"] <= end)
            crisis_data = self.df[mask]

            if len(crisis_data) >= 3:
                row = {
                    "Crisis Period": crisis_name,
                    "Start": start,
                    "End": end,
                    "N": len(crisis_data),
                }

                # Calculate correlations for all 3 variables
                for econ_var in ["VIX", "SP500", "Unemployment"]:
                    corr = crisis_data[["vibrancy", econ_var]].corr().iloc[0, 1]
                    row[f"Corr_{econ_var}"] = corr

                results.append(row)

        # Add calm period
        calm_mask = self.df["VIX"] <= 20
        calm_data = self.df[calm_mask]

        calm_row = {
            "Crisis Period": "Calm Periods (Combined)",
            "Start": "Various",
            "End": "Various",
            "N": len(calm_data),
        }

        for econ_var in ["VIX", "SP500", "Unemployment"]:
            corr = calm_data[["vibrancy", econ_var]].corr().iloc[0, 1]
            calm_row[f"Corr_{econ_var}"] = corr

        results.append(calm_row)

        df_results = pd.DataFrame(results)

        print("\n" + df_results.to_string(index=False))

        # Summary statistics
        print(f"\n\nSummary:")
        crisis_rows = df_results[
            df_results["Crisis Period"] != "Calm Periods (Combined)"
        ]
        calm_corrs = df_results[
            df_results["Crisis Period"] == "Calm Periods (Combined)"
        ]

        for econ_var in ["VIX", "SP500", "Unemployment"]:
            col_name = f"Corr_{econ_var}"
            crisis_mean = crisis_rows[col_name].mean()
            calm_corr = calm_corrs[col_name].values[0]

            print(f"\n{econ_var}:")
            print(f"  Mean crisis correlation: {crisis_mean:+.3f}")
            print(f"  Calm period correlation: {calm_corr:+.3f}")
            print(f"  Difference: {abs(crisis_mean - calm_corr):.3f}")

        return df_results

    def visualize_regime_dependence(self, save_path="regime_analysis_3vars.png"):
        """
        Create publication-quality figure showing regime dependence
        FOR ALL 3 ECONOMIC VARIABLES
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

        # Panel A: Time series with regime shading (VIX)
        ax1 = fig.add_subplot(gs[0, :])
        ax1_twin = ax1.twinx()

        ax1.plot(
            self.df["date"],
            self.df["vibrancy"],
            color="#2E86AB",
            linewidth=2,
            label="Fashion Vibrancy",
        )
        ax1_twin.plot(
            self.df["date"],
            self.df["VIX"],
            color="#A23B72",
            linewidth=1.5,
            linestyle="--",
            label="VIX",
        )

        # Shade crisis periods
        crises = self.define_crisis_periods()
        colors = ["red", "orange", "purple", "brown", "pink"]

        for i, (crisis_name, (start, end)) in enumerate(crises.items()):
            ax1.axvspan(
                pd.to_datetime(start),
                pd.to_datetime(end),
                alpha=0.2,
                color=colors[i % len(colors)],
            )

        ax1.set_ylabel("Vibrancy", fontweight="bold", fontsize=11)
        ax1_twin.set_ylabel("VIX", fontweight="bold", fontsize=11)
        ax1.set_title(
            "Panel A: Fashion Vibrancy vs VIX Over Time", fontweight="bold", fontsize=12
        )
        ax1.legend(loc="upper left")
        ax1_twin.legend(loc="upper right")
        ax1.grid(alpha=0.3)

        # Panel B: Correlation bars for all 3 variables by regime
        ax2 = fig.add_subplot(gs[1, :])

        regime_data = []
        for regime in ["Calm", "High Volatility", "Crisis"]:
            regime_df = self.df[self.df["regime"] == regime]
            if len(regime_df) > 0:
                for econ_var in ["VIX", "SP500", "Unemployment"]:
                    corr = regime_df[["vibrancy", econ_var]].corr().iloc[0, 1]
                    regime_data.append(
                        {"Regime": regime, "Variable": econ_var, "Correlation": corr}
                    )

        regime_corr_df = pd.DataFrame(regime_data)

        # Create grouped bar chart
        x = np.arange(len(regime_corr_df["Regime"].unique()))
        width = 0.25

        for i, var in enumerate(["VIX", "SP500", "Unemployment"]):
            var_data = regime_corr_df[regime_corr_df["Variable"] == var]
            ax2.bar(x + i * width, var_data["Correlation"], width, label=var)

        ax2.set_ylabel("Correlation Coefficient", fontweight="bold", fontsize=11)
        ax2.set_title(
            "Panel B: Correlation Strength by Volatility Regime",
            fontweight="bold",
            fontsize=12,
        )
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(regime_corr_df["Regime"].unique())
        ax2.legend()
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.grid(axis="y", alpha=0.3)

        # Panel C: Vibrancy vs S&P 500
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.scatter(
            self.df["vibrancy"],
            self.df["SP500"],
            c=self.df["VIX"],
            cmap="RdYlGn_r",
            s=50,
            alpha=0.6,
        )
        ax3.set_xlabel("Vibrancy", fontweight="bold")
        ax3.set_ylabel("S&P 500", fontweight="bold")
        ax3.set_title("Panel C: Vibrancy vs S&P 500\n(Color = VIX)", fontweight="bold")
        ax3.grid(alpha=0.3)

        # Add trendline
        z = np.polyfit(self.df["vibrancy"].dropna(), self.df["SP500"].dropna(), 1)
        p = np.poly1d(z)
        ax3.plot(
            self.df["vibrancy"], p(self.df["vibrancy"]), "r--", alpha=0.8, linewidth=2
        )

        # Panel D: Vibrancy vs Unemployment
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.scatter(
            self.df["vibrancy"],
            self.df["Unemployment"],
            c=self.df["VIX"],
            cmap="RdYlGn_r",
            s=50,
            alpha=0.6,
        )
        ax4.set_xlabel("Vibrancy", fontweight="bold")
        ax4.set_ylabel("Unemployment Rate (%)", fontweight="bold")
        ax4.set_title(
            "Panel D: Vibrancy vs Unemployment\n(Color = VIX)", fontweight="bold"
        )
        ax4.grid(alpha=0.3)

        z = np.polyfit(
            self.df["vibrancy"].dropna(), self.df["Unemployment"].dropna(), 1
        )
        p = np.poly1d(z)
        ax4.plot(
            self.df["vibrancy"], p(self.df["vibrancy"]), "r--", alpha=0.8, linewidth=2
        )

        # Panel E: Rolling correlations for all 3 variables
        ax5 = fig.add_subplot(gs[3, :])

        for econ_var, color in zip(
            ["VIX", "SP500", "Unemployment"], ["#A23B72", "#6A994E", "#F18F01"]
        ):
            rolling_corr = (
                self.df["vibrancy"].rolling(window=12).corr(self.df[econ_var])
            )
            ax5.plot(
                self.df["date"],
                rolling_corr,
                label=f"Vibrancy-{econ_var}",
                color=color,
                linewidth=2,
            )

        ax5.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax5.set_ylabel("12-Month Rolling Correlation", fontweight="bold", fontsize=11)
        ax5.set_xlabel("Date", fontweight="bold", fontsize=11)
        ax5.set_title(
            "Panel E: Time-Varying Correlations (12-Month Window)",
            fontweight="bold",
            fontsize=12,
        )
        ax5.legend()
        ax5.grid(alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✅ Figure saved to {save_path}")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Load your data
    df = pd.read_csv("../data/processed_features/fashion_economy_aligned.csv")

    # Run analysis
    analyzer = RegimeDependentAnalyzer(df)

    # 1. Threshold analysis (all 3 variables)
    print("\n" + "=" * 70)
    print("ANALYSIS 1: THRESHOLD COMPARISON")
    print("=" * 70)
    threshold_results = analyzer.threshold_analysis(vix_threshold=25)

    # 2. Crisis-specific Granger (all 3 variables)
    print("\n" + "=" * 70)
    print("ANALYSIS 2: CRISIS-PERIOD GRANGER CAUSALITY")
    print("=" * 70)
    granger_results = analyzer.crisis_specific_granger(maxlag=3)

    # 3. Multi-regime comparison (all 3 variables)
    print("\n" + "=" * 70)
    print("ANALYSIS 3: MULTI-REGIME COMPARISON")
    print("=" * 70)
    regime_table = analyzer.multi_regime_comparison()

    # 4. Create comprehensive figure
    print("\n" + "=" * 70)
    print("ANALYSIS 4: VISUALIZATION")
    print("=" * 70)
    analyzer.visualize_regime_dependence()

    print("\n" + "=" * 70)
    print("✅ ALL ANALYSES COMPLETE")
    print("=" * 70)
