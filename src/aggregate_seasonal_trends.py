"""
Fashion-Economy Data Aggregation Pipeline
==========================================
Production-grade script for aggregating fashion features by season
and aligning with economic indicators (VIX, S&P 500, Unemployment)

UPDATED: Works with actual all_paris_features.csv column names
- brand (not designer)
- collection (not separate season/year)
- vibrancy (not vibrancy_score)

Author: Fashion Economics Research Team
Version: 2.1
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, Dict, Tuple
import warnings
import re

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================


def parse_collection_name(collection: str) -> Tuple[str, int]:
    """
    Parse collection string to extract season and year.

    Expected formats:
    - "Fall 2020" → ("Fall", 2020)
    - "Spring 2021" → ("Spring", 2021)
    - "fall_2020_ready_to_wear" → ("Fall", 2020)
    - "spring_2021_ready_to_wear" → ("Spring", 2021)
    - "FW2020" → ("Fall", 2020)
    - "SS2021" → ("Spring", 2021)

    Args:
        collection: Collection name string

    Returns:
        Tuple of (season, year)
    """
    collection = (
        str(collection).strip().lower()
    )  # Convert to lowercase for easier matching

    # Pattern 1: Underscore format - "fall_2020_ready_to_wear", "spring_2021_ready_to_wear"
    match = re.search(r"(fall|spring|winter|summer)_(\d{4})", collection)
    if match:
        season_str = match.group(1)
        year = int(match.group(2))
        # Normalize to Fall/Spring
        if season_str in ["winter", "fall"]:
            season = "Fall"
        else:
            season = "Spring"
        return season, year

    # Pattern 2: Space format - "Fall 2020", "Spring 2021"
    match = re.search(r"(fall|spring|winter|summer)\s*(\d{4})", collection)
    if match:
        season_str = match.group(1)
        year = int(match.group(2))
        # Normalize to Fall/Spring
        if season_str in ["winter", "fall"]:
            season = "Fall"
        else:
            season = "Spring"
        return season, year

    # Pattern 3: "FW2020", "SS2021", "AW2020"
    match = re.search(r"(fw|ss|aw)(\d{4})", collection)
    if match:
        season_code = match.group(1)
        year = int(match.group(2))
        season = "Fall" if season_code in ["fw", "aw"] else "Spring"
        return season, year

    # Pattern 4: Just year "2020" - assume Fall
    match = re.search(r"(\d{4})", collection)
    if match:
        year = int(match.group(1))
        logger.warning(
            f"Collection '{collection}' has no season indicator, assuming Fall"
        )
        return "Fall", year

    logger.error(f"Could not parse collection: '{collection}'")
    return "Unknown", 0


def load_fashion_data(csv_path: Path) -> pd.DataFrame:
    """
    Load extracted fashion color features from CSV.

    UPDATED: Works with actual column names from feature extraction:
    - brand (instead of designer)
    - collection (instead of separate season/year)
    - vibrancy (instead of vibrancy_score)

    Args:
        csv_path: Path to fashion features CSV

    Returns:
        DataFrame with fashion features

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Fashion data not found: {csv_path}")

    logger.info(f"Loading fashion data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check for actual column names from feature extraction
    actual_required = ["brand", "collection", "vibrancy", "filename"]
    missing_cols = [col for col in actual_required if col not in df.columns]

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse collection into season and year
    logger.info("Parsing collection names to extract season and year...")
    parsed = df["collection"].apply(parse_collection_name)
    df["season"] = parsed.apply(lambda x: x[0])
    df["year"] = parsed.apply(lambda x: x[1])

    # Remove rows where parsing failed
    invalid_rows = (df["season"] == "Unknown") | (df["year"] == 0)
    if invalid_rows.sum() > 0:
        logger.warning(
            f"Removing {invalid_rows.sum()} rows with unparseable collections"
        )
        df = df[~invalid_rows].copy()

    # Rename columns to match expected names
    df = df.rename(columns={"brand": "designer", "vibrancy": "vibrancy_score"})

    # Calculate additional metrics if they don't exist
    if "darkness_score" not in df.columns and "mean_brightness" in df.columns:
        df["darkness_score"] = 1 - df["mean_brightness"]

    if "brightness_score" not in df.columns and "mean_brightness" in df.columns:
        df["brightness_score"] = df["mean_brightness"]

    if "color_diversity" not in df.columns:
        # Simple diversity: range of saturation values or number of distinct colors
        df["color_diversity"] = (
            df["mean_saturation"]
            if "mean_saturation" in df.columns
            else df["vibrancy_score"]
        )

    if "weighted_vibrancy" not in df.columns:
        df["weighted_vibrancy"] = df["vibrancy_score"]

    logger.info(f"Loaded {len(df):,} images from {df['designer'].nunique()} designers")
    logger.info(f"Seasons: {dict(df['season'].value_counts())}")
    logger.info(f"Year range: {df['year'].min()} to {df['year'].max()}")

    # Show sample of parsed collections
    sample_collections = df[["collection", "season", "year"]].drop_duplicates().head(10)
    logger.info(f"\nSample parsed collections:\n{sample_collections.to_string()}")

    return df


# =============================================================================
# FASHION DATA AGGREGATION
# =============================================================================


def assign_show_date(row: pd.Series) -> pd.Timestamp:
    """
    Assign approximate show date for each collection.

    Fashion Week Schedule:
    - Fall/Winter collections: shown in February/March
    - Spring/Summer collections: shown in September/October (previous year)

    Args:
        row: DataFrame row with 'season' and 'year'

    Returns:
        Approximate show date
    """
    year = row["year"]

    if row["season"] == "Fall":
        # Fall shows happen in Feb of that year
        # Use Feb 28 (or 29 for leap years)
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            return pd.to_datetime(f"{year}-02-29")
        else:
            return pd.to_datetime(f"{year}-02-28")
    else:
        # Spring shows happen in Sept/Oct of previous year
        return pd.to_datetime(f"{year - 1}-09-30")


def aggregate_by_show(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all looks from each designer-season-year combination.

    Args:
        df: Raw fashion features DataFrame

    Returns:
        Show-level aggregated DataFrame
    """
    logger.info("Aggregating by show (designer-season-year)...")

    # Assign show dates
    df["show_date"] = df.apply(assign_show_date, axis=1)

    # Define aggregation functions
    agg_funcs = {
        "vibrancy_score": ["mean", "std", "min", "max"],
        "darkness_score": ["mean", "std"],
        "brightness_score": ["mean", "std"],
        "color_diversity": ["mean", "std"],
        "weighted_vibrancy": ["mean"],
        "filename": "count",
    }

    # Aggregate
    show_data = (
        df.groupby(["designer", "year", "season", "show_date"])
        .agg(agg_funcs)
        .reset_index()
    )

    # Flatten column names
    show_data.columns = ["_".join(col).strip("_") for col in show_data.columns]
    show_data.rename(columns={"filename_count": "n_looks"}, inplace=True)

    logger.info(f"Created {len(show_data)} show-level records")

    return show_data


def aggregate_industry_wide(show_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create industry-wide seasonal fashion trends by aggregating across designers.

    Args:
        show_data: Show-level aggregated data

    Returns:
        Industry-wide aggregated DataFrame
    """
    logger.info("Creating industry-wide trends...")

    industry_agg = (
        show_data.groupby("show_date")
        .agg(
            {
                "vibrancy_score_mean": "mean",
                "vibrancy_score_std": "mean",
                "darkness_score_mean": "mean",
                "brightness_score_mean": "mean",
                "color_diversity_mean": "mean",
                "weighted_vibrancy_mean": "mean",
                "n_looks": "sum",
                "designer": "count",
            }
        )
        .reset_index()
    )

    industry_agg.rename(columns={"designer": "n_designers"}, inplace=True)

    # Rename for clarity
    industry_agg.columns = [
        "show_date",
        "vibrancy",
        "vibrancy_std",
        "darkness",
        "brightness",
        "color_diversity",
        "weighted_vibrancy",
        "total_looks",
        "n_designers",
    ]

    logger.info(f"Industry aggregation: {len(industry_agg)} time points")

    return industry_agg


# =============================================================================
# TIME SERIES INTERPOLATION
# =============================================================================


def create_monthly_timeseries(
    industry_agg: pd.DataFrame, end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Interpolate bi-annual fashion signals to monthly frequency.

    Fashion shows happen twice per year, but we need monthly data
    to align with economic indicators. We forward-fill fashion metrics
    between shows (each collection's influence lasts 6 months).

    Args:
        industry_agg: Industry-wide aggregated data
        end_date: Optional end date for time series

    Returns:
        Monthly fashion DataFrame
    """
    logger.info("Interpolating to monthly frequency...")

    if end_date is None:
        end_date = industry_agg["show_date"].max() + pd.DateOffset(months=6)

    # Create monthly date range
    date_range = pd.date_range(
        start=industry_agg["show_date"].min(), end=end_date, freq="M"
    )

    monthly_df = pd.DataFrame({"date": date_range})

    # Merge with show data
    monthly_df = monthly_df.merge(
        industry_agg, left_on="date", right_on="show_date", how="left"
    )

    # Forward fill fashion metrics (collections stay relevant for 6 months)
    fashion_cols = [
        "vibrancy",
        "darkness",
        "brightness",
        "color_diversity",
        "weighted_vibrancy",
    ]

    for col in fashion_cols:
        monthly_df[col] = monthly_df[col].ffill()

    # Mark months with actual shows
    monthly_df["has_show"] = monthly_df["show_date"].notna()

    result = monthly_df[["date"] + fashion_cols + ["has_show"]].copy()

    logger.info(f"Created {len(result)} monthly observations")

    return result


# =============================================================================
# ECONOMIC DATA
# =============================================================================


def load_economic_data(econ_csv: Path) -> Optional[pd.DataFrame]:
    """
    Load economic data with validation.

    Expected columns:
    - Date: Month-end dates
    - VIX: CBOE Volatility Index
    - SP500: S&P 500 Index level
    - Unemployment: U.S. unemployment rate (%)

    Args:
        econ_csv: Path to economic indicators CSV

    Returns:
        Economic DataFrame or None if loading fails
    """
    if not econ_csv.exists():
        logger.error(f"Economic data not found: {econ_csv}")
        return None

    logger.info(f"Loading economic data from: {econ_csv}")

    try:
        df = pd.read_csv(econ_csv)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
        df = df.rename(columns={"Date": "date"})

        # Validate required columns
        required_cols = ["VIX", "SP500", "Unemployment"]
        available_cols = [col for col in required_cols if col in df.columns]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.warning(f"Missing economic columns: {missing_cols}")

            # Try alternative column names
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if "vix" in col_lower and "VIX" not in df.columns:
                    rename_map[col] = "VIX"
                elif (
                    any(term in col_lower for term in ["sp500", "s&p", "s_p"])
                    and "SP500" not in df.columns
                ):
                    rename_map[col] = "SP500"
                elif (
                    any(
                        term in col_lower
                        for term in ["unemployment", "unemp", "unrate"]
                    )
                    and "Unemployment" not in df.columns
                ):
                    rename_map[col] = "Unemployment"

            if rename_map:
                logger.info(f"Renaming columns: {rename_map}")
                df = df.rename(columns=rename_map)
                missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"Still missing required columns: {missing_cols}")
            logger.error("Required format: Date, VIX, SP500, Unemployment")
            return None

        logger.info("Economic data loaded successfully")
        logger.info(f"Variables: {[col for col in df.columns if col != 'date']}")
        logger.info(
            f"Date range: {df['date'].min().strftime('%Y-%m')} to {df['date'].max().strftime('%Y-%m')}"
        )
        logger.info(f"Total observations: {len(df)}")

        return df

    except Exception as e:
        logger.error(f"Error loading economic data: {e}")
        return None


def align_fashion_and_economy(
    fashion_monthly: pd.DataFrame, economic_data: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Merge fashion and economic data, create lagged features.

    Args:
        fashion_monthly: Monthly fashion trends
        economic_data: Economic indicators

    Returns:
        Aligned DataFrame with lagged features
    """
    if economic_data is None:
        logger.error("Cannot align: economic data is None")
        return None

    logger.info("Aligning fashion and economic data...")

    # Merge on date
    combined = fashion_monthly.merge(economic_data, on="date", how="inner")

    logger.info(f"Merge result: {len(combined)} months of aligned data")

    if len(combined) == 0:
        logger.error("No overlapping dates after merge!")
        logger.error(
            f"Fashion date range: {fashion_monthly['date'].min()} to {fashion_monthly['date'].max()}"
        )
        logger.error(
            f"Economic date range: {economic_data['date'].min()} to {economic_data['date'].max()}"
        )
        return combined

    # Create lagged features for economic variables
    economic_vars = ["VIX", "SP500", "Unemployment"]
    lags = [1, 2, 3, 6]

    logger.info("Creating lagged features...")

    for var in economic_vars:
        if var in combined.columns:
            for lag in lags:
                combined[f"{var}_lag{lag}"] = combined[var].shift(lag)

    # Lag vibrancy (for reverse causality tests)
    for lag in lags:
        combined[f"vibrancy_lag{lag}"] = combined["vibrancy"].shift(lag)

    # Drop rows with NaN from lagging
    combined = combined.dropna()

    logger.info(f"Final dataset: {len(combined)} complete observations")

    return combined


# =============================================================================
# ANALYSIS & SUMMARY STATISTICS
# =============================================================================


def create_summary_stats(combined_df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for the aligned dataset.

    Args:
        combined_df: Aligned fashion-economy DataFrame

    Returns:
        Dictionary of summary statistics
    """
    if len(combined_df) == 0:
        logger.warning("Empty dataset - cannot create summary statistics")
        return {
            "n_months": 0,
            "date_range": "No data",
            "fashion_metrics": None,
            "economic_metrics": None,
            "correlations": None,
        }

    min_date = combined_df["date"].min()
    max_date = combined_df["date"].max()
    date_range_str = f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"

    fashion_cols = ["vibrancy", "darkness", "brightness", "color_diversity"]
    economic_cols = ["VIX", "SP500", "Unemployment"]

    # Select only columns that exist
    fashion_cols = [col for col in fashion_cols if col in combined_df.columns]
    economic_cols = [col for col in economic_cols if col in combined_df.columns]

    summary = {
        "n_months": len(combined_df),
        "date_range": date_range_str,
        "fashion_metrics": combined_df[fashion_cols].describe(),
        "economic_metrics": combined_df[economic_cols].describe(),
        "correlations": combined_df[["vibrancy"] + economic_cols].corr(),
    }

    return summary


# =============================================================================
# VISUALIZATION
# =============================================================================


def visualize_trends(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create comprehensive visualization of fashion-economy relationships.

    Args:
        combined_df: Aligned fashion-economy DataFrame
        output_dir: Directory to save figure
    """
    if len(combined_df) < 2:
        logger.warning("Not enough data points for visualization (need at least 2)")
        return

    logger.info("Creating visualizations...")

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Define COVID period for highlighting
    covid_start = pd.to_datetime("2020-03-01")
    covid_end = pd.to_datetime("2020-06-30")

    # Panel 1: Vibrancy vs VIX
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    ax1.plot(
        combined_df["date"],
        combined_df["vibrancy"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Vibrancy",
        color="#2E86AB",
    )
    ax1_twin.plot(
        combined_df["date"],
        combined_df["VIX"],
        marker="s",
        linewidth=2,
        markersize=4,
        label="VIX",
        color="#A23B72",
        linestyle="--",
    )

    ax1.set_ylabel("Fashion Vibrancy Score", fontsize=11, fontweight="bold")
    ax1_twin.set_ylabel("VIX (Market Volatility)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Fashion Vibrancy vs Market Volatility", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    if combined_df["date"].min() <= covid_start <= combined_df["date"].max():
        ax1.axvspan(covid_start, covid_end, alpha=0.2, color="red", label="COVID-19")

    # Panel 2: Vibrancy vs S&P 500
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(
        combined_df["date"],
        combined_df["vibrancy"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Vibrancy",
        color="#2E86AB",
    )
    ax2_twin.plot(
        combined_df["date"],
        combined_df["SP500"],
        marker="^",
        linewidth=2,
        markersize=4,
        label="S&P 500",
        color="#6A994E",
    )

    ax2.set_ylabel("Fashion Vibrancy Score", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("S&P 500 Index", fontsize=11, fontweight="bold")
    ax2.set_title("Fashion Vibrancy vs Stock Market", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    if combined_df["date"].min() <= covid_start <= combined_df["date"].max():
        ax2.axvspan(covid_start, covid_end, alpha=0.2, color="red")

    # Panel 3: Vibrancy vs Unemployment
    ax3 = axes[2]
    ax3_twin = ax3.twinx()

    ax3.plot(
        combined_df["date"],
        combined_df["vibrancy"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Vibrancy",
        color="#2E86AB",
    )
    ax3_twin.plot(
        combined_df["date"],
        combined_df["Unemployment"],
        marker="d",
        linewidth=2,
        markersize=4,
        label="Unemployment Rate",
        color="#F18F01",
    )

    ax3.set_ylabel("Fashion Vibrancy Score", fontsize=11, fontweight="bold")
    ax3_twin.set_ylabel("Unemployment Rate (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Fashion Vibrancy vs Labor Market", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    ax3.grid(alpha=0.3)

    if combined_df["date"].min() <= covid_start <= combined_df["date"].max():
        ax3.axvspan(covid_start, covid_end, alpha=0.2, color="red")

    # Panel 4: Correlation Heatmap
    ax4 = axes[3]
    corr = combined_df[["vibrancy", "VIX", "SP500", "Unemployment"]].corr()
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
    ax4.set_title(
        "Correlation Matrix: Fashion vs Economic Indicators",
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = output_dir / "fashion_economy_trends_3vars.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to: {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():
    """
    Main aggregation pipeline.
    """
    logger.info("=" * 70)
    logger.info("FASHION-ECONOMY DATA AGGREGATION PIPELINE")
    logger.info("Economic Variables: VIX, S&P 500, Unemployment Rate")
    logger.info("=" * 70)

    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    fashion_csv = (
        project_root / "data" / "processed_features" / "all_paris_features.csv"
    )
    economic_csv = project_root / "data" / "economic_data" / "economic_indicators.csv"
    output_dir = project_root / "data" / "processed_features"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load fashion data
        logger.info("\nStep 1: Loading fashion data...")
        fashion_df = load_fashion_data(fashion_csv)

        # Step 2: Aggregate by show
        logger.info("\nStep 2: Aggregating by show...")
        show_data = aggregate_by_show(fashion_df)
        show_csv = output_dir / "fashion_by_show.csv"
        show_data.to_csv(show_csv, index=False)
        logger.info(f"Saved show-level data to: {show_csv}")

        # Step 3: Industry-wide aggregation
        logger.info("\nStep 3: Creating industry-wide trends...")
        industry_data = aggregate_industry_wide(show_data)

        # Step 4: Monthly interpolation
        logger.info("\nStep 4: Interpolating to monthly frequency...")
        monthly_fashion = create_monthly_timeseries(industry_data)

        # Step 5: Load economic data
        logger.info("\nStep 5: Loading economic data...")
        economic_df = load_economic_data(economic_csv)

        if economic_df is None:
            logger.error("\nCRITICAL ERROR: Could not load economic data")
            logger.error("Please ensure economic_indicators.csv has columns:")
            logger.error("  - Date (month-end format)")
            logger.error("  - VIX")
            logger.error("  - SP500")
            logger.error("  - Unemployment")
            return None

        # Step 6: Align datasets
        logger.info("\nStep 6: Aligning fashion and economic data...")
        combined = align_fashion_and_economy(monthly_fashion, economic_df)

        if combined is None or len(combined) == 0:
            logger.error(
                "\nERROR: No overlapping dates between fashion and economic data"
            )
            return None

        # Save aligned data
        aligned_csv = output_dir / "fashion_economy_aligned.csv"
        combined.to_csv(aligned_csv, index=False)
        logger.info(f"Saved aligned data to: {aligned_csv}")

        # Step 7: Summary statistics
        logger.info("\nStep 7: Generating summary statistics...")
        summary = create_summary_stats(combined)

        logger.info(f"\nDate Range: {summary['date_range']}")
        logger.info(f"Total Months: {summary['n_months']}")
        logger.info("\nCorrelations:")
        logger.info(f"\n{summary['correlations']}")

        # Step 8: Visualize
        logger.info("\nStep 8: Creating visualizations...")
        visualize_trends(combined, output_dir)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nOutput files saved to: {output_dir}")
        logger.info("  - fashion_by_show.csv (show-level aggregation)")
        logger.info("  - fashion_economy_aligned.csv (final aligned dataset)")
        logger.info("  - fashion_economy_trends_3vars.png (visualization)")

        return combined

    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    df = main()
