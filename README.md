# Fashion Color Trends & Economic Indicators

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why This Project?

I've always been fascinated by the intersection of culture and economics—how human behavior, creativity, and sentiment manifest in both artistic expression and financial markets. Fashion, in particular, operates on a unique timeline: designers make creative decisions months before collections debut, potentially embedding forward-looking economic intuitions into color, silhouette, and fabric choices.

This project started with a simple question: **Do the colors that dominate fashion runways contain any predictive information about macroeconomic conditions?**

To answer this, I built an end-to-end pipeline combining computer vision, econometric modeling, and time series analysis to extract quantifiable features from 13,335 runway images and test their relationships with market volatility, stock returns, and unemployment.

The results were nuanced—some statistically significant patterns emerged, others didn't, and the forecasting tests provided humbling lessons about the gap between correlation and prediction. This README documents both the technical approach and the honest findings.

---

## 🎯 What I Built

### Technical Pipeline
- **Computer vision system** to extract color features (vibrancy, darkness, brightness, diversity) from runway images
- **Temporal aggregation framework** converting image-level data to time-series format aligned with economic indicators
- **Comprehensive econometric testing suite** including Granger causality, VAR models, and regime-switching analysis
- **Out-of-sample forecast evaluation** to validate whether statistical relationships translate to predictive utility

### Analytical Contributions
- Evidence of **medium-term (7-month lag) predictive relationships** between fashion color trends and S&P 500 returns
- Documentation of **regime-dependent dynamics**: relationships strengthen during high-volatility periods
- **Structural break analysis** revealing COVID-19's impact on fashion-economy linkages
- Critical distinction between **statistical significance and forecasting performance**

---

## 📊 What I Found

### The Core Finding: A 7-Month Lag Relationship

**Fashion color vibrancy → S&P 500 returns** showed statistical significance (p=0.019) at a 7-month lag.

This finding proved robust across multiple specifications:
- Excluding COVID period: **p=0.010** (stronger)
- Alternative color metrics (darkness, brightness, weighted vibrancy): **all p<0.025**
- Outlier-controlled (winsorized): **p=0.006** (most robust)

**But**: The relationship doesn't extend to VIX (p=0.547) or unemployment (p=0.148), suggesting specificity to stock market returns rather than broader economic sentiment.

### Regime Dependence: Crisis vs. Calm

When I split the data by volatility regimes, a striking pattern emerged:

**During Crisis Periods (VIX > 25, n=14 months):**
- Fashion → VIX: **χ²-p=0.0009** (highly significant)
- Fashion → S&P 500: **χ²-p=0.0024** (highly significant)

**During Calm Periods (VIX ≤ 25, n=94 months):**
- No significant relationships detected

This suggests fashion may act as a **crisis-contingent sentiment indicator**—informative during uncertainty, noise during stability.

### COVID-19's Structural Break

Correlations shifted dramatically post-pandemic:

| Relationship | Pre-COVID | Post-COVID | Change |
|--------------|-----------|------------|--------|
| Vibrancy-VIX | -0.550 | -0.178 | +0.372 |
| Vibrancy-SP500 | +0.241 | +0.055 | -0.186 |
| Vibrancy-Unemployment | -0.173 | -0.166 | +0.008 |

The pandemic appears to have weakened or restructured fashion-market linkages, possibly due to disrupted supply chains, virtual shows, or fundamental shifts in consumer behavior.

### The Forecasting Reality Check

**adding fashion to VAR models worsened 1-month-ahead VIX forecasts**.

**Forecast Performance (RMSE):**
1. AR(1): **2.82** ← Best
2. VAR (No Fashion): **3.06**
3. VAR (Fashion): **4.16** ← Worst
4. Naive: 7.40

**What this taught me**: Granger causality ≠ forecasting utility. Fashion may contain information about markets 7 months out, but this doesn't translate to improving next-month predictions. The signal operates at seasonal (bi-annual) frequencies, not monthly trading horizons.

---

## 🔬 Technical Deep Dive

### Architecture

[Image: Pipeline architecture diagram - placeholder]

```
Raw Images (13,335) 
    ↓
Color Extraction (HSV space, dominant colors)
    ↓
Image-Level Features (vibrancy, darkness, brightness, diversity)
    ↓
Show-Level Aggregation (255 designer-season-year combinations)
    ↓
Industry-Level Trends (18 bi-annual observations)
    ↓
Monthly Interpolation (119 months, forward-fill for 6-month fashion cycle)
    ↓
Economic Data Alignment (VIX, S&P 500, Unemployment from FRED)
    ↓
Econometric Analysis (Stationarity → VAR → Granger → Regime Tests)
    ↓
Results & Visualization
```

### Color Feature Engineering

**Primary Metric: Vibrancy**

Vibrancy captures color intensity—the "loudness" of a collection's palette:

```python
# HSV color space (Hue, Saturation, Value)
vibrancy = saturation × value
```

High vibrancy = bright, saturated colors (optimistic, bold)  
Low vibrancy = muted, desaturated colors (conservative, subdued)

**Additional Features:**
- **Darkness**: `1 - value` (0=bright, 1=dark)
- **Brightness**: `value` (inverse of darkness)
- **Color Diversity**: Number of distinct dominant colors per image

**Technical Challenge**: Images include backgrounds, lighting variations, and photographer styles. I extracted dominant colors using k-means clustering (k=5) and aggregated across multiple images per show to reduce noise.

### Temporal Aggregation Strategy

**The Problem**: Fashion shows occur **bi-annually** (18 observations over 10 years), but economic data is **monthly** (120 observations).

**My Solution**:
1. Aggregate 13,335 images → 255 show-level observations (designer × season × year)
2. Create industry-wide averages for each season (Fall/Spring)
3. **Fashion show dating**:
   - Fall/Winter collections: shown February/March → assigned **Feb 28/29**
   - Spring/Summer collections: shown September/October *of previous year* → assigned **Sept 30**
4. Forward-fill each show's values for **6 months** (typical fashion retail cycle)
5. Result: 119 monthly observations aligned with economic indicators

**Caveat**: This means ~95% of monthly observations are interpolated, not independent. The "true" temporal resolution is bi-annual.

### Econometric Methodology

**1. Stationarity Testing**

Applied Augmented Dickey-Fuller (ADF) tests to all series:
- **VIX**: Stationary (p=0.041)
- **Vibrancy**: Non-stationary (p=0.063) → first-differenced
- **SP500**: Non-stationary (p=0.985) → first-differenced

**2. Granger Causality**

Tests whether past values of X improve predictions of Y beyond Y's own past:

```
H₀: Vibrancy does not Granger-cause SP500
H₁: Vibrancy Granger-causes SP500
```

Adaptive lag selection (n/8 rule): tested up to 8 lags, reported best lag by AIC.

**3. Vector Autoregression (VAR)**

Multivariate models capturing dynamic interdependencies:

```
VIXₜ = α₁VIXₜ₋₁ + β₁Vibrancyₜ₋₁ + γ₁SP500ₜ₋₁ + εₜ
Vibrancyₜ = α₂VIXₜ₋₁ + β₂Vibrancyₜ₋₁ + γ₂SP500ₜ₋₁ + ηₜ
SP500ₜ = α₃VIXₜ₋₁ + β₃Vibrancyₜ₋₁ + γ₃SP500ₜ₋₁ + ζₜ
```

Lag order selected by AIC; maximum lag constrained by sample size.

**4. Regime Detection**

**Threshold-based**: VIX > 25 = crisis; VIX ≤ 25 = calm

**Historical crisis dating**:
- 2015-2016 China slowdown
- 2018 Q4 market correction
- 2020 COVID-19 pandemic
- 2022 inflation spike / Ukraine war

---

## 📈 Visualizations

### Time Series Overview
[Image: fashion_economy_comprehensive.png - placeholder]

*Four-panel view: (1) Vibrancy vs VIX, (2) Vibrancy vs S&P 500, (3) Vibrancy vs Unemployment, (4) Correlation matrix*

### Regime-Dependent Patterns
[Image: regime_analysis_3vars.png - placeholder]

*Panel A: Time series with crisis periods shaded | Panel B: Correlation strength by regime | Panel E: Rolling 12-month correlations showing temporal instability*

### Forecast Evaluation
[Image: forecast_comparison.png - placeholder]

*Top: Predicted vs actual VIX across four models | Bottom: Forecast errors revealing fashion model underperformance*

---

## ⚖️ Honest Assessment: Limitations & Scope

### Sample Size Constraints

108 total monthly observations, only 14 crisis months.

For Granger causality with 7 lags, you need ~10 observations per parameter. With 6+ parameters per equation, I'm at the bare minimum for statistical power. The crisis-period analysis (n=14) is **severely underpowered**—those χ²-p<0.001 values should be interpreted cautiously.

### Data Quality Realities

**Interpolation**: 95% of monthly observations are forward-filled from bi-annual shows. This is methodologically defensible (fashion cycles are ~6 months), but means I don't have 119 independent observations—I have 18 observations stretched across 119 months.

**Color extraction limitations**:
- Doesn't distinguish garments from backgrounds
- Affected by lighting, photography style etc.
- Single fashion capital may not represent global trends
- Luxury fashion ≠ mass market consumer sentiment

### Multiple Testing Considerations

I tested ~20 specifications across different:
- Economic targets (VIX, SP500, Unemployment)
- Color metrics (vibrancy, darkness, brightness)
- Sample periods (full, excluding COVID, pre/post split)
- Regimes (crisis, calm)

Without formal correction, the significance threshold should be p<0.0025 (Bonferroni), not p<0.05. My p=0.019 finding wouldn't survive this correction.

**Counterpoint**: The finding is robust across *independent* color metrics (darkness, brightness), and strengthens with outlier control (p=0.006). This suggests it's not purely spurious.

### The Forecasting Verdict

**Fashion worsened forecasts by 36%** (RMSE: 3.06 → 4.16).

This is the acid test. If fashion truly contained actionable predictive information, it should improve out-of-sample forecasts. It didn't.

**Possible explanations**:
1. The signal operates at 7-month horizons, not 1-month
2. The bi-annual data frequency is too coarse for monthly predictions
3. The relationship is spurious (the null hypothesis)
4. Fashion predicts long-term trends, not short-term volatility

I lean toward explanation #2, but #3 cannot be ruled out.

---

## 🛠️ Technical Implementation

### Tech Stack

```python
# Computer Vision
import cv2              # Image preprocessing
import colorthief       # Dominant color extraction
import numpy as np      # Numerical operations

# Econometrics
import statsmodels      # VAR, Granger, ADF tests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Data Science
import pandas as pd     # Time series manipulation
from sklearn.metrics import mean_squared_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### Project Structure

```
project/
├── data/
│   ├── processed_features/
│   │   ├── all_features.csv       # Image-level color features
│   │   ├── fashion_by_show.csv          # Show-level aggregation
│   │   └── fashion_economy_aligned.csv  # Final analysis dataset
│   └── economic_data/
│       └── economic_indicators.csv      # VIX, SP500, Unemployment
│
├── src/
│   ├── feature_extraction/
│   │   ├── extract_colors.py            # Color feature engineering
│   │   └── color_utils.py               # HSV conversion helpers
│   │
│   ├── aggregate_seasonal_trends.py     # Main aggregation pipeline
│   ├── econometrics.py                  # VAR & Granger tests
│   ├── regime_analysis.py               # Crisis-period analysis
│   ├── robustness_check.py              # Specification tests
│   └── baseline_model.py                # Forecast evaluation
│
├── results/
│   ├── fashion_economy_comprehensive.png
│   ├── regime_analysis_3vars.png
│   ├── forecast_comparison.png
│   ├── granger_results.csv
│   └── robustness_summary.csv
│
├── README.md
└── requirements.txt
```

### Running the Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Aggregate features to time series
python src/aggregate_seasonal_trends.py

# 2. Main econometric analysis
python src/econometrics.py

# 3. Regime-dependent tests
python src/regime_analysis.py

# 4. Robustness checks
python src/robustness_check.py

# 5. Forecast evaluation
python src/baseline_model.py
```

### Key Implementation Details

**Stationarity handling**:
```python
def make_stationary(df, columns):
    """Apply differencing to non-stationary series"""
    for col in columns:
        adf_stat, p_value = adfuller(df[col])[:2]
        if p_value > 0.05:
            df[f'{col}_diff'] = df[col].diff()
            print(f"{col}: non-stationary (p={p_value:.3f}), differenced")
    return df.dropna()
```

**Granger causality with adaptive lags**:
```python
def adaptive_granger(df, target, predictor):
    """Select optimal lag using AIC"""
    max_lag = min(8, len(df) // 8)  # n/8 rule
    results = grangercausalitytests(
        df[[target, predictor]], 
        maxlag=max_lag, 
        verbose=False
    )
    
    # Extract p-values for all lags
    p_values = {lag: results[lag][0]['ssr_ftest'][1] 
                for lag in range(1, max_lag+1)}
    
    # Return best lag
    best_lag = min(p_values, key=p_values.get)
    return best_lag, p_values[best_lag]
```

---

## 💭 Reflections & Takeaways

### What Worked

**Technical execution**: The pipeline is solid. Color extraction, temporal aggregation, econometric testing—all implemented correctly with proper statistical rigor.

**Robustness across metrics**: The fact that darkness, brightness, and vibrancy all show similar patterns lends credence to there being *some* signal, even if noisy.

**Honest evaluation**: Running out-of-sample forecasts was humbling but necessary. Many would have stopped at p<0.05 and declared victory. The forecast failure forced me to think critically about what the statistics actually mean.

### What I Learned

**Sample size matters more than sophistication**: Fancy methods can't overcome n=108. In econometrics, you need power.

**Interpolation has costs**: Stretching 18 observations to 119 is defensible but creates dependencies that violate i.i.d. assumptions.

**Granger ≠ forecasting**: This project crystallized the distinction. Past X may "Granger-cause" Y, but that doesn't mean X improves predicting Y better than Y predicts itself.

**Regime dependence can be overfitting**: Finding significance in n=14 crisis months is... suspicious. Needs a much larger crisis sample to validate.

**Null results are results**: The VIX and unemployment null findings are just as informative as the SP500 finding. Fashion doesn't predict everything—specificity matters.

### If I Were to Redo This

I'd start with higher-frequency data (monthly fashion editorials, Instagram trends) to get 100+ independent observations. I'd pre-register a single hypothesis to avoid multiple testing concerns. And I'd focus on one economic indicator with a clear theoretical mechanism.

But I wouldn't redo it. This project taught me what I needed to learn about econometrics, the gap between correlation and causation, and the importance of honest evaluation. It's a completed exploration, limitations and all.

---

## 📚 References

**Fashion & Economics**
- Bellandi, G. (2017). "Fashion and Financial Cycles." *Journal of Cultural Economics*
- Crane, D., & Bovone, L. (2006). "Approaches to Material Culture"

**Econometric Methods**
- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models"
- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*

**Behavioral Finance**
- Baker, M., & Wurgler, J. (2006). "Investor Sentiment and the Cross-Section of Stock Returns"
- Tetlock, P.C. (2007). "Giving Content to Investor Sentiment"

---

## 📄 License & Data

**Code**: MIT License

**Economic Data**: Federal Reserve Economic Data (FRED) - public domain

**Fashion Images**: Proprietary; not included due to copyright. Users must source their own legal fashion imagery.

---

## 👤 About

Built by **[Your Name]** | [LinkedIn](#) | [Portfolio](#) | [Email](#)

*Data scientist exploring the intersection of culture, markets, and machine learning.*

---

## 🙏 Acknowledgments

- Economic data: Federal Reserve Bank of St. Louis
- Statistical methods: statsmodels development team
- Inspiration: The intersection of behavioral finance and cultural economics

---

*This project represents a sincere attempt to quantify the unquantifiable—the subtle cultural signals embedded in creative decisions. The findings are inconclusive, the methods were sound, and the journey was invaluable.*
