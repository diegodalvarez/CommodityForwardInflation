# Commodity Forward Inflation

The overall goal of this repository is to examine the relationship between inflation and energy commodity futures and develop systematic trading strategies that can harvest the associated premium.

The research begins with 5-year, 5-year forward inflation expectations and is motivated by the relationship between crude oil prices and U.S. 5y5y forward inflation expectations documented in the [FRED Blog - Oil Prices and Expected Inflation](https://fredblog.stlouisfed.org/2020/04/oil-prices-and-expected-inflation/).

The analysis extends this relationship across multiple energy futures and inflation measures, including U.S. 5y5y forward inflation expectations and Bloomberg Economic Inflation Surprise Indexes. The futures universe includes:

`CL` `CO` `HO` `NG` `QS` `XB`

## Writeup

The full technical writeup contains the methodology, empirical results, portfolio construction, and robustness analysis.

| | PDF |
|---|---|
| Technical writeup containing methodology and results | <a href="CommodityForwardInflation.pdf">![PDF](https://img.icons8.com/ios-filled/50/000000/pdf.png)</a> |

## Strategies

Several trading approaches are investigated:

1. **Inflation Z-Score Strategies**  
   Energy futures are traded based directly on standardized inflation measures, with positions determined by the level and direction of the inflation signal.

2. **OLS Forecasted Return Strategies**  
   Inflation measures are used as explanatory variables in OLS regressions of energy futures returns. The resulting forecasts are used to generate trading signals and are evaluated using full-sample, train/test, and expanding out-of-sample methodologies.

3. **OLS Residual Strategies**  
   Regression residuals are interpreted as deviations from model-implied fair value, with positions taken in the expectation that these deviations revert.

4. **Optimized Decile Strategies**  
   Inflation signals and OLS residuals are sorted into deciles and used to construct more selective long/short strategies based on the historical performance of different signal levels.

## Research Results

The results show a consistent relationship between inflation measures and energy futures returns. The strongest performance generally comes from the inflation z-score and OLS forecast-based strategies.

The strategies are evaluated using:

- Full-sample / in-sample regressions
- Train/test analysis with a 2018 cutoff
- Expanding out-of-sample regressions
- Volatility-targeted returns
- Equal-weighted portfolio construction
- Sharpe ratio analysis
- Decile-based signal optimization

The resulting portfolios show reasonably strong risk-adjusted returns across multiple energy futures and inflation measures, with the OLS forecast-based portfolio producing Sharpe ratios around 1 across the main specifications.

## Data

The analysis uses data from several sources:

- **Federal Reserve Bank of St. Louis (FRED)** — 5y5y forward inflation expectations and energy spot prices
- **Bloomberg Terminal** — energy futures, 5y5y forward inflation, and Bloomberg Economic Inflation Surprise Indexes

Energy futures are constructed using front-month contracts and Bloomberg's ratio-based roll adjustment methodology.

## Repository Layout

```text
CommodityForwardInflation/
│
├── 0Background.ipynb
├── 1SignalStrategies.ipynb
├── 2CrossStrategies.ipynb
│
├── src/
│   ├── Backtesting.py
│   ├── CrossSectionBacktest.py
│   ├── FactorAnalysis.py
│   ├── PrepData.py
│   ├── SignalGenerator.py
│   ├── SignalOptimizer.py
│   └── run.py
│
└── README.md

