# Commodity Forward Inflation
This repo examines a series of strategies based on trading energy futures based on inflation measurements, specifically 5y5y forward inflation. The motivation for this project is based on the modelled relationship between Crude prices and US 5y5y forward inflation from *[The FRED Blog - Oil prices and expected inflation](https://fredblog.stlouisfed.org/2020/04/oil-prices-and-expected-inflation/?utm_source=series_page&utm_medium=related_content&utm_term=related_resources&utm_campaign=fredblog)*. ```CL``` ```CO``` ```HO``` ```NG``` ```QS``` ```XB```

This repo contains various strategies trading energy-based futures based on inflation the following strategies are implemented. 

Combined portfolio is 1.4 sharpe which consists of the following

1. Trading Z-Scores of 5y5y US Inflation Forwards - this model trades energy futures using the z-score. (In sample 1.2 out of sample 1.1)
2. Cross-Section Residual Model (~2 sharpe) - this model trades energy futures cross-sectionally based on the 5y5y forward inflation OLS model

## Writeup
|         | PDF          |
|---------|---------------------|
| Technical writeup containing methodology & results | <a href="CommodityForwardInflation.pdf">![PDF](https://img.icons8.com/ios-filled/50/000000/pdf.png)</a> |