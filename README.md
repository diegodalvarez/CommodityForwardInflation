# Commodity Forward Inflation
This FRED article *[The FRED Blog - Oil prices and expected inflation](https://fredblog.stlouisfed.org/2020/04/oil-prices-and-expected-inflation/?utm_source=series_page&utm_medium=related_content&utm_term=related_resources&utm_campaign=fredblog)* show an interesting graph that links the relationship between 5y5y forward inflation expectations to spot crude prices. This repo is built on top of a similar model, but instead uses the 5y5y forward inflation rate (for both UK and US) and the risk-adjusted roll-adjusted 10% volatility targetted commodity futures. At the moment this model uses energy futures solely ```CL``` ```CO``` ```HO``` ```NG``` ```QS``` ```XB```. 

Below are the correlations of returns 
![image](https://github.com/user-attachments/assets/4df6b007-217e-4a3c-afe7-7b607071ac66)

The expanding out-of-sample returns
![image](https://github.com/user-attachments/assets/898cdbd5-f9df-4c37-b26a-3953fe9f47ba)

The sharpes of each security
![image](https://github.com/user-attachments/assets/efe8f323-f12e-4bd4-be84-e1294c20f8cb)

Equal weight portfolio returns
![image](https://github.com/user-attachments/assets/127ac2d0-ef2d-49aa-92aa-d0b586559bec)

Annualized Sharpe
![image](https://github.com/user-attachments/assets/2f424f1c-f671-454e-8688-dd74de056d13)

The comparison of sharpe to other commodity benchmarks
![image](https://github.com/user-attachments/assets/b0202aaf-bacb-4d72-9e44-4b40311e2bab)

OLS Regression
![image](https://github.com/user-attachments/assets/a0bd7455-cece-49b2-b6b3-6c399c5e2cb4)
