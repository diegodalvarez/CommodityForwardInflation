# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:40:51 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

class FactorModel:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.fact_path = os.path.join(self.data_path, "FactorData")
        
        if not os.path.exists(self.fact_path):
            os.makedirs(self.fact_path)
        
        self.vol_adj_window = 10
    
    def _prep_fred_data(self) -> pd.DataFrame: 
        
        path   = os.path.join(self.data_path, "FRED", "CombinedData.parquet")
        df_raw = pd.read_parquet(path = path, engine = "pyarrow")
        
        df_adj = (df_raw
            .loc[lambda x: x.group == "spot_price"]
            .pivot(index = "date", columns = "ticker", values = "value")
            .diff()
            .apply(lambda x: x / x.ewm(span = self.vol_adj_window, adjust = False).std())
            .reset_index()
            .melt(id_vars = "date", value_name = "spot_radj")
            .dropna())
        
        df_inf = (df_raw
            .loc[lambda x: x.group == "inflation"]
            .drop(columns = ["group", "ticker"])
            .rename(columns = {"value": "forward_inf"}))
        
        df_combined = (df_inf
            .set_index("date")
            .apply(lambda x: np.log(x).diff())
            .merge(right = df_adj, how = "inner", on = ["date"])
            .set_index("date"))
        
        return df_combined
    
    def linear_spot_fred_models(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Linear Spot Models")
        
        out_path = os.path.join(self.fact_path, "SpotLinearModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_combined = self._prep_fred_data()
        tickers = df_combined.ticker.drop_duplicates().sort_values().to_list()
        models  = {}
        
        for ticker in tickers: 
        
            df_tmp = (df_combined
                .loc[lambda x: x.ticker == ticker]
                .assign(lag_inf = lambda x: x.forward_inf.shift())
                .dropna())
        
            model = (sm
                .OLS(
                    endog = df_tmp.spot_radj,
                    exog  = sm.add_constant(df_tmp.forward_inf))
                .fit())
        
            lag_model = (sm
                .OLS(
                    endog = df_tmp.spot_radj,
                    exog  = sm.add_constant(df_tmp.lag_inf))
                .fit(cov_type = "HC1"))
        
            models[ticker + "_lag0"] = model
            models[ticker + "_lag1"] = lag_model
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(models, f)
            
    def logit_spot_fred_models(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Logistic Spot Models")
        
        out_path = os.path.join(self.fact_path, "SpotLogitModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_combined = self._prep_fred_data()
        tickers = df_combined.ticker.drop_duplicates().sort_values().to_list()
        models  = {}
        
        for ticker in tickers:
        
            df_tmp = (
                df_combined
                .loc[lambda x: x.ticker == ticker]
                .assign(
                    lag_inf=lambda x: x.forward_inf.shift(),
                    direction=lambda x: (x.spot_radj > 0).astype(int)
                )
                .dropna()
            )
        
            # -------------------------
            # Lag 0
            # -------------------------
        
            model = (
                sm.Logit(
                    endog=df_tmp["direction"],
                    exog=sm.add_constant(
                        df_tmp[["forward_inf"]]
                    )
                )
                .fit(disp=False)
            )
        
            # -------------------------
            # Lag 1
            # -------------------------
        
            lag_model = (
                sm.Logit(
                    endog=df_tmp["direction"],
                    exog=sm.add_constant(
                        df_tmp[["lag_inf"]]
                    )
                )
                .fit(disp=False)
            )
        
            models[ticker + "_lag0"] = model
            models[ticker + "_lag1"] = lag_model
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(models, f)
            
    def get_param(self, models: dict, model_type: str) -> pd.DataFrame: 
            
        df_list = []
        
        if model_type == "OLS": 
            
            for key in models.keys():
                
                df_params = (
                    models[key]
                    .params
                    .to_frame(name="coef")
                    .reset_index()
                )
                
                df_pvalues = (
                    models[key]
                    .pvalues
                    .to_frame(name="pvalue")
                    .reset_index()
                )
                
                df_tvalues = (
                    models[key]
                    .tvalues
                    .to_frame(name="tvalue")
                    .reset_index()
                )
                
                df_add = (
                    df_params
                    .merge(
                        right=df_pvalues,
                        how="inner",
                        on=["index"]
                    )
                    .merge(
                        right=df_tvalues,
                        how="inner",
                        on=["index"]
                    )
                    .assign(
                        r2=models[key].rsquared,
                        name=key
                    )
                    .rename(
                        columns={"index": "param_name"}
                    )
                )
                
                df_list.append(df_add)
                
        elif model_type == "Logit":
            
            for key in models.keys():
                
                df_params = (
                    models[key]
                    .params
                    .to_frame(name="coef")
                    .reset_index()
                )
                
                df_pvalues = (
                    models[key]
                    .pvalues
                    .to_frame(name="pvalue")
                    .reset_index()
                )
                
                df_zvalues = (
                    models[key]
                    .tvalues
                    .to_frame(name="zvalue")
                    .reset_index()
                )
                
                df_add = (
                    df_params
                    .merge(
                        right=df_pvalues,
                        how="inner",
                        on=["index"]
                    )
                    .merge(
                        right=df_zvalues,
                        how="inner",
                        on=["index"]
                    )
                    .assign(
                        pseudo_r2=models[key].prsquared,
                        name=key
                    )
                    .rename(
                        columns={"index": "param_name"}
                    )
                )
                
                df_list.append(df_add)
                
        else:
            raise ValueError(
                f"Unknown model_type: {model_type}"
            )
        
        df_params = pd.concat(
            df_list,
            ignore_index=True
        )
        
        return df_params
    
    def _get_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df
                .sort_index()
                .assign(
                    diff_val = lambda x: x.value.diff(),
                    lag_diff = lambda x: x.diff_val.shift())
                .dropna())
        
        return df_out
    
    def inflation_measure_factor(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Inflation Signal Factor")
        
        vol_path = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        inf_path = os.path.join(self.data_path, "InflationMeasures")
        out_path = os.path.join(self.data_path, "FactorData", "InfMeasureDiffModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Saving data\n")
            return None
        
        df_vol_rtn = (pd
                .read_parquet(path = vol_path, engine = "pyarrow")
                .melt(
                    id_vars    = ["date", "security", "rtn"],
                    var_name   = "hedge_type",
                    value_name = "hedge_val")
                .dropna()
                .assign(vol_rtn = lambda x: x.rtn * x.hedge_val)
                .drop(columns = ["rtn"]) 
                .assign(hedge_type = lambda x: np.where(x.hedge_type == "weight", "perfect", "lagged"))
                .rename(columns = {"security": "fut_ticker"})
                .drop(columns = ["hedge_val"]))
        
        df_inf = (pd
                .read_parquet(path = inf_path, engine = "pyarrow")
                .set_index("date")
                .assign(security = lambda x: x.security.str.split(" ").str[0])
                .groupby("security")
                .apply(self._get_diff)
                .reset_index())
        
        df_combined = (df_inf
                .drop(columns = ["value"])
                .melt(
                    id_vars    = ["date", "security", "country"],
                    var_name   = "inf_type",
                    value_name = "inf_val")
                .rename(columns = {"security": "inf_ticker"})
                .merge(right = df_vol_rtn, how = "inner", on = ["date"])
                .assign(name = lambda x: x.inf_ticker + " " + x.inf_type + " " + x.hedge_type  + " " + x.fut_ticker))
        
        names  = df_combined.name.drop_duplicates().sort_values().to_list()
        models = {}

        for name in names: 
            
            df_tmp = (df_combined
                    .loc[lambda x: x.name == name]
                    .set_index("date"))
            
            model = (sm
                    .OLS(
                        endog = df_tmp.vol_rtn,
                        exog  = sm.add_constant(df_tmp.inf_val))
                    .fit())
            
            models[name] = model
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f: pickle.dump(models, f)
        
    def get_inflation_measure_model_param(self, models: dict, model_names: list) -> pd.DataFrame: 
        
        df_lists = []
        for model_name in model_names: 
        
            tmp_model = models[model_name]
            df_params = (tmp_model
                .params
                .to_frame(name = "param_val")
                .reset_index())
        
            df_pvalue = (tmp_model
                .pvalues
                .to_frame(name = "pvalue")
                .reset_index())
        
            df_tvalue = (tmp_model
                .tvalues
                .to_frame(name = "tvalue")
                .reset_index())
        
            df_add = (df_params
                .merge(right = df_pvalue, how = "inner", on = ["index"])
                .merge(right = df_tvalue, how = "inner", on = ["index"])
                .rename(columns = {"index": "param"})
                .assign(name = model_name))
            
            df_lists.append(df_add)
        
        df_params = pd.concat(df_lists)
        
        return df_params
        
def main() -> None: 
    
    factor_model = FactorModel()
    #factor_model.linear_spot_fred_models()
    #factor_model.logit_spot_fred_models()
    factor_model.inflation_measure_factor()
    
if __name__ == "__main__": main()