# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:11:25 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class SignalStrategies:
    
    def __init__(self) -> pd.DataFrame: 
        
        self.path      = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        self.sig_path  = os.path.join(self.data_path, "Signals")
        
        self.slice_year = 2018
        
    def get_raw_factor(self, verbose: bool = True) -> pd.DataFrame: 
        
        if verbose: print("Getting Raw Signal")
        
        out_path      = os.path.join(self.sig_path, "RawFactor.parquet")
        inf_path      = os.path.join(self.data_path, "InflationMeasures")
        forward_path  = os.path.join(inf_path, "InflationForward.parquet")
        surprise_path = os.path.join(inf_path, "InflationSurprise.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_forward = (pd.read_parquet(
            path = forward_path, engine = "pyarrow").
            drop(columns = ["security"]).
            pivot(index = "date", columns = "country", values = "value").
            apply(lambda x: np.log(x).diff()).
            reset_index().
            melt(id_vars = "date").
            assign(group = "forward_inf"))
        
        df_surprise = (pd.read_parquet(
            path = surprise_path, engine = "pyarrow").
            pivot(index = "date", columns = "country", values = "value").
            diff().
            reset_index().
            melt(id_vars = "date").
            assign(group = "surprise_inf"))
        
        df_factor = (pd.concat([
            df_forward, df_surprise]).
            dropna())
        
        if verbose: print("Saving data\n")
        df_factor.to_parquet(path = out_path, engine = "pyarrow")
    
    def get_ols(self, df: pd.DataFrame, slice_year: int = 2018, min_obs: int = 30) -> dict: 
        
        full_sample_model = (sm
                .OLS(
                    endog = df.vol_rtn,
                    exog  = sm.add_constant(df.z_score))
                .fit())
        
        df_tmp = (df
                .reset_index()
                .assign(year = lambda x: x.date.dt.year)
                .set_index("date"))
        
        df_insample  = df_tmp.loc[lambda x: x.year <= slice_year]
        
        insample_model = (sm
                .OLS(
                    endog = df_insample.vol_rtn,
                    exog  = sm.add_constant(df_insample.z_score))
                .fit())
        
        rolling_model = (RollingOLS(
            endog     = df.vol_rtn,
            exog      = sm.add_constant(df.z_score),
            min_nobs  = 30,
            expanding = True)
            .fit())
        
        out_dict = {
            "full_sample"    : full_sample_model,
            "in_sample"      : insample_model,
            "expanding_model": rolling_model}
        
        return out_dict
    
    def get_ols_resid_regression(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting OLS Regression Residuals In-Sample & Out-Sample")
        out_path  = os.path.join(self.sig_path, "OLSResidModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data collected\n")
            return None
        
        fut_path = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        df_rtn   = (pd
                .read_parquet(path = fut_path, engine = "pyarrow")
                .drop(columns = ["lag_weight"])
                .assign(vol_rtn = lambda x: x.weight * x.rtn)
                .drop(columns = ["rtn", "weight"])
                .dropna())
        
        sig_path  = os.path.join(self.sig_path, "ZScore.parquet")
        df_zscore = pd.read_parquet(path = sig_path, engine = "pyarrow")
        
        df_combined = (df_rtn
                .merge(right = df_zscore, how = "inner", on = ["date"])
                .assign(group_var = lambda x: x.security + " " + x.country + " " + x.group)
                .set_index("date")
                .rename(columns = {"value": "z_score"}))
        
        groups   = df_combined.group_var.drop_duplicates().sort_values().to_list()
        out_dict = {}
        
        for group in groups: 
            
            df_input = (df_combined
                    .loc[lambda x: x.group_var == group])
            
            models          = self.get_ols(df_input, self.slice_year)
            out_dict[group] = models
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)
            
    def get_ols_pred_regression(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting OLS Regression Predicted In-Sample & Out-Sample")
        out_path  = os.path.join(self.sig_path, "OLSPredModels.pkl")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data collected")
            return None
        
        fut_path = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        df_rtn   = (pd
                .read_parquet(path = fut_path, engine = "pyarrow")
                .drop(columns = ["lag_weight"])
                .assign(vol_rtn = lambda x: x.weight * x.rtn)
                .drop(columns = ["rtn", "weight"])
                .dropna())
        
        sig_path  = os.path.join(self.sig_path, "ZScore.parquet")
        df_zscore = (pd
                     .read_parquet(path = sig_path, engine = "pyarrow")
                     .set_index("date")
                     .groupby(["country", "group"])
                     .apply(lambda x: x.z_score.shift())
                     .to_frame(name = "z_score")
                     .reset_index()
                     .dropna())
        
        df_combined = (df_rtn
                .merge(right = df_zscore, how = "inner", on = ["date"])
                .assign(group_var = lambda x: x.security + " " + x.country + " " + x.group)
                .set_index("date")
                .rename(columns = {"value": "z_score"}))
        
        groups   = df_combined.group_var.drop_duplicates().sort_values().to_list()
        out_dict = {}
        
        for group in groups: 
            
            df_input = (df_combined
                    .loc[lambda x: x.group_var == group])
            
            models          = self.get_ols(df_input)
            out_dict[group] = models
            
        if verbose: print("Saving data\n")
        with open(out_path, "wb") as f:
            pickle.dump(out_dict, f)
        
    def get_zscore(self, window: int = 30, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Z-Score Values")
        out_path = os.path.join(self.data_path, "Signals", "ZScore.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data collected\n")
            return None
        
        z_path = os.path.join(self.sig_path, "RawFactor.parquet")
        df_out = (pd
              .read_parquet(path = z_path, engine = "pyarrow").
              pivot(index = "date", columns = ["country", "group"], values = "value")
              .apply(
                  lambda x: (x - x.ewm(span = window, adjust = False).mean()) / 
                  x.ewm(span = window, adjust = False).std())
              .reset_index()
              .melt(id_vars = [("date", "")])
              .rename(columns = {("date", ""): "date"})
              .dropna()
              .rename(columns = {"value": "z_score"}))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    signal_strategies = SignalStrategies()
    signal_strategies.get_raw_factor()
    signal_strategies.get_zscore()
    signal_strategies.get_ols_resid_regression()
    signal_strategies.get_ols_pred_regression()
    
if __name__ == "__main__": main()