# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:41:24 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
tqdm.pandas()

class CrossSectionBacktest:
    
    def __init__(self) -> None:
        
        self.src_path   = os.getcwd()
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        self.cross_path = os.path.join(self.data_path, "CrossSectional")
        
        if not os.path.exists(self.cross_path):
            os.makedirs(self.cross_path)
            
        self.q = 2
            
    def _get_leg(self, df: pd.DataFrame, q: int = 2) -> pd.DataFrame:
        
        df_out = (df
                .pivot(index = "security", columns = "date", values = "lag_resid")
                .apply(lambda x: pd.qcut(x = x, q = q, labels = ["LowerGroup", "UpperGroup"]))
                .reset_index()
                .melt(id_vars = "security", value_name = "group"))
        
        return df_out
        
    def get_residual_legs(self, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Getting Cross-Sectional Residual Backtest")
        
        resid_path = os.path.join(self.data_path, "Backtests", "OLSResidualBacktest.parquet")
        rtn_path   = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        out_path   = os.path.join(self.data_path, "Backtests", "CrossectionaLOLSResidual.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_raw = (pd
                .read_parquet(path = resid_path, engine = "pyarrow")
                .drop(columns = ["resid", "signal_rtn", "vol_rtn", "name", "group"])
            .assign(group_var = lambda x: 
                    x.target + " " +  
                    x.regression + " " + 
                    x.sample_group + " " + 
                    x.country + " " + 
                    x.signal_name))
        
        df_date_selector = (df_raw
                .drop(columns = [
                    "sample_group", "regression", "fut_rtn", "target", 
                    "lag_resid", "group_var"])
                .drop_duplicates()
                .groupby(["date", "country", "signal_name"])
                .agg("count")
                .reset_index()
                .loc[lambda x: x.security >= 4]
                .drop(columns = ["security"]))
        
        df_namer = (df_raw
                [[
                    "group_var", "target", "regression", "sample_group", 
                    "country", "signal_name"]]
                .drop_duplicates())
        
        df_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .assign(
                    lagged  = lambda x: x.weight * x.rtn,
                    perfect = lambda x: x.lag_weight * x.rtn)
                [["date", "security", "lagged", "perfect"]]
                .melt(
                    id_vars    = ["date", "security"],
                    var_name   = "target",
                    value_name = "rtn")
                .assign(date = lambda x: pd.to_datetime(x.date)))
        
        df_out = (df_raw
                .merge(
                    right = df_date_selector, 
                    how   = "inner", 
                    on    = df_date_selector.columns.to_list())
                #.loc[lambda x: x.group_var == x.group_var.min()]
                .groupby("group_var")
                .apply(self._get_leg, self.q)
                .reset_index()
                .merge(right = df_namer, how = "inner", on = ["group_var"])
                .drop(columns = ["group_var", "level_1"])
                .assign(date = lambda x: pd.to_datetime(x.date))
                .merge(right = df_rtn, how = "inner", on = ["date", "security", "target"]))
        
        if verbose: print("Saving data")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 

    cross_backtest  = CrossSectionBacktest()
    cross_backtest.get_residual_legs()
    
if __name__ == "__main__":main()