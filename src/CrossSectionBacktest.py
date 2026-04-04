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
        
        self.energy_tickers = ["CL", "CO", "HO", "NG", "QS", "XB"]
        self.inf_tickers    = ["BCMPGBIF", "BCMPUSIF"]
        self.data_path      = os.path.join(os.getcwd(), "CrossSectionalBacktests")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        
    def _get_inf(self, path: str) -> pd.DataFrame:
        
        inf_paths = [os.path.join(path, ticker + ".parquet") for ticker in self.inf_tickers]
        df_inf    = (pd.read_parquet(
            path = inf_paths, engine = "pyarrow").
            assign(country = lambda x: np.where(x.security.str.split(" ").str[0] == "BCMPGBIF", "UK", "US")))
        
        return df_inf
    
    def _get_fut(self, fut_path: str) -> None:
                
        fut_paths  = [os.path.join(fut_path, ticker + ".parquet") for ticker in self.energy_tickers]
        df_fut_rtn = (pd.read_parquet(
            path = fut_paths, engine = "pyarrow").
            assign(security = lambda x: x.security.str.split(" ").str[0]).
            pivot(index = "date", columns = "security", values = "PX_LAST").
            pct_change().
            reset_index().
            melt(id_vars = "date", value_name = "fut_rtn").
            dropna())
        
        return df_fut_rtn
    
    def _get_is_resid(self, df: pd.DataFrame) -> pd.DataFrame: 
    
        df_out = (sm.OLS(
            endog = df.fut_rtn,
            exog  = sm.add_constant(df.inf_surp)).
            fit().
            resid.
            to_frame(name = "resid").
            assign(lag_resid = lambda x: x.resid.shift()).
            merge(right = df, how = "inner", on = ["date"]))
    
        return df_out
    
    def _get_leg(self, df: pd.DataFrame) -> pd.DataFrame: 
    
        try:
        
            df_out = (df.assign(
                group = lambda x: pd.qcut(x = x.resid, q = 2, labels = ["lower_group", "upper_group"])))
    
            return df_out
    
        except:
            pass

    
    def get_is_resid(self, inf_path: str, fut_path: str, verbose: bool = True) -> pd.DataFrame: 
        
        out_path = os.path.join(self.data_path, "InSampleResid.parquet")
        if os.path.exists(out_path) == True: 
            if verbose: print("Already have in-sample Resid Saved")
            return None
        
        if verbose: print("Generating In-Sample Resids")
        
        df_inf_prep =  (self._get_inf(
            inf_path).pivot(
            index = "date", columns = "country", values = "value").
            diff().
            shift().
            reset_index().
            melt(id_vars = "date", value_name = "inf_surp").
            dropna())
                
        df_fut_rtn = self._get_fut(fut_path)
                
        df_resid = (df_inf_prep.merge(
            right = df_fut_rtn, how = "inner", on = ["date"]).
            assign(group_var = lambda x: x.country + " " + x.security).
            set_index("date").
            groupby("group_var").
            apply(self._get_is_resid, include_groups = False).
            reset_index().
            dropna().
            assign(resid_group = "is_resid"))
    
        if verbose: print("Saving Data\n")
        df_resid.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_is_leg(self, verbose: bool = True) -> pd.DataFrame: 
        
        out_path = os.path.join(self.data_path, "InSampleGroup.parquet")
        if os.path.exists(out_path) == True: 
            if verbose: print("Already have in-sample Group Saved")
            return None
        
        if verbose: print("Generating In-Sample Resids")
        
        in_path  = os.path.join(self.data_path, "InSampleResid.parquet")
        df_out   = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            drop(columns = ["group_var"]).
            dropna().
            assign(group_var = lambda x: x.date.astype(str) + " " + x.country).
            groupby("group_var").
            progress_apply(lambda group: self._get_leg(group)).
            reset_index())
        
        if verbose == True: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        

inf_path = r"A:\BBGData\data"
fut_path = r"A:\BBGFuturesManager_backup_backup\data\PXFront"

backtest  = CrossSectionBacktest()
#backtest.get_is_resid(inf_path, fut_path)
backtest.get_is_leg()