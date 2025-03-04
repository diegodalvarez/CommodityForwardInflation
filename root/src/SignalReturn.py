# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 20:32:41 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   DataCollect import DataManager

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class SignalReturn(DataManager):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.signal_path = os.path.join(self.data_path, "SignalRtn")
        if os.path.exists(self.signal_path) == False: os.makedirs(self.signal_path)
        
    def get_raw_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "RawRtn.parquet")
        try:
            
            if verbose == True: print("Seaching for raw return")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
        
            df_inf = (self.get_inflation_swap().pivot(
                index = "date", columns = "security", values = "log_diff").
                shift().
                reset_index().
                melt(id_vars = "date", var_name = "inf_ticker"))
            
            df_out = (self.get_energy_fut()[
                ["date", "security", "vol_rtn"]].
                dropna().
                merge(right = df_inf, how = "inner", on = ["date"]).
                dropna().
                assign(signal_rtn = lambda x: np.sign(x.value) * x.vol_rtn))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_is_beta(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        model = (sm.OLS(
            endog = df_tmp.vol_rtn / df_tmp.vol_rtn.ewm(span = 10, adjust = False).std().fillna(1),
            exog  = sm.add_constant(df_tmp.value)).
            fit())
        
        beta, pvalue = model.params["value"], model.pvalues["value"]
        df_out       = (df.assign(
            beta   = beta,
            pvalue = pvalue))
        
        return df_out
        
    def get_is_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "InSampleRtn.parquet")
        try:
            
            if verbose == True: print("Seaching for in sample return")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
            
            df_out = (self.get_raw_rtn().drop(
                columns = "signal_rtn").
                groupby(["security", "inf_ticker"]).
                apply(self._get_is_beta).
                reset_index(drop = True).
                assign(signal_rtn = lambda x: np.sign(x.beta * x.value) * x.vol_rtn))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_oos_beta(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.sort_values(
            "date").
            set_index("date"))
        
        df_out = (RollingOLS(
            endog     = df_tmp.vol_rtn / df_tmp.vol_rtn.ewm(span = 10, adjust = False).std().fillna(1),
            exog      = sm.add_constant(df_tmp.value),
            expanding = True).
            fit().
            params
            [["value"]].
            shift().
            rename(columns = {"value": "lag_beta"}).
            merge(right = df_tmp, how = "inner", on = ["date"]))
        
        return df_out
    
    def get_oos_rtn(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "OutSampleRtn.parquet")
        try:
            
            if verbose == True: print("Seaching for in sample return")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
            
            df_out = (self.get_raw_rtn().drop(
                columns = ["signal_rtn"]).
                groupby(["security", "inf_ticker"]).
                apply(self._get_oos_beta).
                drop(columns = ["security", "inf_ticker"]).
                reset_index().
                assign(signal_rtn = lambda x: np.sign(x.lag_beta * x.value) * x.vol_rtn))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
def main() -> None:
         
    SignalReturn().get_is_rtn(verbose = True)
    SignalReturn().get_raw_rtn(verbose = True)
    SignalReturn().get_oos_rtn(verbose = True)

if __name__ == "__main__": main()