# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 23:08:57 2025

@author: Diego
"""

import os
import pandas as pd
from   SignalReturn import SignalReturn

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class PortfolioAttributes(SignalReturn):
    
    def __init__(self) -> None:
        
        super().__init__()
        self.port_attributes = os.path.join(self.data_path, "PortAttributes")
        if os.path.exists(self.port_attributes) == False: os.makedirs(self.port_attributes)
        
    def _prep_benchmark(self) -> pd.DataFrame: 
        
        df_yf = (self.get_yf().drop(
            columns = ["close"]).
            rename(columns = {
                "adj_close": "value",
                "ticker"   : "security"}))
        
        df_out = (pd.concat([
            df_yf, self.get_commod_benchmark()]).
            assign(date = lambda x: pd.to_datetime(x.date)).
            pivot(index = "date", columns = "security", values = "value").
            pct_change().
            reset_index().
            melt(
                id_vars    = "date", 
                var_name   = "benchmark", 
                value_name = "benchmark_rtn").
            dropna())
        
        return df_out
    
    def _prep_rtn(self) -> pd.DataFrame: 
        
        df_out = (self.get_oos_rtn()[
            ["inf_ticker", "date", "signal_rtn"]].
            groupby(["date", "inf_ticker"]).
            agg("mean").
            reset_index().
            assign(date = lambda x: pd.to_datetime(x.date)).
            pivot(index = "date", columns = "inf_ticker", values = "signal_rtn").
            reset_index().
            melt(
                id_vars    = "date",
                var_name   = "port",
                value_name = "port_rtn").
            dropna())
        
        return df_out
    
    def _measure_alpha(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index().
            dropna())
        
        model = (sm.OLS(
            endog = df_tmp.port_rtn,
            exog  = sm.add_constant(df_tmp.benchmark_rtn)).
            fit())
        
        df_param = (model.params.to_frame(
            name = "param_val").
            reset_index())
        
        df_out = (model.pvalues.to_frame(
            name = "pval").
            reset_index().
            merge(df_param, how = "inner", on = ["index"]))
        
        return df_out
    
    def OLSPerformance(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.signal_path, "OLSParams.parquet")
        try:
            
            if verbose == True: print("Seaching for OLS Params")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
        
            df_out = (self._prep_benchmark().merge(
                right = self._prep_rtn(), how = "inner", on = ["date"]).
                dropna().
                groupby(["benchmark", "port"]).
                apply(self._measure_alpha).
                reset_index().
                drop(columns = ["level_2"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_rolling_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.set_index(
            "date").
            sort_index())
        
        df_out = (RollingOLS(
            endog  = df_tmp.port_rtn,
            exog   = sm.add_constant(df_tmp.benchmark_rtn),
            window = 30).
            fit().
            params)
        
        return df_out
    
    def RollingOLSPerformance(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.signal_path, "RollingOLSParams.parquet")
        try:
            
            if verbose == True: print("Seaching for Rolling OLS Params")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
            
            df_out = (self._prep_benchmark().merge(
                right = self._prep_rtn(), how = "inner", on = ["date"]).
                groupby(["benchmark", "port"]).
                apply(self._get_rolling_ols).
                reset_index())
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out

def main() -> None:
    
    PortfolioAttributes().RollingOLSPerformance(verbose = True)
    PortfolioAttributes().OLSPerformance(verbose = True)
    
if __name__ == "__main__": main()