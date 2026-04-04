# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:11:25 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class SignalStrategies:
    
    def __init__(self) -> None: 
        
        self.path      = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        
    def _get_ols_beta(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        fs_beta = (sm.OLS(
            endog = df.fut_rtn,
            exog  = sm.add_constant(df.inf_diff)).
            fit().
            params.
            to_dict()
            ["inf_diff"])
        
        df_out = (RollingOLS(
            endog     = df.fut_rtn,
            exog      = sm.add_constant(df.inf_diff),
            expanding = True,
            min_nobs  = 30).
            fit().
            params.
            drop(columns = ["const"]).
            rename(columns = {"inf_diff": "os_beta"}).
            assign(
                lag_inf     = lambda x: x.inf_diff.shift(),
                lag_os_beta = lambda x: x.os_beta.shift(),
                fs_beta     = fs_beta).
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out