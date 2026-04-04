# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 14:31:39 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class PrepData:
    
    def __init__(self) -> None: 
        
        self.path      = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        
        self.forward_tickers  = ["FWISBP55", "FWISUS55"]
        self.surprise_tickers = ["BCMPGBIF", "BCMPUSIF"]
        self.energy_tickers   = ["CL", "CO", "HO", "NG", "QS", "XB"]
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        
        #self.fut_path = r"A:\BlpData\BBGFutPX\1"
        self.fut_path = r"A:\2025Backup\BBGFuturesManager_backup\data\PXFront"
        self.bbg_path = r"A:\BBGData\data"
        
    def _get_fut_data(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "FutPX.parquet")
        if verbose:
            print("Getting Futures PX data")
            
        if os.path.exists(out_path):
            if verbose:
                print("Already Have Futures PX Data")
            return None
        
        fut_paths = [
            os.path.join(self.fut_path, ticker + ".parquet")
            for ticker in self.energy_tickers]
        
        df_px = (pd.read_parquet(
            path = fut_paths, engine = "pyarrow").
            assign(security = lambda x: x.security.str.split(" ").str[0]))
        
        if verbose: 
            print("Saving futures data")
            
        df_px.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_forward_inflation(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "InflationForward.parquet")
        if verbose:
            print("Getting 5y5y Forward Inflation data")
            
        if os.path.exists(out_path):
            if verbose:
                print("Already Have inflation data")
            return None
        
        paths    = [os.path.join(
            self.bbg_path, ticker + ".parquet")
            for ticker in self.forward_tickers]
        
        df_out = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            assign(country = lambda x: np.where(x.security.str.split(" ").str[0] == "FWISBP55", "UK", "US")).
            drop(columns = ["variable"]))
        
        if verbose: 
            print("Saving forward inflation data")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_inflation_surprise(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "InflationSurprise.parquet")
        if verbose:
            print("Getting Inflation Surprise Data")
        
        if os.path.exists(out_path):
            if verbose:
                print("Already Have Inflation Surprise Data")
            return None
        
        paths    = [os.path.join(
            self.bbg_path, ticker + ".parquet")
            for ticker in self.surprise_tickers]
        
        df_out = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            assign(country = lambda x: np.where(x.security.str.split(" ").str[0] == "BCMPGBIF", "UK", "US")).
            drop(columns = ["variable"]))
        
        if verbose: 
            print("Saving Inflation Surprise Data")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
        
    data_prep = PrepData()
    data_prep._get_fut_data()
    data_prep._get_forward_inflation()
    data_prep._get_inflation_surprise()
    
if __name__ == "__main__": main()