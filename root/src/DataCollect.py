# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 19:24:54 2025

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf

class DataManager:
    
    def __init__(self) -> None:
        
        self.dir       = os.path.dirname(os.path.abspath(__file__))  
        self.root_path = os.path.abspath(
            os.path.join(os.path.abspath(
                os.path.join(self.dir, os.pardir)), os.pardir))
        
        self.data_path      = os.path.join(self.root_path, "data")
        self.raw_data_path  = os.path.join(self.data_path, "RawData")
        
        if os.path.exists(self.data_path) == False: os.makedirs(self.data_path)
        if os.path.exists(self.raw_data_path) == False: os.makedirs(self.raw_data_path)
        
        self.energy_tickers = ["CL", "CO", "HO", "NG", "QS", "XB"]
        self.fut_path       = (
            r"C:\Users\Diego\Desktop\app_prod\BBGFuturesManager\data"
            r"\AdjustedVolTargetedPXFront\ConstantVolTargeting")
        
        self.yf_tickers = ["^SPGSCI", "^BCOM"]
        self.benchmarks = ["SGCOCOC2", "SGIXTFCY", "SGMDDBMF"]
        
        self.bbg_path    = r"C:\Users\Diego\Desktop\app_prod\BBGData\data"
        self.inf_tickers = ["FWISBP55", "FWISUS55"] 
        
    def get_energy_fut(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "EnergyCommodPX.parquet")
        try:
            
            if verbose == True: print("Seaching for energy futures")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting") 
        
            paths = [os.path.join(
                self.fut_path, ticker + ".parquet")
                for ticker in self.energy_tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                drop(columns = ["px", "kind"]).
                assign(
                    date     = lambda x: pd.to_datetime(x.date).dt.date,
                    security = lambda x: x.security.str.split(" ").str[0]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def _get_diff(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.sort_values(
            "date").
            assign(log_diff = lambda x: x.log_val.diff()))
        
        return df_out
    
    def get_inflation_swap(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "InflationForwards.parquet")
        try:
            
            if verbose == True: print("Seaching for inflation swaps")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
        
            paths = [
                os.path.join(self.bbg_path, ticker + ".parquet")
                for ticker in self.inf_tickers]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(
                    date     = lambda x: pd.to_datetime(x.date).dt.date,
                    security = lambda x: x.security.str.split(" ").str[0]).
                drop(columns = ["variable"]).
                assign(log_val = lambda x: np.log(x.value)).
                groupby("security").
                apply(self._get_diff).
                reset_index(drop = True))
                
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
    def get_yf(self, verbose: bool = False) -> pd.DataFrame:
        
        file_path = os.path.join(self.raw_data_path, "YFCommodIndices.parquet")
        try:
            
            if verbose == True: print("Seaching for YF Commodities")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
        
            start_date = dt.date(year = 1990, month = 1, day = 1)
            df_out     = (pd.concat([
                yf.Ticker(ticker).history(auto_adjust = False, start = start_date).assign(ticker = ticker)
                for ticker in self.yf_tickers]).
                reset_index().
                rename(columns = {
                    "Date"     : "date",
                    "Close"    : "close",
                    "Adj Close": "adj_close"}).
                assign(
                    ticker = lambda x: x.ticker.str.replace("^", ""), 
                    date   = lambda x: pd.to_datetime(x.date).dt.date)
                [["date", "close", "adj_close", "ticker"]])
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
        
        return df_out
    
    def get_commod_benchmark(self, verbose: bool = False) -> pd.DataFrame: 
        
        file_path = os.path.join(self.raw_data_path, "CommodityBenchmarks.parquet")
        try:
            
            if verbose == True: print("Seaching for Commodity Benchamrks")
            df_out = pd.read_parquet(path = file_path, engine = "pyarrow")
            if verbose == True: print("Found Data\n")
            
        except: 
            
            if verbose == True: print("Couldn't find it, collecting it now") 
            
            paths = [
                os.path.join(self.bbg_path, ticker + ".parquet")
                for ticker in self.benchmarks]
            
            df_out = (pd.read_parquet(
                path = paths, engine = "pyarrow").
                assign(
                    date     = lambda x: pd.to_datetime(x.date).dt.date,
                    security = lambda x: x.security.str.split(" ").str[0]).
                drop(columns = ["variable"]))
            
            if verbose == True: print("Saving data\n")
            df_out.to_parquet(path = file_path, engine = "pyarrow")
            
        return df_out
    
def main() -> None:
            
    DataManager().get_inflation_swap(verbose = True)
    DataManager().get_energy_fut(verbose = True)
    DataManager().get_yf(verbose = True)
    DataManager().get_commod_benchmark(verbose = True)
    
if __name__ == "__main__": main()
        
        