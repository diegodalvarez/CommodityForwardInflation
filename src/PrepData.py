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
        self.px_path   = os.path.join(self.data_path, "FutData")
        self.inf_path  = os.path.join(self.data_path, "InflationMeasures")
        
        self.forward_tickers  = ["FWISBP55", "FWISUS55"]
        self.surprise_tickers = ["BCMPGBIF", "BCMPUSIF"]
        #self.energy_tickers   = ["CL", "CO", "HO", "NG", "QS", "XB"]
        
        if not os.path.exists(self.data_path): 
            os.makedirs(self.data_path)
            
        if not os.path.exists(self.px_path):
            os.makedirs(self.px_path)
            
        if not os.path.exists(self.inf_path):
            os.makedirs(self.inf_path)
        
        #self.fut_path = r"A:\BlpData\BBGFutPX\1"
        self.fut_path = r"A:\2025Backup\BBGFuturesManager_backup\data\PXFront"
        self.bbg_path = r"A:\BBGData\data"
        
        self.vol_target = 0.1
        self.vol_window = 100
        
    def _get_fut_data(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.px_path, "FutPX.parquet")
    
        if verbose:
            print("Getting Futures PX data")
            
        if os.path.exists(out_path):
            if verbose:
                print("Already Have Futures PX Data\n")
            return None
        
        tick_path = os.path.join(self.data_path, "InflationTickerGuide.xlsx")
        tickers   = (pd
                .read_excel(io = tick_path, sheet_name = "FutGuide")
                .assign(ticker = lambda x: x.Ticker.str.split("1").str[0])
                .ticker
                .to_list())
        
        fut_paths = [
            os.path.join(self.fut_path, ticker + ".parquet")
            for ticker in tickers]
        
        df_px = (pd.read_parquet(
            path = fut_paths, engine = "pyarrow").
            assign(security = lambda x: x.security.str.split(" ").str[0]))
        
        if verbose: 
            print("Saving futures data")
            
        df_px.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_forward_inflation(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.inf_path, "InflationForward.parquet")
        if verbose:
            print("Getting 5y5y Forward Inflation data")
            
        if os.path.exists(out_path):
            if verbose:
                print("Already Have inflation data\n")
            return None
        
        tick_path = os.path.join(self.data_path, "InflationTickerGuide.xlsx")
        tickers   = (pd
                .read_excel(io = tick_path, sheet_name = "InflationMeasures")
                .loc[lambda x: x.Group == "Forward"]
                .assign(ticker = lambda x: x.Name.str.split(" ").str[0])
                .ticker
                .to_list())
        
        paths = [os.path.join(self.bbg_path, ticker + ".parquet") for ticker in tickers]
        
        df_out = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            assign(country = lambda x: np.where(x.security.str.split(" ").str[0] == "FWISBP55", "UK", "US")).
            drop(columns = ["variable"]))
        
        if verbose: 
            print("Saving forward inflation data")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_inflation_surprise(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.inf_path, "InflationSurprise.parquet")
        if verbose:
            print("Getting Inflation Surprise Data")
        
        if os.path.exists(out_path):
            if verbose:
                print("Already Have Inflation Surprise Data\n")
            return None
        
        tick_path = os.path.join(self.data_path, "InflationTickerGuide.xlsx")
        tickers   = (pd
                .read_excel(io = tick_path, sheet_name = "InflationMeasures")
                .loc[lambda x: x.Group == "InflationSurprise"]
                .assign(ticker = lambda x: x.Name.str.split(" ").str[0])
                .ticker
                .to_list())
        
        paths = [os.path.join(self.bbg_path, ticker + ".parquet") for ticker in tickers]
        
        df_out = (pd.read_parquet(
            path = paths, engine = "pyarrow").
            assign(country = lambda x: np.where(x.security.str.split(" ").str[0] == "BCMPGBIF", "UK", "US")).
            drop(columns = ["variable"]))
        
        if verbose: 
            print("Saving Inflation Surprise Data")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _combine_fred_data(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting combined FRED data")
        
        fred_path = os.path.join(self.data_path, "FRED")
        out_path  = os.path.join(fred_path, "CombinedData.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n\n")
            return None
        
        files     = [file for file in os.listdir(fred_path) if file.split(".")[-1] == "csv"]
        tick_path = os.path.join(self.data_path, "InflationTickerGuide.xlsx")
        df_lists  = []
        
        ticker_dict = (pd
                .read_excel(io = tick_path, sheet_name = "FRED")
                .set_index("ticker")
                .group
                .to_dict())
        
        for file in files: 
            
            tmp_path = os.path.join(fred_path, file)
            df_add   = (pd
                    .read_csv(filepath_or_buffer = tmp_path)
                    .rename(columns = {"observation_date": "date"})
                    .assign(date = lambda x: pd.to_datetime(x.date).dt.date)
                    .melt(id_vars = "date", var_name = "ticker"))
            
            df_lists.append(df_add)
        
        df_out = (pd
                .concat(df_lists)
                .assign(group = lambda x: x.ticker.map(ticker_dict)))
        
        if verbose: print("Saving data\n")    
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_vol_target(self, df: pd.DataFrame, vol_target: float = 0.1, vol_window: int = 100) -> pd.DataFrame: 
        
        df_out = (df
                .sort_values("date")
                .assign(
                    weight     = lambda x: vol_target / (x.rtn.ewm(span = vol_window, adjust = False).std() * np.sqrt(252)),
                    lag_weight = lambda x: x.weight.shift()))
        
        return df_out
        
    def vol_target_rtn(self, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Getting volatility targeted returns")
        
        out_path = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        in_path = os.path.join(self.data_path, "FutData", "FutPX.parquet")
        df_out  = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .set_index("date")
                .groupby("security")
                .apply(lambda x: x.sort_index().PX_LAST.pct_change())
                .reset_index()
                .rename(columns = {"PX_LAST": "rtn"})
                .set_index("date")
                .groupby("security")
                .apply(self._get_vol_target, self.vol_target, self.vol_window)
                .reset_index())
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
        
    data_prep = PrepData()
    data_prep._get_fut_data()
    data_prep._get_forward_inflation()
    data_prep._get_inflation_surprise()
    data_prep._combine_fred_data()
    data_prep.vol_target_rtn()
    
if __name__ == "__main__": main()