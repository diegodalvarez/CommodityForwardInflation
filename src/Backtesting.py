# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:46:46 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm

class Backtesting:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.back_path = os.path.join(self.data_path, "Backtests")
        
        if not os.path.exists(self.back_path):
            os.makedirs(self.back_path)
            
        self.vol_target = 0.1
        self.vol_window = 100
        self.threshold  = 0.1
            
    def _lag_signal(self, df: pd.DataFrame, signal_name: str = "value") -> pd.DataFrame: 
        
        df_out = (df
                  .sort_index()
                  .assign(lag_zscore = lambda x: x[signal_name].shift())
                  .reset_index())
        
        return df_out
    
    def _vol_target(
            self, 
            df        : pd.DataFrame, 
            lag       : int,
            name      : str,
            vol_target: float, 
            vol_window: int,
            threshold : float) -> pd.DataFrame:
        
        df_out = (df
                .pivot(index = "date", columns = name, values = "signal_rtn")
                .apply(
                    lambda x: x * 
                    (vol_target / (x.ewm(span = vol_window, adjust = False).std().shift(lag) * np.sqrt(252))))
                .apply(lambda x: np.where(np.abs(x) > threshold, np.nan, x))
                .reset_index()
                .melt(id_vars = "date", value_name = "vol_rtn")
                .dropna())
        
        return df_out
    
    def _vol_target_rtn(
            self,
            df        : pd.DataFrame,
            vol_target: float,
            vol_window: int) -> pd.DataFrame: 
        
        df_out = (df
                .sort_index()
                .assign(
                    signal_rtn = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.rtn,
                    weight     = lambda x: 
vol_target / (x.signal_rtn.ewm(span = vol_window, adjust = False).std() * np.sqrt(252)),
                    lag_weight = lambda x: x.weight.shift()))
            
        return df_out
            
    def get_signal_backtest(self, verbose: bool = True) ->  None: 
        
        if verbose: 
            print("Working on Signal Backtest")
        
        out_path = os.path.join(self.back_path, "SignalBacktest.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have the data\n")
            return None
        
        sig_path  = os.path.join(self.data_path, "Signals", "ZScore.parquet")
        df_zscore = (pd
                     .read_parquet(path = sig_path, engine = "pyarrow")
                     .set_index("date")
                     .groupby(["country", "group"])
                     .apply(self._lag_signal, "z_score")
                     .drop(columns = ["z_score"])
                     .reset_index()
                     .drop(columns = ["level_2"])
                     .dropna())

        rtn_path = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        df_rtn   = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                [["date", "security", "rtn"]]
                .dropna())
        
        df_combined = (df_zscore
                .merge(right = df_rtn, how = "inner", on = ["date"])
                .assign(
                    signal_rtn = lambda x: np.sign(x.lag_zscore) * x.rtn,
                    group_name = lambda x: x.group + " " + x.country + " " + x.security))
        
        df_perf = (self
                   ._vol_target(
                       df         = df_combined, 
                       lag        = 0,
                       name       = "group_name",
                       vol_target = self.vol_target, 
                       vol_window = self.vol_window, 
                       threshold  = self.threshold)
                   .assign(target = "perfect"))
        
        df_lag = (self
                   ._vol_target(
                       df         = df_combined, 
                       lag        = 1,
                       name       = "group_name",
                       vol_target = self.vol_target, 
                       vol_window = self.vol_window, 
                       threshold  = self.threshold)
                   .assign(target = "lagged"))
        
        df_out = (pd
                .concat([df_perf, df_lag])
                .assign(
                    str_split = lambda x: x.group_name.str.split(" "),
                    inf_name  = lambda x: x.str_split.str[0],
                    country   = lambda x: x.str_split.str[1],
                    security  = lambda x: x.str_split.str[2])
                .drop(columns = ["group_name", "str_split"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_residual_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Getting Residual Backtest")
        
        model_path = os.path.join(self.data_path, "Signals", "OLSResidModels.pkl")
        fut_path   = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        sig_path   = os.path.join(self.data_path, "Signals", "ZScore.parquet")
        out_path   = os.path.join(self.back_path, "OLSResidualBacktest.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_raw_rtn = (pd
                .read_parquet(path = fut_path)
                .set_index("date"))
        
        df_rtn = (df_raw_rtn
                .drop(columns = ["weight", "lag_weight"])
                .dropna())
        
        df_perf_rtn = (df_raw_rtn
                .assign(perf_rtn = lambda x: x.weight * x.rtn)
                [["security", "perf_rtn"]]
                .dropna())
        
        df_zscore = (pd
                     .read_parquet(path = sig_path, engine = "pyarrow")
                     .set_index("date"))
        
        with open(file = model_path, mode = "rb") as f: models = pickle.load(f)
        
        names    = list(models.keys())
        df_lists = [] 
        
        for name in names: 
            
            tmp_models                     = models[name]
            security, country, signal_name = name.split(" ")
            
            df_tmp_rtn = (df_rtn
                    .loc[lambda x: x.security == security]
                    .rename(columns = {"rtn": "fut_rtn"}))
            
            df_tmp_perf_rtn = (df_perf_rtn
                    .loc[lambda x: x.security == security])
            
            df_tmp_zscore = (df_zscore
                    .loc[lambda x: (x.country == country) & (x.group == signal_name)])
            
            # for the full-sample in-sample
            df_fs_is = (tmp_models
                    ["full_sample"]
                    .resid
                    .to_frame(name = "resid")
                    .assign(lag_resid = lambda x: x.resid.shift())
                    .merge(right = df_tmp_rtn, how = "inner", on = ["date"])
                    .assign(
                        regression   = "full_sample",
                        sample_group = "in_sample",
                        signal_rtn   = lambda x: -np.sign(x.lag_resid) * x.fut_rtn)
                    .reset_index())
            
            is_model   = tmp_models["in_sample"]
            slice_date = (is_model
                    .resid
                    .index
                    .max())
            
            # the train/test model
            df_train_test = (df_tmp_zscore
                    .merge(right = df_tmp_perf_rtn, how = "inner", on = ["date"])
                    .merge(right = df_tmp_rtn, how = "inner", on = ["date", "security"])
                    .reset_index()
                    .assign(sample_group = lambda x: np.where(x.date <= slice_date, "in_sample", "out_sample"))
                    .set_index("date")
                    .assign(
                        y_pred       = lambda x: is_model.predict(sm.add_constant(x.z_score)),
                        resid        = lambda x: x.perf_rtn - x.y_pred,
                        lag_resid    = lambda x: x.resid.shift(),
                        signal_rtn   = lambda x: -np.sign(x.lag_resid) * x.fut_rtn,
                        regression   = "train_test")
                    .drop(columns = ["y_pred", "perf_rtn", "z_score", "country", "group"])
                    .reset_index())
            
            # the expanding model
            df_expanding = (tmp_models
                    ["expanding_model"]
                    .params
                    .dropna()
                    .rename(columns = {"z_score": "beta"})
                    .merge(right = df_tmp_zscore,   how = "inner", on = ["date"])
                    .merge(right = df_tmp_perf_rtn, how = "inner", on = ["date"])
                    .merge(right = df_tmp_rtn,      how = "inner", on = ["date", "security"])
                    .assign(
                        regression   = "expanding",
                        sample_group = "out_sample",
                        y_pred     = lambda x: (x.beta * x.z_score) + x.const,
                        resid      = lambda x: x.perf_rtn - x.y_pred,
                        lag_resid  = lambda x: x.resid.shift(),
                        signal_rtn = lambda x: -np.sign(x.lag_resid) * x.fut_rtn)
                    .drop(columns = [
                        "const", "beta", "perf_rtn", "y_pred", "z_score", 
                        "country", "group"])
                    .reset_index())
            
            df_tmp_combined = (pd
                    .concat([df_fs_is, df_train_test, df_expanding])
                    .assign(
                        group       = name, 
                        name        = lambda x: x.group + " " + x.regression,
                        country     = country,
                        signal_name = signal_name))
            
            df_input = (df_tmp_combined
                    [["date", "name", "signal_rtn"]])
            
            df_perf = (self
                       ._vol_target(
                           df         = df_input, 
                           lag        = 0,
                           name       = "name",
                           vol_target = self.vol_target, 
                           vol_window = self.vol_window, 
                           threshold  = self.threshold)
                       .assign(target = "perfect"))
            
            df_lag = (self
                       ._vol_target(
                           df         = df_input, 
                           lag        = 1,
                           name       = "name",
                           vol_target = self.vol_target, 
                           vol_window = self.vol_window, 
                           threshold  = self.threshold)
                       .assign(target = "lagged"))
            
            df_add = (pd
                    .concat([df_perf, df_lag])
                    .merge(right = df_tmp_combined, how = "inner", on = ["date", "name"]))
            
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_pred_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Getting Pred Backtest")
        
        model_path = os.path.join(self.data_path, "Signals", "OLSPredModels.pkl")
        fut_path   = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        sig_path   = os.path.join(self.data_path, "Signals", "ZScore.parquet")
        out_path   = os.path.join(self.back_path, "OLSPredBacktest.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_raw_rtn = (pd
                .read_parquet(path = fut_path)
                .set_index("date"))
        
        df_rtn = (df_raw_rtn
                .drop(columns = ["weight", "lag_weight"])
                .dropna())
        
        df_perf_rtn = (df_raw_rtn
                .assign(perf_rtn = lambda x: x.weight * x.rtn)
                [["security", "perf_rtn"]]
                .dropna())
        
        df_zscore = (pd
                     .read_parquet(path = sig_path, engine = "pyarrow")
                     .set_index("date")
                     .groupby(["country", "group"])
                     .apply(lambda x: x.z_score.shift())
                     .to_frame(name = "lag_zscore")
                     .dropna()
                     .reset_index())
        
        with open(file = model_path, mode = "rb") as f: models = pickle.load(f)
        
        names    = list(models.keys())
        df_lists = [] 
        
        for name in names: 
            
            tmp_models                     = models[name]
            security, country, signal_name = name.split(" ")
            
            df_tmp_rtn = (df_rtn
                    .loc[lambda x: x.security == security]
                    .rename(columns = {"rtn": "fut_rtn"}))
            
            df_tmp_perf_rtn = (df_perf_rtn
                    .loc[lambda x: x.security == security])
            
            df_tmp_zscore = (df_zscore
                    .loc[lambda x: (x.country == country) & (x.group == signal_name)])
            
            # for the full-sample in-sample
            df_fs_is = (tmp_models
                    ["full_sample"]
                    .fittedvalues
                    .to_frame(name = "y_pred")
                    .merge(right = df_tmp_rtn, how = "inner", on = ["date"])
                    .assign(
                        group        = name,
                        country      = country,
                        regression   = "full_sample",
                        sample_group = "in_sample",
                        signal_rtn   = lambda x: np.sign(x.y_pred) * x.fut_rtn)
                    .reset_index())
            
            is_model   = tmp_models["in_sample"]
            slice_date = (is_model
                    .resid
                    .index
                    .max())
            
            # the train/test model
            df_train_test = (df_tmp_zscore
                    .merge(right = df_tmp_rtn, how = "inner", on = ["date"])
                    .assign(
                        group        = name,
                        sample_group = lambda x: np.where(x.date <= slice_date, "in_sample", "out_sample"),
                        y_pred       = lambda x: is_model.predict(sm.add_constant(x.lag_zscore)),
                        signal_rtn   = lambda x: np.sign(x.y_pred) * x.fut_rtn,
                        regression   = "train_test")
                    .drop(columns = ["lag_zscore"]))
            
            # the expanding model
            df_expanding = (tmp_models
                    ["expanding_model"]
                    .params
                    .dropna()
                    .rename(columns = {"z_score": "beta"})
                    .merge(right = df_tmp_zscore, how = "inner", on = ["date"])
                    .merge(right = df_tmp_rtn,    how = "inner", on = ["date"])
                    .assign(
                        y_pred       = lambda x: (x.beta * x.lag_zscore) + x.const,
                        signal_rtn   = lambda x: np.sign(x.y_pred) * x.fut_rtn,
                        regression   = "expanding_regression",
                        sample_group = "out_sample")
                    .drop(columns = ["beta", "const", "lag_zscore"]))
    
            df_tmp_combined = (pd
                    .concat([df_fs_is, df_train_test, df_expanding])
                    .assign(
                        group       = name, 
                        name        = lambda x: x.group + " " + x.regression,
                        country     = country,
                        signal_name = signal_name))
            
            df_input = (df_tmp_combined
                    [["date", "name", "signal_rtn"]])
            
            df_perf = (self
                       ._vol_target(
                           df         = df_input, 
                           lag        = 0,
                           name       = "name",
                           vol_target = self.vol_target, 
                           vol_window = self.vol_window, 
                           threshold  = self.threshold)
                       .assign(target = "perfect"))
            
            df_lag = (self
                       ._vol_target(
                           df         = df_input, 
                           lag        = 1,
                           name       = "name",
                           vol_target = self.vol_target, 
                           vol_window = self.vol_window, 
                           threshold  = self.threshold)
                       .assign(target = "lagged"))
            
            df_add = (pd
                    .concat([df_perf, df_lag])
                    .merge(right = df_tmp_combined, how = "inner", on = ["date", "name"]))
            
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_opt_zscore_backtest(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Optimized Z-Score Backtests")
        
        in_path  = os.path.join(self.data_path, "OptimizedSignals", "OptimizedZScore.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "OptimizedZScore.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_out = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .assign(group_var = lambda x: x.signal_name + " " + x.optimization)
                .set_index("date")
                .groupby("group_var")
                .apply(self._vol_target_rtn, self.vol_target, self.vol_window)
                .reset_index())
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_opt_residual_backtest(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting Optimized Residual Backtest")
        
        in_path  = os.path.join(self.data_path, "OptimizedSignals", "OptimizedResidual.parquet")
        out_path = os.path.join(self.data_path, "Backtests", "OptimizedResid.parquet") 
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        df_out = (pd
                .read_parquet(path = in_path, engine = "pyarrow")
                .assign(group_var = lambda x: x.group + " " + x.security + " " + x.optimization + " " + x.country)
                #.loc[lambda x: x.group_var == x.group_var.min()]
                .groupby("group_var")
                .apply(self._vol_target_rtn, self.vol_target, self.vol_window)
                .reset_index()
                .drop(columns = ["level_1"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
    
    backtesting = Backtesting()
    backtesting.get_signal_backtest()
    backtesting.get_residual_backtest()
    backtesting.get_pred_backtest()
    backtesting.get_opt_zscore_backtest()
    backtesting.get_opt_residual_backtest()
    
if __name__ == "__main__": main()