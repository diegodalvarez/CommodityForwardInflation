# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 21:57:07 2026

@author: Diego
"""

import os
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm

from tqdm import tqdm
tqdm.pandas()

class Optimizer:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.opt_path  = os.path.join(self.data_path, "OptimizedSignals")
        
        if not os.path.exists(self.opt_path):
            os.makedirs(self.opt_path)
            
        self.q          = 10
        self.slice_year = 2018
    
    def _get_decile(self, df: pd.DataFrame, signal_name: str, q: int = 10) -> pd.DataFrame: 
        
        df_out = (df
                .assign(
                    decile     = lambda x: pd.qcut(x = x[signal_name], q = q, labels = [i + 1 for i in range(q)]),
                    lag_decile = lambda x: x.decile.shift())
                .dropna())
        
        return df_out
    
    def _opt_full_sample_decile(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = df.reset_index()
        
        df_decile_sharpe = (df
                .reset_index(drop = True)
                [["vol_rtn", "lag_decile"]]
                .groupby("lag_decile")
                .agg(lambda x: x.mean() / x.std() * np.sqrt(252))
                .reset_index()
                .rename(columns = {"vol_rtn": "sharpe"}))
        
        df_tmp_decile = (df_decile_sharpe
                .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                .assign(decile_group = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_tmp_decile
                .drop(columns = ["lag_decile"])
                .groupby("decile_group")
                .agg("prod")
                .assign(signal_scaler = lambda x: np.where(x.sharpe > 0, 1, np.nan))
                .drop(columns = ["sharpe"])
                .merge(right = df_tmp_decile, how = "inner", on = ["decile_group"])
                .merge(right = df_tmp, how = "outer", on = ["lag_decile"]))
        
        return df_out
    
    def _get_train_test_decile(self, df: pd.DataFrame, q: int, slice_date: dt.date) -> pd.DataFrame: 
        
        df_train_test = (df
                .reset_index()
                .assign(sample_group = lambda x: np.where(x.date <= slice_date, "in_sample", "out_sample")))
          
        df_insample = (df_train_test
                .loc[lambda x: x.sample_group == "in_sample"])
        
        _, bins = (pd
                   .qcut(
                       x       = df_insample.z_score,
                       q       = q,
                       labels  = [i + 1 for i in range(q)],
                       retbins = True))
        
        bins[0], bins[-1] = -np.inf, np.inf
        
        df_out = (df_train_test
                .assign(decile = lambda x: pd
                        .cut(
                            x      = x.z_score, 
                            bins   = bins,
                            labels = range(1,q+1)),
                        lag_decile = lambda x: x.decile.shift()))
        
        return df_out
    
    def _opt_train_test_decile(self, df: pd.DataFrame, rtn_name: str = "vol_rtn") -> pd.DataFrame: 
    
        df_sharpe = (df
                .loc[lambda x: x.sample_group == "in_sample"]
                .reset_index(drop = True)
                [["lag_decile", rtn_name]]
                .groupby("lag_decile")
                .agg(lambda x: x.mean() / x.std() * np.sqrt(252))
                .rename(columns = {rtn_name: "sharpe"})
                .reset_index())
        
        df_tmp_decile = (df_sharpe
                .loc[lambda x: x.lag_decile.isin([1,2,9,10])]
                .assign(decile_group = lambda x: np.where(x.lag_decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_tmp_decile
                .drop(columns = ["lag_decile"])
                .groupby("decile_group")
                .agg("prod")
                .assign(signal_scaler = lambda x: np.where(x.sharpe > 0, 1, np.nan))
                .drop(columns = ["sharpe"])
                .merge(right = df_tmp_decile,    how = "outer", on = ["decile_group"])
                .merge(right = df.reset_index(), how = "outer", on = ["lag_decile"]))
        
        return df_out
    
    def _expanding_optimize(
            self, 
            df         : pd.DataFrame,
            q          : int,
            signal_name: str, 
            rtn_name   : str,
            min_obs    : int = 5,
            verbose    : bool = False) -> pd.DataFrame: 
        
        try:
            if verbose: print("Working on ", df.name)
        except: 
            pass
        
        df = df.sort_index()
    
        opt_dates = (
            df.index
            .to_series()
            .groupby(pd.Grouper(freq="W-FRI"))
            .max()
            .dropna()
            .to_list())[min_obs:]
        
        df_out = []
    
        if verbose: iterable = tqdm(opt_dates)
        else      : iterable = opt_dates
    
        for opt_date in iterable:
    
            # -------------------------
            # In-sample
            # -------------------------
    
            df_is = df.loc[:opt_date, [signal_name, rtn_name]].copy()
            
            df_is["decile"], bins = pd.qcut(
                df_is[signal_name],
                q=q,
                labels=False,
                retbins=True
            )
            
            df_is["decile"] += 1
    
            # Lag the decile
            df_is["lag_decile"] = df_is["decile"].shift()
            
            df_sharpe = (
                df_is
                .dropna(subset=["lag_decile"])
                .groupby("lag_decile")[rtn_name]
                .agg(["mean", "std"])
                .assign(
                    sharpe=lambda x:
                        x["mean"] / x["std"] * np.sqrt(252)
                )
                .reset_index()
                [["lag_decile", "sharpe"]]
            )
                
            df_tmp_decile = (
                df_sharpe
                .loc[lambda x: x["lag_decile"].isin([1, 2, 9, 10])]
                .assign(
                    group=lambda x: np.where(
                        x["lag_decile"] <= 2,
                        "lgroup",
                        "ugroup"
                    )
                )
            )
            
            df_signal_scaler = (
                df_tmp_decile
                .groupby("group")["sharpe"]
                .prod()
                .gt(0)
                .astype(int)
                .rename("signal_scaler")
            )
            
            df_oos = df.loc[
                (df.index > opt_date) &
                (df.index <= opt_date + pd.Timedelta(days=7))
            ].copy()
    
            # Apply IS bins to OOS observations
            df_oos["decile"] = (
                pd.cut(
                    df_oos[signal_name],
                    bins=bins,
                    labels=False,
                    include_lowest=True
                ) + 1
            )
            
            df_oos["lag_decile"] = df_oos["decile"].shift()
            
            df_oos = df_oos.reset_index().merge(
                df_tmp_decile[
                    ["lag_decile", "sharpe", "group"]
                ],
                how="left",
                on="lag_decile"
            )
            
            # Map lgroup / ugroup to OOS deciles
            df_oos["group"] = np.select(
                [
                    df_oos["lag_decile"].isin([1, 2]),
                    df_oos["lag_decile"].isin([9, 10])
                ],
                [
                    "lgroup",
                    "ugroup"
                ],
                default=None
            )
            
            df_oos["signal_scaler"] = (
                df_oos["group"]
                .map(df_signal_scaler)
            )
            
            df_oos["opt_date"] = opt_date
    
            df_out.append(df_oos)
         
        if verbose: print("\n")   
         
        return pd.concat(df_out)
        
    def get_optimized_zscore(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Optimized Z-Score\n")
        out_path = os.path.join(self.opt_path, "OptimizedZScore.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        rtn_path   = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        df_raw_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .assign(vol_rtn = lambda x: x.weight * x.rtn))
        
        df_perf_rtn = (df_raw_rtn
            [["date", "security", "vol_rtn"]]
            .dropna())
        
        df_rtn = (df_raw_rtn
                [["date", "security", "rtn"]]
                .dropna())
        
        sig_path     = os.path.join(self.data_path, "Signals", "ZScore.parquet")
        df_signal = (pd
                .read_parquet(path = sig_path, engine = "pyarrow")
                .set_index("date"))
    
        # for the full sample case
        df_full_sample_decile = (df_signal
                .groupby(["country", "group"])
                .apply(self._get_decile, "z_score", self.q)
                .reset_index())
        
        if verbose: print("Getting Full-Sample In-Sample Optimized Returns")
        
        df_is_opt_rtn = (df_full_sample_decile
                .drop(columns = ["z_score"])
                .merge(right = df_perf_rtn, how = "inner", on = ["date"])
                .set_index("date")
                .assign(signal_name = lambda x: x.country + " " + x.group + " " + x.security)
                .groupby("signal_name")
                .progress_apply(lambda group: self._opt_full_sample_decile(group))
                .reset_index()
                .drop(columns = ["level_1", "vol_rtn"])
                .assign(
                    sample_group = "in_sample",
                    opt_date     = df_full_sample_decile.date.max(),
                    optimization = "full_sample"))
        
        slice_date = (df_signal
                .reset_index()
                .assign(year = lambda x: x.date.dt.year)
                .loc[lambda x: x.year <= self.slice_year]
                .date
                .max())
        
        df_train_test_decile = (df_signal
                .groupby(["country", "group"])
                .apply(self._get_train_test_decile, self.q, slice_date)
                .reset_index()
                .drop(columns = ["level_2"]))
        
        if verbose: print("\nGetting Train/Test Optimized Returns")
        
        df_train_test_opt_rtn = (df_train_test_decile
                .merge(right = df_perf_rtn, how = "inner", on = ["date"])
                .dropna()
                .set_index("date")
                .assign(signal_name = lambda x: x.country + " " + x.group + " " + x.security)
                .groupby("signal_name")
                .progress_apply(lambda group: self._opt_train_test_decile(group))
                .reset_index()
                .drop(columns = ["level_1", "vol_rtn", "z_score"])
                .assign(
                    opt_date     = slice_date,
                    optimization = "train_test"))
        
        if verbose: print("\nRunning Expanding Out-of-Sample Optimized Returns")
        
        df_exp_opt_rtn = (df_signal
                .merge(right = df_perf_rtn, how = "inner", on = ["date"])
                .rename(columns = {"group": "signal_name"})
                #.loc[lambda x: x.security == x.security.min()]
                #.loc[lambda x: x.country == x.country.min()]
                #.loc[lambda x: x.signal_name == x.signal_name.min()]
                .set_index("date")
                .groupby(["country", "signal_name", "security"])
                .apply(self._expanding_optimize, self.q, "z_score", "vol_rtn", 5, verbose = True)
                .reset_index()
                .drop(columns = ["level_3", "z_score"])
                .rename(columns = {
                    "group"      : "decile_group",
                    "signal_name": "group"})
                .assign(
                    sample_group = "out_sample",
                    signal_name  = lambda x: x.country + " " + x.group + " " + x.security,
                    optimization = "expanding"))
        
        df_out = (pd
                .concat([df_is_opt_rtn, df_train_test_opt_rtn, df_exp_opt_rtn])
                .merge(right = df_rtn, how = "inner", on = ["date", "security"])
                .merge(right = df_signal, how = "inner", on = ["date", "country", "group"])
                .drop(columns = ["vol_rtn"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def get_optimized_resid(self, verbose: bool = True) -> None: 
        
        if verbose: print("Getting Optimized Residual Model")
        
        signal_path = os.path.join(self.data_path, "Signals")
        resid_path  = os.path.join(signal_path, "OLSResidModels.pkl")
        zscore_path = os.path.join(signal_path, "ZScore.parquet")
        rtn_path    = os.path.join(self.data_path, "FutData", "VolHedgedRtn.parquet")
        out_path    = os.path.join(self.opt_path, "OptimizedResidual.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Saving data\n")
            return None
        
        df_raw_rtn = (pd
                .read_parquet(path = rtn_path, engine = "pyarrow")
                .assign(vol_rtn = lambda x: x.weight * x.rtn)
                .drop(columns = ["weight", "lag_weight"])
                .set_index("date")
                .dropna())
        
        df_vol_rtn = df_raw_rtn[["security", "vol_rtn"]]
        
        df_zscore = (pd
                .read_parquet(path = zscore_path, engine = "pyarrow")
                .set_index("date"))
        
        with open(resid_path, "rb") as f: models = pickle.load(f)
        
        names    = list(models.keys())
        df_lists = []
        
        for name in names: 
            
            tmp_models                     = models[name]
            security, country, signal_name = name.split(" ")
            
            df_tmp_vol_rtn = (df_vol_rtn
                    .loc[lambda x: x.security == security])
            
            df_tmp_zscore = (df_zscore
                    .loc[lambda x: x.country == country]
                    .loc[lambda x: x.group == signal_name])
            
            df_tmp_rtn = (df_raw_rtn
                    .loc[lambda x: x.security == security]
                    [["security", "rtn"]]
                    .reset_index())
            
            # for the full-sample in-sample
            df_fs_resid = (tmp_models
                    ["full_sample"]
                    .resid
                    .to_frame(name = "resid"))
            
            df_fs_opt_rtn = (df_fs_resid
                            .pipe(self._get_decile, "resid")
                            .merge(
                                right = df_tmp_vol_rtn,
                                how   = "inner",
                                on    = ["date"])
                            .pipe(self._opt_full_sample_decile)
                            .drop(columns = ["vol_rtn"])
                            .assign(
                                sample_group = "in_sample",
                                opt_date     = df_fs_resid.index.max(),
                                optimization = "full_sample",
                                country      = country,
                                group        = signal_name))
            
            
            is_model   = tmp_models["in_sample"]
            slice_date = (is_model
                    .resid
                    .index
                    .max())
            
            df_train_test_resid = (df_tmp_vol_rtn
                    .reset_index()
                    .merge(right = df_tmp_zscore, how = "inner", on = ["date"])
                    .assign(
                        sample_group = lambda x: np.where(x.date <= slice_date, "in_sample", "out_sample"),
                        y_pred       = lambda x: is_model.predict(sm.add_constant(x.z_score)),
                        resid        = lambda x: x.vol_rtn - x.y_pred))

            df_train_test_decile = (self
                                    ._get_train_test_decile(
                                        df         = df_train_test_resid, 
                                        q          = self.q, 
                                        slice_date = slice_date)
                                    .drop(columns = ["index"]))

            df_train_test_opt_rtn = (df_train_test_decile
                    .set_index("date")
                    .pipe(self._opt_train_test_decile, "vol_rtn")
                    .assign(
                        opt_date     = slice_date,
                        optimization = "train_test")
                    .drop(columns = ["vol_rtn", "y_pred", "z_score"]))
            
            df_expanding_rtn = (tmp_models
                    ["expanding_model"]
                    .params
                    .dropna()
                    .rename(columns = {"z_score": "beta"})
                    .merge(right = df_tmp_zscore,  how = "inner", on = ["date"])
                    .merge(right = df_tmp_vol_rtn, how = "inner", on = ["date"])
                    .assign(
                        y_pred = lambda x: (x.beta * x.z_score) + x.const,
                        resid  = lambda x: x.vol_rtn - x.y_pred)
                    #.reset_index()
                    #.assign(year = lambda x: x.date.dt.year)
                    #.loc[lambda x: x.year == x.year.max()]
                    #.set_index("date")
                    .rename(columns = {"group": "signal_group"}))
            
            df_exp_opt_rtn = (self
                              ._expanding_optimize(
                                  df          = df_expanding_rtn, 
                                  q           = self.q, 
                                  signal_name = "resid", 
                                  rtn_name    = "vol_rtn", 
                                  verbose     = True)
                              .drop(columns = [
                                  "beta", "const", "vol_rtn", "y_pred", 
                                  "z_score"])
                              .rename(columns = {
                                  "group"       : "decile_group",
                                  "signal_group": "group"})
                              .assign(
                                  optimization = "expanding",
                                  sample_group = "out_sample"))
            
            df_add = (pd
                    .concat([df_exp_opt_rtn, df_train_test_opt_rtn, df_fs_opt_rtn])
                    .merge(right = df_tmp_rtn, how = "inner", on = ["date", "security"]))
            
            df_lists.append(df_add)
            
        if verbose: print("Saving data\n")
        df_out = pd.concat(df_lists)
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
            
    opt = Optimizer()
    opt.get_optimized_zscore()
    opt.get_optimized_resid()
    
if __name__ == "__main__": main()