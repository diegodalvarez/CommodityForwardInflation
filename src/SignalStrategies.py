# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:11:25 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class SignalStrategies:
    
    def __init__(self) -> pd.DataFrame: 
        
        self.path      = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        
    def _get_factor(self) -> pd.DataFrame: 
        
        forward_path  = os.path.join(self.data_path, "InflationForward.parquet")
        surprise_path = os.path.join(self.data_path, "InflationSurprise.parquet")
        
        df_forward = (pd.read_parquet(
            path = forward_path, engine = "pyarrow").
            drop(columns = ["security"]).
            pivot(index = "date", columns = "country", values = "value").
            apply(lambda x: np.log(x).diff()).
            reset_index().
            melt(id_vars = "date").
            assign(group = "forward_inf"))
        
        df_surprise = (pd.read_parquet(
            path = surprise_path, engine = "pyarrow").
            pivot(index = "date", columns = "country", values = "value").
            diff().
            reset_index().
            melt(id_vars = "date").
            assign(group = "surprise_inf"))
        
        df_factor = (pd.concat([
            df_forward, df_surprise]).
            dropna())
        
        return df_factor
    
    def _get_fut_rtn(self) -> pd.DataFrame: 
        
        fut_path  = os.path.join(self.data_path, "FutPX.parquet")
        
        df_fut_rtn = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            pivot(index = "date", columns = "security", values = "PX_LAST").
            pct_change().
            reset_index().
            melt(id_vars = "date", value_name = "fut_rtn").
            dropna())
        
        return df_fut_rtn
    
    def _prep_data(self) -> None: 
        
        df_factor  = self._get_factor() 
        df_fut_rtn = self._get_fut_rtn()
        
        df_combined = (df_factor.merge(
            right = df_fut_rtn, how = "inner", on = ["date"]))
        
        return df_combined
    
    def _get_resid(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_tmp = (df.sort_index().assign(
            lag_value = lambda x: x.value.shift()).
            dropna())
            
        df_is_resid = (sm.OLS(
            endog = df_tmp.fut_rtn,
            exog  = sm.add_constant(df_tmp.lag_value)).
            fit().
            resid.
            to_frame(name = "resid").
            assign(
                sample_group = "is",
                lag_resid    = lambda x: x.resid.shift()).
            merge(right = df_tmp, how = "inner", on = ["date"]))
        
        df_oos_resid = (RollingOLS(
            endog     = df_tmp.fut_rtn,
            exog      = sm.add_constant(df_tmp.lag_value),
            expanding = True,
            min_nobs  = 30).
            fit().
            params.
            rename(columns = {
                "const"    : "alpha",
                "lag_value": "beta"}).
            merge(right = df_tmp, how = "inner", on = ["date"]).
            assign(
                resid        = lambda x: x.fut_rtn - (x.alpha + x.beta * x.lag_value),
                lag_resid    = lambda x: x.resid.shift(),
                sample_group = "os").
            drop(columns = ["alpha", "beta"]))
        
        df_out = (pd.concat([
            df_oos_resid, df_is_resid]))
        
        return df_out
    
    def _get_ols_regression(self, verbose: bool = True) -> None:
        
        if verbose: print("Getting OLS Regression Residuals In-Sample & Out-Sample")
        out_path = os.path.join(self.data_path, "OLSResid.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data collected")
            return None
        
        df_out = (self._prep_data().assign(
            group_var = lambda x: x.country + " " + x.security + " " + x.group).
            set_index("date").
            groupby("group_var").
            apply(self._get_resid, include_groups = False).
            reset_index().
            dropna().
            drop(columns = ["group_var"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_zscore(self, window: int = 30, verbose: bool = True) -> None: 
        if verbose: print("Getting Z-Score Values")
        out_path = os.path.join(self.data_path, "ZScore.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data collected")
            return None
        
        display(self._get_factor())
        return-1
        
        df_zscore = (self._get_factor().pivot(
            index = "date", columns = ["country", "group"], values = "value").
            apply(
                lambda x: (x - x.ewm(span = window, adjust = False).mean()) / 
                x.ewm(span = window, adjust = False).std()).
            reset_index().
            melt(id_vars = [("date", "")]).
            rename(columns = {("date", ""): "date"}).
            dropna())
        
        df_out = (self._get_fut_rtn().merge(
            right = df_zscore, how = "inner", on = ['date']))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _optimize_is_decile(self, df: pd.DataFrame, param_name: str, q: int = 10) -> pd.DataFrame: 
        
        df_decile = (df.sort_values(
            "date").
            assign(decile = lambda x: pd.qcut(x = x.value, q = q, labels = range(1, q+1)).shift()).
            dropna())
        
        df_decile_sharpe = (df_decile[
            ["decile", "fut_rtn"]].
            groupby("decile").
            agg(lambda x: x.mean() / x.std() * np.sqrt(252)))
        
        df_decile_tmp = (df_decile_sharpe.query(
            "decile == [1,2,9,10]").
            reset_index().
            assign(dec_group = lambda x: np.where(x.decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_decile_tmp.drop(
            columns = ["decile"]).
            groupby("dec_group").
            agg("prod").
            assign(signal_scaler = lambda x: np.where(x.fut_rtn > 0, 1, 0)).
            drop(columns = ["fut_rtn"]).
            merge(right = df_decile_tmp, how = "outer", on = ["dec_group"]).
            rename(columns = {"fut_rtn": "sharpe"}).
            merge(right = df_decile, how = "outer", on = ["decile"]).
            drop(columns = ["dec_group"]).
            assign(signal_rtn = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fut_rtn))
        
        return df_out

    def _optimize_os_decile(
        self, 
        df        : pd.DataFrame, 
        param_name: str, 
        q         : int = 10,
        min_obs   : int = 30) -> pd.DataFrame: 
    
        df      = df.sort_values("date")
        name    = df[["security", "group", "country"]].drop_duplicates().agg(" ".join, axis = 1).item()
        dates   = df.date.drop_duplicates().sort_values().to_list()
        results = []

        
        for i in tqdm(
                iterable = range(min_obs, len(dates)),
                desc     = "OOS Decile {}".format(name)):   
        
        #for i in range(min_obs, len(dates)):
            
            date  = dates[i]
            df_is = df.iloc[:i]
            df_os = df.iloc[i:i+1]
            
            if len(df_is) <= min_obs:
                continue
            
            _, bins = pd.qcut(
                x          = df_is[param_name],
                q          = q, 
                retbins    = True,
                duplicates = "drop")
            
            bins[0]  = -np.inf
            bins[-1] = np.inf
            
            df_is_decile = (df_is.assign(
                decile             = lambda x: pd.cut(
                    x              = x[param_name],
                    bins           = bins,
                    labels         = range(1, len(bins)),
                    include_lowest = True)))
            
            grp = (df_is_decile.groupby(
                "decile")
                ["fut_rtn"])
            
            sharpe  = (grp.mean() / grp.std()) * np.sqrt(252)
            df_tail = (sharpe.loc[
                sharpe.index.isin([1,2,9,10])].
                to_frame(name = "sharpe"))
            
            if df_tail.empty:
                continue
            
            df_tail["group"] = np.where(df_tail.index <= 2, "lgroup", "ugroup")
            sharpe_prod      = df_tail.groupby("group")["sharpe"].prod()
            
            l_signal = np.where(sharpe_prod.get("lgroup") > 0, 1, 0)
            u_signal = np.where(sharpe_prod.get("ugroup") > 0, 1, 0)
            
            sharpe_dict = sharpe.to_dict()
            last_is     = df_is_decile.iloc[[-1]].copy()
            last_is["signal_scaler"] = (np.select(
                condlist = [
                    last_is["decile"].astype(int) <= 2,
                    last_is["decile"].astype(int) >= 9],
                choicelist = [l_signal, u_signal],
                default    = np.nan))
            
            df_add = (df_os.assign(
                decile        = last_is["decile"].values[0],
                sharpe        = lambda x: x.decile.map(sharpe_dict),
                signal_scaler = last_is["signal_scaler"].values[0],
                signal_rtn    = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fut_rtn))
            
            '''
            df_add = (df_os.assign(
                decile = lambda x: pd.cut(
                    x      = x[param_name],
                    bins   = bins,
                    labels = range(1, len(bins))),
                signal_scaler = lambda x: np.select(
                    condlist   = [x.decile.astype(int) <= 2, x.decile.astype(int) >= 9],
                    choicelist = [l_signal, u_signal],
                    default    = np.nan),
                sharpe = lambda x: x.decile.map(sharpe_dict),
                signal_rtn = lambda x: np.sign(x.sharpe.astype(float) * x.signal_scaler) * x.fut_rtn))
            
            results.append(df_add)
            '''
            results.append(df_add)
        
        df_out = (pd.concat(
            objs         = results,
            ignore_index = True))
        
        return df_out
        
    def _optimize_zscore(self, param_name: str = "value", verbose: bool = True) -> None: 
        
        in_path  = os.path.join(self.data_path, "ZScore.parquet")
        out_path = os.path.join(self.data_path, "OptimizedZScore.parquet")
        
        if os.path.exists(out_path):
            if verbose: 
                print("Already Have Optimized Z-Scores")
                
            return None
        
        if verbose: print("Working on generating optimized z-scores")
        
        df_zscore = pd.read_parquet(path = in_path, engine = "pyarrow")
        
        df_is = (df_zscore.assign(
            group_var = lambda x: x.security + " " + x.country + " " + x.group).
            groupby("group_var").
            apply(self._optimize_is_decile, "value").
            reset_index().
            drop(columns = ["group_var", "level_1"]).
            assign(sample_group = "in_sample"))
        
        df_oos = (df_zscore.assign(
            group_var = lambda x: x.country + " " + x.group + " " + x.security).
            groupby("group_var").
            apply(self._optimize_os_decile, param_name, include_groups = False).
            reset_index().
            drop(columns = ["group_var", "level_1", "value"]).
            assign(sample_group = "out_sample"))
        
        df_out = pd.concat([df_is, df_oos])
        
        if verbose: print("Saving Results")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
signal_strategies = SignalStrategies()
#signal_strategies._get_ols_regression()
signal_strategies._get_zscore()
#signal_strategies._optimize_zscore()