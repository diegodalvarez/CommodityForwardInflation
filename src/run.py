
from PrepData import PrepData
from Backtesting import Backtesting
from SignalOptimizer import Optimizer
from FactorAnalysis import FactorModel
from SignalGenerator import SignalStrategies
from CrossSectionBacktest import CrossSectionBacktest

def main() -> None: 

    # Prep Data
    data_prep = PrepData()
    data_prep._get_fut_data()
    data_prep._get_forward_inflation()
    data_prep._get_inflation_surprise()
    data_prep._combine_fred_data()
    data_prep.vol_target_rtn()

    # Generate Siganls
    signal_strategies = SignalStrategies()
    signal_strategies.get_raw_factor()
    signal_strategies.get_zscore()
    signal_strategies.get_ols_resid_regression()
    signal_strategies.get_ols_pred_regression()


    # run the precursory factor analysis     
    factor_model = FactorModel()
    factor_model.linear_spot_fred_models()
    factor_model.logit_spot_fred_models()
    factor_model.inflation_measure_factor()
    factor_model.inflation_signal_factor()
    factor_model.raw_zscore_port_trend_exposure()

    # Optimize Signals
    opt = Optimizer()
    opt.get_optimized_zscore()
    opt.get_optimized_resid()

    # Generating Backtests for 
    backtesting = Backtesting()
    backtesting.get_signal_backtest()
    backtesting.get_residual_backtest()
    backtesting.get_pred_backtest()
    backtesting.get_opt_zscore_backtest()
    backtesting.get_opt_residual_backtest()

    # develop the cross-sectional backtest
    cross_backtest  = CrossSectionBacktest()
    cross_backtest.get_residual_legs()

if __name__ == "__main__": main()