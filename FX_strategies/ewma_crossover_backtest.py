import pandas as pd
import numpy as np
from FX_strategies.crossover_momentum import Strategy


class EWMACrossoverBacktester:
    """
    Vectorised backtester for the EWMA vol-normalised crossover strategy.

    Wraps FX_strategies.crossover_momentum.Strategy in the same interface
    as SMAVectorBacktester and MomVectorBacktester, enabling uniform
    multi-currency loops and portfolio construction.

    Parameters
    ----------
    symbol : str
        Currency pair label matching a column in the data CSV.
    start : str
        Start date in "MM-DD-YYYY" format.
    end : str
        End date in "MM-DD-YYYY" format.
    data_path : str
        Path to spots_currencies_universe.csv.
    windows_lst : list of tuple, optional
        EWMA window pairs (short_span, long_span). Defaults to the
        medium window set [(16,64), (32,128), (64,256)].
    vol_window : int
        Rolling volatility estimation window in trading days (default 60,
        approximately 3 months — adapted from the intraday default of 480).
    vol_scale : float
        Target volatility scaling factor applied to each signal (default 0.1).
    longonly : bool
        If True, negative signals are clipped to zero. Default False.
    annual : int
        Annualisation factor. Use 252 for daily data (adapted from the
        intraday default of 35040 in crossover_momentum.py).
    """

    def __init__(
        self,
        symbol,
        start,
        end,
        data_path='../data/spots_currencies_universe.csv',
        windows_lst=None,
        vol_window=60,
        vol_scale=0.1,
        longonly=False,
        annual=252,
    ):
        self.symbol      = symbol
        self.start       = start
        self.end         = end
        self.data_path   = data_path
        self.windows_lst = [(16, 64), (32, 128), (64, 256)] if windows_lst is None else windows_lst
        self.vol_window  = vol_window
        self.vol_scale   = vol_scale
        self.longonly    = longonly
        self.annual      = annual
        self.results     = None
        self.get_data()

    def get_data(self):
        raw = pd.read_csv(
            self.data_path, index_col=0, parse_dates=True, dayfirst=True
        ).dropna()
        raw.index = pd.to_datetime(raw.index, dayfirst=True)
        raw = pd.DataFrame(raw[self.symbol])
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: 'price'}, inplace=True)
        raw['return'] = np.log(raw['price'] / raw['price'].shift(1))
        self.data = raw

    def run_strategy(self):
        """
        Compute the vol-normalised EWMA signal, apply a one-day execution
        lag, and compute cumulative performance.

        Returns
        -------
        tuple of (float, float)
            (aperf, operf): gross cumulative return and out/underperformance
            relative to buy-and-hold, both rounded to 2 decimal places.

        Side effects
        ------------
        Populates self.results with columns:
            return, signal, strategy, creturns, cstrategy
        """
        strat = Strategy(
            windows_lst=self.windows_lst,
            vol_window=self.vol_window,
            vol_scale=self.vol_scale,
            longonly=self.longonly,
            annual=self.annual,
        )
        signal = strat.compute_signal(self.data['return'].dropna())

        results = pd.DataFrame({
            'return': self.data['return'],
            'signal': signal,
        }).dropna()

        results['strategy'] = results['signal'].shift(1) * results['return']
        results.dropna(inplace=True)
        results['creturns']  = results['return'].cumsum().apply(np.exp)
        results['cstrategy'] = results['strategy'].cumsum().apply(np.exp)
        self.results = results

        aperf = results['cstrategy'].iloc[-1]
        operf = aperf - results['creturns'].iloc[-1]
        return round(aperf, 2), round(operf, 2)

    def set_parameters(self, **kwargs):
        """Update any signal parameter and re-run get_data if data_path changes."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        if 'data_path' in kwargs:
            self.get_data()


if __name__ == '__main__':
    import os
    import sys
    project_root = os.path.abspath('..')
    sys.path.insert(0, project_root)
    data_path = os.path.join(project_root, 'data', 'spots_currencies_universe.csv')

    bt = EWMACrossoverBacktester(
        symbol='EURUSD',
        start='12-12-2005',
        end='03-24-2020',
        data_path=data_path,
    )
    aperf, operf = bt.run_strategy()
    print(f'EURUSD EWMA gross: {aperf:.4f}x  vs BnH: {operf:+.4f}x')
