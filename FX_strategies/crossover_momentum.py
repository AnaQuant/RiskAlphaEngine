from abc import ABCMeta, abstractmethod
from typing import TypeVar
import numpy as np
import pandas as pd
from datetime import datetime

PandasDataframe = TypeVar('pandas.core.frame.DataFrame')

#logger.info('###############################################')
#logger.info('Running Strategy: {}'.format(str(datetime.now())))


class BaseMomentum:
	__metaclass__ = ABCMeta

	@staticmethod
	def std_vol(returns: pd.Series, window=480, annual=35040) -> pd.Series:
		"""
		Measure standard
		:param returns:
		:param window: window over which to calculate std
		:param annual: annualisation factor
		:return: rolling standard deviation
		"""
		return returns.rolling(window=window, min_periods=window).std() * np.sqrt(annual)

	@staticmethod
	def macd_ewma(price: pd.Series, short_window: int, long_window: int) -> pd.Series:
		"""
		Exponentially weighted moving average crossover binary signal
		:param price: df of price
		:param short_window: int of short window
		:param long_window: int of long window
		:return: df with binary long short
		"""
		signal = pd.Series(index=price.index, dtype=float)
		hl_short = np.log(0.5) / (np.log(1 - 2 / (short_window + 1)))
		hl_long = np.log(0.5) / (np.log(1 - 2 / (long_window + 1)))
		short_mavg = price.ewm(halflife=hl_short, adjust=False, min_periods=short_window).mean()
		long_mavg = price.ewm(halflife=hl_long, adjust=False, min_periods=long_window).mean()
		signal[short_mavg > long_mavg] = 1
		signal[short_mavg < long_mavg] = -1
		return signal

	@abstractmethod
	def compute_signal(self, ccy_market_data: pd.DataFrame) -> pd.DataFrame:
		pass


class Strategy(BaseMomentum):

	def __init__(self, windows_lst=None, vol_window=None, vol_scale=None, longonly=None, annual=None):
		"""
		:param windows_lst: List of tuples defining the crossover windows
		:param vol_window: Window over which to measure the volatility
		:param vol_scale: Volatility scaling of the underlying signal
		:param longonly: If True then only allow long signal
		:param annual: annualisation factor
		"""
		self.windows_lst = [(16, 64), (32, 128), (64, 256)] if windows_lst is None else windows_lst
		self.vol_window = 480 if vol_window is None else vol_window
		self.vol_scale = 0.1 if vol_scale is None else vol_scale
		self.longonly = True if longonly is None else longonly
		self.annual = 35040 if annual is None else annual

	def compute_signal(self, returns_ser: pd.Series) -> pd.Series:
		"""
		Simple exponentially weighted crossover momentum strategy
		:param returns_ser: dataframe of the log returns of the asset(s)
		:return: dataframe of the signals for asset(s)
		"""
		#logger.info('Running crossover momentum')

		#logger.info('Calculating volatility. window: {}, annual: {}'.format(self.vol_window, self.annual))
		raw_vol = BaseMomentum.std_vol(returns_ser, window=self.vol_window, annual=self.annual)

		# Vol normalise returns create synthetic price
		#logger.info('Creating volatility adjusted spot. annaul: {}'.format(self.annual))
		adj_return = returns_ser / raw_vol * np.sqrt(self.annual)
		adj_spot = adj_return.cumsum()

		signals = pd.Series(0, index=adj_spot.index, dtype=float)
		for window in self.windows_lst:
			#logger.info('Calculating crossover momentum: {}, vol_scale: {}'.format(window, self.vol_scale))
			momentum_df = self.macd_ewma(adj_spot, window[0], window[1])
			scaled_momentum = (momentum_df / raw_vol) * self.vol_scale
			signals = signals + scaled_momentum

		signals = signals / len(self.windows_lst)
		signals = signals.clip(-1, 1)
		if self.longonly:
			#logger.info('Restricting to long only')
			signals[signals < 0] = 0

		return signals
