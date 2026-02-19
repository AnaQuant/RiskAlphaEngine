import numpy as np
import pandas as pd
import os
from scipy.stats import norm

def smoothing_tstats(e):
    smt_param = 1
    if np.abs(e) > smt_param:
        return np.sign(e)
    else:
        return np.sin(e * np.pi / (2 * smt_param))

def log_return(spot):
    return np.log(spot).diff()

def common_term(ir_dom, ir_for, vol, T):
    res = (ir_dom - ir_for) * T / 100 + vol * vol * T / 2
    return res

def model_vol(real_vol, imp_vol):
    vol = np.maximum(real_vol, imp_vol / 100)
    return vol

class RiskReversalMomentum(object):

    def __init__(self, symbol, start, end, amount, data_folder="../data/", tc=0):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.tc = tc
        self.results = None
        self.get_data(data_folder=data_folder)
        self.decay = 0.97
        self.deltaShift = 0.4
        self.deltaATM = 1

    def get_data(self, data_folder=""):
        ccy_pair = self.symbol
        filename = f"mktdata_{ccy_pair}.csv"
        filepath = os.path.join(data_folder, filename) if data_folder else filename
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File '{filepath}' not found!")
        raw = pd.read_csv(filepath, index_col=0, parse_dates=True, dayfirst=True).dropna()
        raw['return'] = np.log(raw['Spot'] / raw['Spot'].shift(1))
        self.data = raw


    def realised_vol(self):
        spot_return = self.data['return']
        return_ewma = spot_return.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        adjReturn = spot_return - return_ewma
        adjReturnSqr = 252 * adjReturn * adjReturn
        variance_ewma = adjReturnSqr.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()

        return np.sqrt(variance_ewma)



    def tstat_deltaRR(self):
        RR = self.data['RiskRev']
        deltaRR = RR.diff()
        deltaRRsqr = deltaRR * deltaRR
        beta = deltaRR.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        gamma = deltaRRsqr.ewm(com=self.decay / (1.0 - self.decay), adjust=False).mean()
        window_size = round(1.5 * (1 + self.decay) / (1 - self.decay))
        stdErr = np.sqrt((gamma - beta * beta) / (window_size - 1))
        result = beta / stdErr

        return result


    def run_strategy(self):
        signal = self.tstat_deltaRR()
        signal = signal.apply(smoothing_tstats, args=())
        data = self.data.copy().dropna()
        data['position'] = signal
        data['strategy'] = data['position'].shift(1) * data['return']
        data['creturns'] =  self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data


    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to the symbol.
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = '%s | TC = %.4f' % (self.symbol, self.tc)
        self.results[['creturns', 'cstrategy']].plot(title=title,
                                                     figsize=(10, 6))

