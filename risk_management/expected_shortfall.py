"""
Expected Shortfall (ES/CVaR) Engine for Multi-Asset Portfolio
Supports historical simulation, parametric, and Monte Carlo methods
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ExpectedShortfallEngine:
    
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.portfolio_returns = None
        self.weights = None
        self.asset_returns = None
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate log returns from price data_ingestion"""
        return np.log(prices / prices.shift(1)).dropna()
    
    def create_portfolio(self, 
                        asset_returns: pd.DataFrame, 
                        weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Create portfolio returns from individual asset returns
        Default to equal weights if not specified
        """
        self.asset_returns = asset_returns
        
        if weights is None:
            # Equal weights
            n_assets = len(asset_returns.columns)
            weights = {asset: 1/n_assets for asset in asset_returns.columns}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        self.weights = weights
        
        # Calculate portfolio returns
        portfolio_returns = sum(asset_returns[asset] * weight 
                              for asset, weight in weights.items())
        
        self.portfolio_returns = portfolio_returns
        return portfolio_returns
    
    def historical_es(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Expected Shortfall using Historical Simulation
        Returns both VaR and ES
        """
        alpha = 1 - confidence_level
        
        # Calculate VaR (quantile)
        var = np.percentile(returns, alpha * 100)
        
        # Calculate ES (mean of losses beyond VaR)
        tail_losses = returns[returns <= var]
        es = tail_losses.mean() if len(tail_losses) > 0 else var
        
        return var, es
    
    def parametric_es(self, 
                     returns: pd.Series, 
                     confidence_level: float = 0.95,
                     distribution: str = 'normal') -> Tuple[float, float]:
        """
        Calculate Expected Shortfall using parametric approach
        Supports normal and t-distribution
        """
        alpha = 1 - confidence_level
        mu = returns.mean()
        sigma = returns.std()
        
        if distribution == 'normal':
            # Normal distribution
            var = stats.norm.ppf(alpha, mu, sigma)
            # ES formula for normal distribution
            es = mu - sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            
        elif distribution == 't':
            # Fit t-distribution
            params = stats.t.fit(returns)
            df, loc, scale = params
            
            var = stats.t.ppf(alpha, df, loc, scale)
            # ES formula for t-distribution (approximation)
            es = loc - scale * (stats.t.pdf(stats.t.ppf(alpha, df), df) * (df + (stats.t.ppf(alpha, df))**2) / 
                               (alpha * (df - 1)))
        
        return var, es
    
    def monte_carlo_es(self, 
                      returns: pd.Series, 
                      confidence_level: float = 0.95,
                      n_simulations: int = 10000) -> Tuple[float, float]:
        """
        Calculate Expected Shortfall using Monte Carlo simulation
        """
        alpha = 1 - confidence_level
        
        # Fit normal distribution to historical returns
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate Monte Carlo scenarios
        mc_returns = np.random.normal(mu, sigma, n_simulations)
        
        # Calculate VaR and ES
        var = np.percentile(mc_returns, alpha * 100)
        tail_losses = mc_returns[mc_returns <= var]
        es = tail_losses.mean()
        
        return var, es
    
    def rolling_es_analysis(self, 
                           returns: pd.Series, 
                           window: int = 252,
                           confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Perform rolling Expected Shortfall analysis
        """
        rolling_results = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            var, es = self.historical_es(window_returns, confidence_level)
            
            rolling_results.append({
                'date': returns.index[i],
                'var': var,
                'es': es,
                'return': returns.iloc[i]
            })
        
        return pd.DataFrame(rolling_results).set_index('date')


    def summary_statistics(self, returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate comprehensive summary statistics.

        Args:
            returns: Return series (uses portfolio returns if None)

        Returns:
            Dictionary with summary statistics
        """
        if returns is None:
            if self.portfolio_returns is None:
                raise ValueError("No return data available.")
            returns = self.portfolio_returns

        return {
            'observations': len(returns),
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'annualized_return': returns.mean() * 252,
            'annualized_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'min_return': returns.min(),
            'max_return': returns.max(),
            'var_95_1d': np.percentile(returns, 5),
            'var_99_1d': np.percentile(returns, 1)
        }
    