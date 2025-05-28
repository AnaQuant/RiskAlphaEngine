"""
VaR Calculation Engines

This module implements three core VaR methodologies:
1. Historical VaR - Direct percentile method
2. Parametric VaR - Variance-covariance with EWMA
3. Monte Carlo VaR - Full simulation approach
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VaREngine:
    """
    Comprehensive VaR calculation engine supporting multiple methodologies
    """
    
    def __init__(self, confidence_level: float = 0.05):
        """
        Initialize VaR engine
        
        Args:
            confidence_level: VaR confidence level (default 5% for 95% VaR)
        """
        self.confidence_level = confidence_level
        self.alpha = confidence_level
        
    def historical_var(self, returns: pd.Series, window: int = 250) -> Dict:
        """
        Historical VaR using direct percentile method
        
        Args:
            returns: Portfolio return series
            window: Lookback window for calculation
            
        Returns:
            Dictionary with VaR, Expected Shortfall, and metadata
        """
        if len(returns) < window:
            window = len(returns)
            
        # Get historical returns for the window
        hist_returns = returns.tail(window)
        
        # Calculate VaR as percentile
        var = np.percentile(hist_returns, self.alpha * 100)
        
        # Calculate Expected Shortfall (CVaR)
        tail_returns = hist_returns[hist_returns <= var]
        expected_shortfall = tail_returns.mean() if len(tail_returns) > 0 else var
        
        return {
            'VaR': var,
            'Expected_Shortfall': expected_shortfall,
            'method': 'Historical',
            'window': window,
            'observations': len(hist_returns),
            'tail_observations': len(tail_returns)
        }
    
    def parametric_var(self, returns: pd.Series, lambda_ewma: float = 0.94) -> Dict:
        """
        Parametric VaR using EWMA volatility estimation
        
        Args:
            returns: Portfolio return series
            lambda_ewma: EWMA decay factor (default 0.94 for RiskMetrics)
            
        Returns:
            Dictionary with VaR, Expected Shortfall, and metadata
        """
        # Calculate EWMA volatility
        ewma_var = self._calculate_ewma_variance(returns, lambda_ewma)
        current_vol = np.sqrt(ewma_var)
        
        # Assume normal distribution for parametric VaR
        z_score = stats.norm.ppf(self.alpha)
        mean_return = returns.mean()
        
        # Calculate VaR and ES
        var = mean_return + z_score * current_vol
        expected_shortfall = mean_return - current_vol * stats.norm.pdf(z_score) / self.alpha
        
        return {
            'VaR': var,
            'Expected_Shortfall': expected_shortfall,
            'method': 'Parametric',
            'volatility': current_vol,
            'mean_return': mean_return,
            'lambda_ewma': lambda_ewma
        }
    
    def monte_carlo_var(self, returns: pd.Series, n_simulations: int = 10000, 
                       distribution: str = 'normal') -> Dict:
        """
        Monte Carlo VaR using simulated returns
        
        Args:
            returns: Portfolio return series
            n_simulations: Number of Monte Carlo simulations
            distribution: 'normal' or 't' for Student-t distribution
            
        Returns:
            Dictionary with VaR, Expected Shortfall, and metadata
        """
        mean_return = returns.mean()
        vol = returns.std()
        
        if distribution == 'normal':
            # Normal distribution simulation
            simulated_returns = np.random.normal(mean_return, vol, n_simulations)
        elif distribution == 't':
            # Student-t distribution (more realistic for financial returns)
            # Estimate degrees of freedom
            df = self._estimate_t_degrees_of_freedom(returns)
            simulated_returns = stats.t.rvs(df, loc=mean_return, scale=vol, size=n_simulations)
        else:
            raise ValueError("Distribution must be 'normal' or 't'")
        
        # Calculate VaR and ES from simulations
        var = np.percentile(simulated_returns, self.alpha * 100)
        tail_returns = simulated_returns[simulated_returns <= var]
        expected_shortfall = tail_returns.mean()
        
        return {
            'VaR': var,
            'Expected_Shortfall': expected_shortfall,
            'method': f'Monte Carlo ({distribution})',
            'n_simulations': n_simulations,
            'distribution': distribution,
            'mean_return': mean_return,
            'volatility': vol
        }
    
    def _calculate_ewma_variance(self, returns: pd.Series, lambda_ewma: float) -> float:
        """Calculate EWMA variance (RiskMetrics methodology)"""
        if len(returns) == 0:
            return 0
            
        # Initialize with first observation
        ewma_var = returns.iloc[0] ** 2
        
        # Update with EWMA formula
        for ret in returns.iloc[1:]:
            ewma_var = lambda_ewma * ewma_var + (1 - lambda_ewma) * ret ** 2
            
        return ewma_var
    
    def _estimate_t_degrees_of_freedom(self, returns: pd.Series) -> float:
        """Estimate degrees of freedom for Student-t distribution"""
        # Simple method: use excess kurtosis to estimate df
        kurtosis = returns.kurtosis()
        if kurtosis > 0:
            # df = 6/excess_kurtosis + 4 (rough approximation)
            df = max(6.0 / kurtosis + 4, 3)  # Minimum df = 3
        else:
            df = 10  # Default if kurtosis is not positive
        return df

class PortfolioRiskAnalyzer:
    """
    Portfolio risk analysis combining multiple VaR methods
    """
    
    def __init__(self, confidence_level: float = 0.05):
        self.var_engine = VaREngine(confidence_level)
        self.confidence_level = confidence_level
        
    def analyze_portfolio_risk(self, portfolio_returns: pd.Series, 
                             portfolio_value: float = 1000000) -> pd.DataFrame:
        """
        Comprehensive portfolio risk analysis
        
        Args:
            portfolio_returns: Portfolio return series
            portfolio_value: Portfolio value for VaR scaling
            
        Returns:
            DataFrame with all VaR methods comparison
        """
        results = []
        
        # Historical VaR
        hist_result = self.var_engine.historical_var(portfolio_returns)
        results.append({
            'Method': 'Historical',
            'VaR_%': hist_result['VaR'] * 100,
            'VaR_Dollar': hist_result['VaR'] * portfolio_value,
            'ES_%': hist_result['Expected_Shortfall'] * 100,
            'ES_Dollar': hist_result['Expected_Shortfall'] * portfolio_value,
            'Window': hist_result['window']
        })
        
        # Parametric VaR
        param_result = self.var_engine.parametric_var(portfolio_returns)
        results.append({
            'Method': 'Parametric',
            'VaR_%': param_result['VaR'] * 100,
            'VaR_Dollar': param_result['VaR'] * portfolio_value,
            'ES_%': param_result['Expected_Shortfall'] * 100,
            'ES_Dollar': param_result['Expected_Shortfall'] * portfolio_value,
            'Volatility_%': param_result['volatility'] * 100
        })
        
        # Monte Carlo VaR (Normal)
        mc_normal_result = self.var_engine.monte_carlo_var(portfolio_returns, distribution='normal')
        results.append({
            'Method': 'Monte Carlo (Normal)',
            'VaR_%': mc_normal_result['VaR'] * 100,
            'VaR_Dollar': mc_normal_result['VaR'] * portfolio_value,
            'ES_%': mc_normal_result['Expected_Shortfall'] * 100,
            'ES_Dollar': mc_normal_result['Expected_Shortfall'] * portfolio_value,
            'Simulations': mc_normal_result['n_simulations']
        })
        
        # Monte Carlo VaR (Student-t)
        mc_t_result = self.var_engine.monte_carlo_var(portfolio_returns, distribution='t')
        results.append({
            'Method': 'Monte Carlo (Student-t)',
            'VaR_%': mc_t_result['VaR'] * 100,
            'VaR_Dollar': mc_t_result['VaR'] * portfolio_value,
            'ES_%': mc_t_result['Expected_Shortfall'] * 100,
            'ES_Dollar': mc_t_result['Expected_Shortfall'] * portfolio_value,
            'Simulations': mc_t_result['n_simulations']
        })
        
        return pd.DataFrame(results)
    
    def rolling_var_analysis(self, portfolio_returns: pd.Series, 
                           method: str = 'historical', window: int = 250) -> pd.DataFrame:
        """
        Rolling VaR analysis for time series visualization
        
        Args:
            portfolio_returns: Portfolio return series
            method: VaR method ('historical', 'parametric', 'monte_carlo')
            window: Rolling window size
            
        Returns:
            DataFrame with rolling VaR and ES
        """
        rolling_results = []
        
        for i in range(window, len(portfolio_returns)):
            window_returns = portfolio_returns.iloc[i-window:i]
            
            if method == 'historical':
                result = self.var_engine.historical_var(window_returns, window)
            elif method == 'parametric':
                result = self.var_engine.parametric_var(window_returns)
            elif method == 'monte_carlo':
                result = self.var_engine.monte_carlo_var(window_returns, n_simulations=5000)
            else:
                raise ValueError("Method must be 'historical', 'parametric', or 'monte_carlo'")
            
            rolling_results.append({
                'Date': portfolio_returns.index[i],
                'VaR': result['VaR'],
                'Expected_Shortfall': result['Expected_Shortfall'],
                'Method': method
            })
        
        return pd.DataFrame(rolling_results).set_index('Date')

def create_sample_portfolio_returns(n_days: int = 1000, 
                                  annual_return: float = 0.08,
                                  annual_vol: float = 0.15) -> pd.Series:
    """
    Create sample portfolio returns for demonstration
    
    Args:
        n_days: Number of days to simulate
        annual_return: Expected annual return
        annual_vol: Annual volatility
        
    Returns:
        Simulated portfolio returns
    """
    np.random.seed(42)  # For reproducibility
    
    # Convert annual to daily
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    # Generate returns with some autocorrelation and fat tails
    returns = []
    prev_return = 0
    
    for _ in range(n_days):
        # Add some autocorrelation and regime switching
        shock = np.random.normal(0, 1)
        if np.random.random() < 0.05:  # 5% chance of fat tail event
            shock *= 3  # 3x normal shock for tail events
            
        # AR(1) process with regime switching
        current_return = 0.1 * prev_return + daily_return + daily_vol * shock
        returns.append(current_return)
        prev_return = current_return
    
    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    return pd.Series(returns, index=dates, name='Portfolio_Returns')