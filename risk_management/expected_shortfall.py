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
            assert df > 1, f"t-distribution requires df > 1 for finite ES; got df={df:.2f}"

            var = stats.t.ppf(alpha, df, loc, scale)
            # Closed-form ES for Student-t (standard parameterisation, then shift/scale)
            z_alpha = stats.t.ppf(alpha, df)          # standard quantile
            f_z     = stats.t.pdf(z_alpha, df)        # standard pdf at that quantile
            es = loc - scale * (f_z / alpha) * (df + z_alpha ** 2) / (df - 1)

        return var, es
    
    def monte_carlo_es(self,
                      returns: pd.Series,
                      confidence_level: float = 0.95,
                      n_simulations: int = 10000,
                      random_state: int = 42) -> Tuple[float, float]:
        """
        Calculate Expected Shortfall using Monte Carlo simulation
        """
        np.random.seed(random_state)
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


    def component_es(self,
                    asset_returns: pd.DataFrame,
                    weights: Dict[str, float],
                    confidence_level: float = 0.95,
                    delta: float = 0.01) -> Dict:
        """
        Decompose portfolio ES into per-asset component contributions via numerical perturbation.

        Args:
            asset_returns:    DataFrame of individual asset return series (columns = assets).
            weights:          Dict mapping asset name → portfolio weight (will be normalised).
            confidence_level: Confidence level for ES computation (e.g. 0.95).
            delta:            Weight perturbation size for numerical differentiation.

        Returns:
            Dict with portfolio_es, component_es, percent_contribution, marginal_es (all pd.Series).
        """
        # Normalise weights
        total = sum(weights.values())
        w = {k: v / total for k, v in weights.items()}
        assets = list(w.keys())
        w_vec = np.array([w[a] for a in assets])

        def _portfolio_es(wts: np.ndarray) -> float:
            port_ret = asset_returns[assets].values @ wts
            port_series = pd.Series(port_ret)
            _, es = self.historical_es(port_series, confidence_level)
            return es

        base_es = _portfolio_es(w_vec)

        marginal = np.zeros(len(assets))
        for i in range(len(assets)):
            w_perturbed = w_vec.copy()
            w_perturbed[i] += delta
            w_perturbed /= w_perturbed.sum()   # renormalise
            es_perturbed = _portfolio_es(w_perturbed)
            marginal[i] = (es_perturbed - base_es) / delta

        component = marginal * w_vec
        # Rescale so components sum exactly to base_es
        if component.sum() != 0:
            component = component * (base_es / component.sum())

        marginal_s   = pd.Series(marginal,   index=assets, name='marginal_es')
        component_s  = pd.Series(component,  index=assets, name='component_es')
        pct_s        = pd.Series(component / base_es * 100, index=assets, name='pct_contribution')

        return {
            'portfolio_es':       base_es,
            'component_es':       component_s,
            'percent_contribution': pct_s,
            'marginal_es':        marginal_s,
        }

    def acerbi_szekely_test(self,
                            returns: pd.Series,
                            es_series: pd.Series,
                            var_series: Optional[pd.Series] = None,
                            confidence_level: float = 0.95,
                            n_bootstrap: int = 500,
                            random_state: int = 42) -> Dict:
        """
        Acerbi-Szekely (2014) Z1 test for ES model validity.

        Tests whether the rolling ES forecasts are consistent with the realised tail losses.
        Z1 ≈ 0 under a correctly specified model; significantly negative → ES underestimated.

        Args:
            returns:          Realised return series.
            es_series:        Rolling ES forecasts aligned to returns index (negative values).
            var_series:       Rolling VaR forecasts (optional; derived from es_series if None).
            confidence_level: Alpha = 1 - confidence_level (tail probability).
            n_bootstrap:      Bootstrap replications for p-value estimation.
            random_state:     Seed for bootstrap reproducibility.

        Returns:
            Dict with Z1_stat, p_value, n_exceedances, interpretation.
        """
        alpha = 1 - confidence_level
        aligned = pd.DataFrame({'ret': returns, 'es': es_series}).dropna()
        if var_series is not None:
            aligned['var'] = var_series.reindex(aligned.index)
        else:
            aligned['var'] = aligned['es']   # use ES as the breach threshold if no VaR provided

        exceedances = aligned[aligned['ret'] < aligned['var']]
        n_exc = len(exceedances)
        T = len(aligned)

        if n_exc == 0 or aligned['es'].abs().max() < 1e-12:
            return {
                'Z1_stat': np.nan, 'p_value': np.nan,
                'n_exceedances': 0,
                'interpretation': 'No exceedances — cannot compute Z1.',
            }

        z1_obs = (exceedances['ret'] / exceedances['es'].abs()).sum() / (T * alpha) + 1.0

        # Bootstrap p-value: under H0, permute the return series and recompute Z1
        rng = np.random.default_rng(random_state)
        ret_vals = aligned['ret'].values
        es_vals  = aligned['es'].values
        var_vals = aligned['var'].values

        boot_z1 = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            perm = rng.permutation(len(ret_vals))
            r_b = ret_vals[perm]
            exc_mask = r_b < var_vals
            n_exc_b = exc_mask.sum()
            if n_exc_b == 0:
                boot_z1[b] = 0.0
            else:
                boot_z1[b] = (r_b[exc_mask] / np.abs(es_vals[exc_mask])).sum() / (T * alpha) + 1.0

        p_value = float((boot_z1 <= z1_obs).mean())

        if p_value < 0.05:
            interp = 'Reject H₀ at 5% — ES appears systematically underestimated (Z1 < 0).'
        else:
            interp = 'Fail to reject H₀ — ES estimates are consistent with realised tail losses.'

        return {
            'Z1_stat':        float(z1_obs),
            'p_value':        p_value,
            'n_exceedances':  n_exc,
            'interpretation': interp,
        }

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
    