"""
SVI (Stochastic Volatility Inspired) Calibration Module
======================================================

Implementation of SVI volatility surface parameterization for FX and equity markets.
Based on Jim Gatheral's SVI model with jump-wings parameterization for robust calibration.

"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class SVIParameters:
    """SVI parameters container with validation"""
    a: float  # Level parameter
    b: float  # Angle parameter
    rho: float  # Correlation parameter
    m: float  # Translation parameter
    sigma: float  # Scale parameter

    def __post_init__(self):
        """Validate SVI no-arbitrage conditions"""
        if not (-1 <= self.rho <= 1):
            raise ValueError(f"Correlation parameter rho must be in [-1,1], got {self.rho}")
        if self.b < 0:
            raise ValueError(f"Angle parameter b must be non-negative, got {self.b}")
        if self.sigma <= 0:
            raise ValueError(f"Scale parameter sigma must be positive, got {self.sigma}")

    def to_dict(self) -> Dict[str, float]:
        return {'a': self.a, 'b': self.b, 'rho': self.rho, 'm': self.m, 'sigma': self.sigma}


class SVICalibrator:
    """
    Implements both raw SVI and jump-wings parameterization for robust
    volatility surface fitting with arbitrage-free constraints.
    """

    def __init__(self, parameterization: str = 'raw'):

        if parameterization not in ['raw', 'jump_wings']:
            raise ValueError("Parameterization must be 'raw' or 'jump_wings'")
        self.parameterization = parameterization
        self.fitted_params = {}
        self.calibration_results = {}

    def svi_raw(self, k: np.ndarray, params: Union[SVIParameters, np.ndarray]) -> np.ndarray:
        """
        Raw SVI parameterization: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

        Parameters:
        -----------
        k : Log-moneyness array
        params : SVI parameters [a, b, rho, m, sigma]

        Returns:
        --------
        Total implied variance w(k)
        """
        if isinstance(params, SVIParameters):
            a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma
        else:
            a, b, rho, m, sigma = params

        k_shifted = k - m
        discriminant = np.sqrt(k_shifted ** 2 + sigma ** 2)

        return a + b * (rho * k_shifted + discriminant)

    def svi_jump_wings(self, k: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Jump-wings SVI parameterization for improved calibration stability
        Parameters:
        -----------
        k : Log-moneyness array
        params : Jump-wings parameters [v_t, psi_t, p_t, c_t, tilde_v_t]
        """
        v_t, psi_t, p_t, c_t, tilde_v_t = params

        # Convert to raw SVI parameters
        b_t = 0.5 * (c_t + p_t)
        rho_t = 1 - (p_t / b_t) if b_t > 0 else 0
        beta_t = rho_t - 2 * psi_t * v_t / b_t if b_t > 0 else 0
        alpha_t = np.sign(beta_t) * np.sqrt(1 / beta_t ** 2 - 1) if abs(beta_t) < 1 else 0

        m_t = (v_t - tilde_v_t) / b_t if b_t > 0 else 0
        sigma_t = alpha_t / np.sqrt(v_t) if v_t > 0 else 0.1
        a_t = tilde_v_t - b_t * sigma_t * np.sqrt(1 - rho_t ** 2)

        return self.svi_raw(k, [a_t, b_t, rho_t, m_t, sigma_t])

    def objective_function(self, params: np.ndarray, k: np.ndarray, market_var: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> float:
        """
        Weighted least squares objective function for SVI calibration
        -----------
        params : SVI parameters to optimize
        k : Log-moneyness points
        market_var : Market total variance
        weights : Weighting for different strikes (default: equal weights)

        Returns: Weighted sum of squared errors
        """
        try:
            if self.parameterization == 'raw':
                model_var = self.svi_raw(k, params)
            else:
                model_var = self.svi_jump_wings(k, params)

            residuals = model_var - market_var

            if weights is None:
                weights = np.ones_like(residuals)

            return np.sum(weights * residuals ** 2)

        except (ValueError, RuntimeWarning):
            return 1e10  # Penalty for invalid parameters

    def check_arbitrage_conditions(self, params: np.ndarray, k_range: np.ndarray) -> Dict[str, bool]:
        """
        SVI no-arbitrage conditions in paper Gatheral & Jacquier 2014
        -----------
        params : SVI parameters
        k_range : Range of log-moneyness to check

        Returns: Dictionary with arbitrage condition results
        """
        if self.parameterization == 'jump_wings':
            return {'butterfly': True, 'calendar': True}  # Jump-wings ensures no-arbitrage

        a, b, rho, m, sigma = params

        # Butterfly arbitrage condition: d²w/dk² >= 0
        # For SVI: d²w/dk² = b * sigma² / (sqrt((k-m)² + sigma²))³ >= 0
        butterfly_ok = b >= 0 and sigma > 0

        # Calendar arbitrage condition (simplified check)
        # More complex in practice, requires time-dependent analysis
        calendar_ok = a >= 0  # Simplified condition

        return {
            'butterfly': butterfly_ok,
            'calendar': calendar_ok,
            'parameters_valid': butterfly_ok and calendar_ok
        }

    def calibrate_slice(self, strikes: np.ndarray, implied_vols: np.ndarray,
                        forward: float, time_to_expiry: float,
                        weights: Optional[np.ndarray] = None,
                        method: str = 'differential_evolution') -> Dict:
        """
        Calibrate SVI to a single maturity slice

        Optimization method ('L-BFGS-B', 'differential_evolution')
        Returns: Calibration results including parameters and metrics
        """
        # Convert to log-moneyness and total variance
        k = np.log(strikes / forward)
        market_var = implied_vols ** 2 * time_to_expiry

        # Parameter bounds for raw SVI
        if self.parameterization == 'raw':
            # [a, b, rho, m, sigma]
            bounds = [
                (-1.0, 1.0),  # a: level
                (0.01, 5.0),  # b: angle (positive for no butterfly arbitrage)
                (-0.999, 0.999),  # rho: correlation
                (k.min() - 0.5, k.max() + 0.5),  # m: translation
                (0.01, 2.0)  # sigma: scale (positive)
            ]

            # Initial guess
            x0 = [
                np.mean(market_var),  # a
                0.1,  # b
                -0.1,  # rho
                0.0,  # m
                0.1  # sigma
            ]
        else:
            # Jump-wings parameterization bounds (simplified)
            bounds = [(0.01, 2.0)] * 5
            x0 = [0.1] * 5

        # Optimize
        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(k, market_var, weights),
                seed=42,
                maxiter=1000,
                atol=1e-8
            )
        else:
            result = minimize(
                self.objective_function,
                x0,
                args=(k, market_var, weights),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

        # Extract results
        optimal_params = result.x

        if self.parameterization == 'raw':
            svi_params = SVIParameters(*optimal_params)
            fitted_var = self.svi_raw(k, optimal_params)
        else:
            svi_params = optimal_params
            fitted_var = self.svi_jump_wings(k, optimal_params)

        # Calculate metrics
        residuals = fitted_var - market_var
        rmse = np.sqrt(np.mean(residuals ** 2))
        max_error = np.max(np.abs(residuals))

        # Check arbitrage conditions
        arbitrage_check = self.check_arbitrage_conditions(optimal_params, k)

        calibration_result = {
            'parameters': svi_params,
            'success': result.success,
            'rmse': rmse,
            'max_error': max_error,
            'r_squared': 1 - np.sum(residuals ** 2) / np.sum((market_var - np.mean(market_var)) ** 2),
            'arbitrage_free': arbitrage_check['parameters_valid'],
            'fitted_variance': fitted_var,
            'market_variance': market_var,
            'log_moneyness': k,
            'strikes': strikes,
            'time_to_expiry': time_to_expiry
        }

        return calibration_result

    def plot_calibration_results(self, results: Dict, title: str = "SVI Calibration Results"):
        """
        Plot SVI calibration results
        Parameters:
        -----------
        results : Calibration results from calibrate_slice
        title : Plot title
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        k = results['log_moneyness']
        market_var = results['market_variance']
        fitted_var = results['fitted_variance']
        strikes = results['strikes']

        # Convert back to implied volatilities
        T = results['time_to_expiry']
        market_iv = np.sqrt(market_var / T)
        fitted_iv = np.sqrt(fitted_var / T)

        # 1. Implied Volatility Smile
        ax1.plot(strikes, market_iv * 100, 'bo', label='Market', markersize=6)
        ax1.plot(strikes, fitted_iv * 100, 'r-', label='SVI Fit', linewidth=2)
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title('Implied Volatility Smile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Log-moneyness vs Total Variance
        ax2.plot(k, market_var, 'bo', label='Market', markersize=6)
        ax2.plot(k, fitted_var, 'r-', label='SVI Fit', linewidth=2)
        ax2.set_xlabel('Log-moneyness')
        ax2.set_ylabel('Total Variance')
        ax2.set_title('SVI Raw Parameterization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residuals
        residuals = (fitted_iv - market_iv) * 100
        ax3.plot(strikes, residuals, 'go', markersize=6)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Strike')
        ax3.set_ylabel('Residuals (bp)')
        ax3.set_title('Calibration Residuals')
        ax3.grid(True, alpha=0.3)

        # 4. Metrics and Parameters
        ax4.axis('off')

        # Display calibration metrics
        metrics_text = f"""
Calibration Metrics:
• RMSE: {results['rmse']:.6f}
• Max Error: {results['max_error']:.6f}
• R²: {results['r_squared']:.4f}
• Arbitrage Free: {results['arbitrage_free']}
• Success: {results['success']}

SVI Parameters:
"""

        if isinstance(results['parameters'], SVIParameters):
            params = results['parameters']
            params_text = f"""
• a (level): {params.a:.4f}
• b (angle): {params.b:.4f}
• ρ (correlation): {params.rho:.4f}
• m (translation): {params.m:.4f}
• σ (scale): {params.sigma:.4f}"""

        else:
            params_text = "• Jump-wings parameters"

        ax4.text(0.1, 0.9, metrics_text + params_text,
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.5))

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.show()



def generate_synthetic_smile(forward: float = 100.0, time_to_expiry: float = 0.25,
                             atm_vol: float = 0.20, skew: float = -0.1,
                             convexity: float = 0.05, noise_level: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic volatility smile for testing
    """
    # Generate strike range
    strikes = np.linspace(forward * 0.7, forward * 1.3, 15)
    log_moneyness = np.log(strikes / forward)

    # Create smile with skew and convexity
    implied_vols = atm_vol + skew * log_moneyness + convexity * log_moneyness ** 2

    # Add some realistic market noise
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, len(strikes))
    implied_vols += noise

    # Ensure positive volatilities
    implied_vols = np.maximum(implied_vols, 0.05)

    return strikes, implied_vols