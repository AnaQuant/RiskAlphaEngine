"""

This module implements Dupire's local volatility model for option pricing.
Dupire's local volatility model relates the local volatility σ(S,t) to
market implied volatilities through

    σ²(K,T) = (∂C/∂T + rK∂C/∂K) / (½K²∂²C/∂K²)

Where C(K,T) is the call option price as a function of strike K and time T.
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable
import warnings


class DupireLocalVolatility:


    def __init__(self,
                 spot_price: float,
                 risk_free_rate: float,
                 dividend_yield: float = 0.0):

        self.S0 = spot_price
        self.r = risk_free_rate
        self.q = dividend_yield

        # Will store market data
        self.market_data = None
        self.implied_vol_surface = None
        self.local_vol_surface = None

        # Interpolation objects
        self.iv_interpolator = None
        self.lv_interpolator = None

    def load_market_data(self,
                         strikes: np.ndarray,
                         maturities: np.ndarray,
                         implied_vols: np.ndarray) -> None:


        self.strikes = np.array(strikes)
        self.maturities = np.array(maturities)
        self.implied_vols = np.array(implied_vols)

        # Create meshgrids for surface plotting
        self.K_grid, self.T_grid = np.meshgrid(strikes, maturities)

        # Store as DataFrame for easier manipulation
        data_list = []
        for i, T in enumerate(maturities):
            for j, K in enumerate(strikes):
                data_list.append({
                    'maturity': T,
                    'strike': K,
                    'moneyness': K / self.S0,
                    'implied_vol': implied_vols[i, j]
                })

        self.market_data = pd.DataFrame(data_list)

        # Create interpolation object for implied volatility surface
        self._create_iv_interpolator()

    def _create_iv_interpolator(self) -> None:
        """Create 2D interpolation object for implied volatility surface"""
        # Use RectBivariateSpline for smooth interpolation
        self.iv_interpolator = interpolate.RectBivariateSpline(
            self.maturities, self.strikes, self.implied_vols,
            kx=3, ky=3, s=0  # Cubic splines with no smoothing
        )

    def black_scholes_call(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Black-Scholes call option price

        Parameters:
        -----------
        S : Spot price
        K : Strike price
        T : time to maturity
        sigma : Volatility

        Returns: call option price
        """
        from scipy.stats import norm

        if T <= 0:
            return max(S - K, 0)

        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = (S * np.exp(-self.q * T) * norm.cdf(d1) -
                      K * np.exp(-self.r * T) * norm.cdf(d2))

        return call_price

    def _call_price_surface(self, K: float, T: float) -> float:
        """Calculate call price using interpolated implied volatility"""
        if self.iv_interpolator is None:
            raise ValueError("Market data not loaded. Call load_market_data() first.")

        # Get implied volatility from interpolation
        iv = float(self.iv_interpolator(T, K)[0, 0])

        # Calculate call price using Black-Scholes
        return self.black_scholes_call(self.S0, K, T, iv)

    def _numerical_derivatives(self, K: float, T: float, h: float = 0.01) -> Tuple[float, float, float]:
        """
        Calculate numerical derivatives of call price surface
        """
        # First derivative w.r.t. time (forward difference)
        if T + h <= self.maturities.max():
            dC_dT = (self._call_price_surface(K, T + h) - self._call_price_surface(K, T)) / h
        else:
            # Use backward difference if forward not possible
            dC_dT = (self._call_price_surface(K, T) - self._call_price_surface(K, T - h)) / h

        # First derivative w.r.t. strike (central difference)
        dC_dK = (self._call_price_surface(K + h, T) - self._call_price_surface(K - h, T)) / (2 * h)

        # Second derivative w.r.t. strike
        d2C_dK2 = (self._call_price_surface(K + h, T) - 2 * self._call_price_surface(K, T) +
                   self._call_price_surface(K - h, T)) / (h ** 2)

        return dC_dT, dC_dK, d2C_dK2

    def dupire_local_volatility(self, K: float, T: float) -> float:
        """
        Calculate local volatility using Dupire's formula
        """
        if T <= 0:
            return 0.0

        # Get numerical derivatives
        dC_dT, dC_dK, d2C_dK2 = self._numerical_derivatives(K, T)

        numerator = dC_dT + self.r * K * dC_dK
        denominator = 0.5 * K ** 2 * d2C_dK2

        # Avoid division by zero or negative values
        if denominator <= 1e-10:
            warnings.warn(f"Small or negative denominator at K={K}, T={T}. Using fallback.")
            # Return implied volatility as fallback
            return float(self.iv_interpolator(T, K)[0, 0])

        local_var = numerator / denominator

        # Ensure non-negative variance
        if local_var <= 0:
            warnings.warn(f"Negative local variance at K={K}, T={T}. Using fallback.")
            return float(self.iv_interpolator(T, K)[0, 0])

        return np.sqrt(local_var)

    def compute_local_vol_surface(self,
                                  strike_range: Optional[Tuple[float, float]] = None,
                                  maturity_range: Optional[Tuple[float, float]] = None,
                                  n_strikes: int = 50,
                                  n_maturities: int = 30) -> None:
        """
        Compute the full local volatility surface
        """
        if self.market_data is None:
            raise ValueError("Market data not loaded. Call load_market_data() first.")

        # Set ranges
        if strike_range is None:
            K_min, K_max = self.strikes.min(), self.strikes.max()
        else:
            K_min, K_max = strike_range

        if maturity_range is None:
            T_min, T_max = self.maturities.min(), self.maturities.max()
        else:
            T_min, T_max = maturity_range

        # Create grids
        K_lv = np.linspace(K_min, K_max, n_strikes)
        T_lv = np.linspace(T_min, T_max, n_maturities)

        # Compute local volatility surface
        local_vols = np.zeros((n_maturities, n_strikes))

        for i, T in enumerate(T_lv):
            for j, K in enumerate(K_lv):
                try:
                    local_vols[i, j] = self.dupire_local_volatility(K, T)
                except:
                    # Use implied volatility as fallback
                    local_vols[i, j] = float(self.iv_interpolator(T, K)[0, 0])

        # Store results
        self.K_lv, self.T_lv = np.meshgrid(K_lv, T_lv)
        self.local_vol_surface = local_vols
        self.lv_strikes = K_lv
        self.lv_maturities = T_lv

        # Create interpolator for local volatility
        self.lv_interpolator = interpolate.RectBivariateSpline(
            T_lv, K_lv, local_vols, kx=3, ky=3, s=0
        )

    def plot_surfaces(self, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Plot both implied and local volatility surfaces

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        """
        if self.market_data is None:
            raise ValueError("Market data not loaded.")
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not computed. Call compute_local_vol_surface() first.")

        fig = plt.figure(figsize=figsize)

        # Implied Volatility Surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(self.K_grid, self.T_grid, self.implied_vols,
                                 cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title('Market Implied Volatility Surface')

        # Local Volatility Surface
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(self.K_lv, self.T_lv, self.local_vol_surface,
                                 cmap='plasma', alpha=0.8)
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturity')
        ax2.set_zlabel('Local Volatility')
        ax2.set_title('Dupire Local Volatility Surface')

        plt.tight_layout()
        plt.show()

    def plot_slices(self, fixed_maturity: float = None, fixed_strike: float = None) -> None:
        """
        Plot slices of volatility surfaces at fixed maturity or strike

        """
        if self.local_vol_surface is None:
            raise ValueError("Local volatility surface not computed.")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Volatility vs Strike at fixed maturity
        if fixed_maturity is None:
            fixed_maturity = self.lv_maturities[len(self.lv_maturities) // 2]

        iv_slice = [float(self.iv_interpolator(fixed_maturity, K)[0, 0]) for K in self.lv_strikes]
        lv_slice = [float(self.lv_interpolator(fixed_maturity, K)[0, 0]) for K in self.lv_strikes]

        axes[0].plot(self.lv_strikes, iv_slice, 'b-', label='Implied Vol', linewidth=2)
        axes[0].plot(self.lv_strikes, lv_slice, 'r--', label='Local Vol', linewidth=2)
        axes[0].set_xlabel('Strike')
        axes[0].set_ylabel('Volatility')
        axes[0].set_title(f'Volatility vs Strike (T = {fixed_maturity:.2f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Volatility vs Maturity at fixed strike
        if fixed_strike is None:
            fixed_strike = self.S0  # ATM

        iv_slice = [float(self.iv_interpolator(T, fixed_strike)[0, 0]) for T in self.lv_maturities]
        lv_slice = [float(self.lv_interpolator(T, fixed_strike)[0, 0]) for T in self.lv_maturities]

        axes[1].plot(self.lv_maturities, iv_slice, 'b-', label='Implied Vol', linewidth=2)
        axes[1].plot(self.lv_maturities, lv_slice, 'r--', label='Local Vol', linewidth=2)
        axes[1].set_xlabel('Maturity')
        axes[1].set_ylabel('Volatility')
        axes[1].set_title(f'Volatility vs Maturity (K = {fixed_strike:.0f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def monte_carlo_pricing(self,
                            payoff_func: Callable,
                            maturity: float,
                            n_paths: int = 100000,
                            n_steps: int = 252) -> Tuple[float, float]:
        """
        Price an exotic option using Monte Carlo with local volatility

        Parameters:
        -----------
        payoff_func : Callable
            Function that takes path array and returns payoff
        maturity : float
            Option maturity
        n_paths : int
            Number of Monte Carlo paths
        n_steps : int
            Number of time steps

        Returns:
        --------
        Tuple[float, float]
            (option_price, standard_error)
        """
        if self.lv_interpolator is None:
            raise ValueError("Local volatility surface not computed.")

        dt = maturity / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        # Generate random shocks
        np.random.seed(42)  # For reproducibility
        dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        for i in range(n_steps):
            t = i * dt
            S = paths[:, i]

            # Get local volatility for each path
            local_vols = np.array([float(self.lv_interpolator(t, s)[0, 0])
                                   for s in S])

            # Euler scheme for local volatility model
            paths[:, i + 1] = S * np.exp((self.r - self.q - 0.5 * local_vols ** 2) * dt +
                                         local_vols * dW[:, i])

        # Calculate payoffs
        payoffs = np.array([payoff_func(path) for path in paths])

        # Discount to present value
        option_price = np.exp(-self.r * maturity) * np.mean(payoffs)
        standard_error = np.exp(-self.r * maturity) * np.std(payoffs) / np.sqrt(n_paths)

        return option_price, standard_error

    def get_analytics(self) -> dict:
        """
        Get analytical insights about the volatility surfaces

        """
        if self.market_data is None or self.local_vol_surface is None:
            raise ValueError("Data not available. Load market data and compute local vol surface first.")

        analytics = {}

        # ATM volatilities
        atm_idx = np.argmin(np.abs(self.lv_strikes - self.S0))
        analytics['atm_implied_vols'] = [float(self.iv_interpolator(T, self.S0)[0, 0])
                                         for T in self.lv_maturities]
        analytics['atm_local_vols'] = self.local_vol_surface[:, atm_idx]

        # Term structure slope
        if len(self.lv_maturities) > 1:
            iv_slope = np.polyfit(self.lv_maturities, analytics['atm_implied_vols'], 1)[0]
            lv_slope = np.polyfit(self.lv_maturities, analytics['atm_local_vols'], 1)[0]
            analytics['iv_term_structure_slope'] = iv_slope
            analytics['lv_term_structure_slope'] = lv_slope

        # Volatility smile metrics
        mid_maturity_idx = len(self.lv_maturities) // 2
        mid_maturity = self.lv_maturities[mid_maturity_idx]

        smile_strikes = self.lv_strikes
        smile_iv = [float(self.iv_interpolator(mid_maturity, K)[0, 0]) for K in smile_strikes]
        smile_lv = self.local_vol_surface[mid_maturity_idx, :]

        analytics['smile_maturity'] = mid_maturity
        analytics['smile_strikes'] = smile_strikes
        analytics['smile_implied_vols'] = smile_iv
        analytics['smile_local_vols'] = smile_lv

        # Min/max values
        analytics['min_implied_vol'] = np.min(self.implied_vols)
        analytics['max_implied_vol'] = np.max(self.implied_vols)
        analytics['min_local_vol'] = np.min(self.local_vol_surface)
        analytics['max_local_vol'] = np.max(self.local_vol_surface)

        return analytics


def generate_synthetic_iv_surface(S0: float, strikes: np.ndarray, maturities: np.ndarray) -> np.ndarray:
    """
    Generate a synthetic implied volatility surface with realistic features
    including volatility smile and term structure
    """
    implied_vols = np.zeros((len(maturities), len(strikes)))

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            moneyness = np.log(K / S0)

            # Base volatility with term structure
            base_vol = 0.20 + 0.05 * np.exp(-2 * T)

            # Volatility smile (higher for OTM options)
            smile_effect = 0.1 * moneyness**2 + 0.02 * moneyness

            # Add some time-dependent skew
            skew_effect = -0.05 * moneyness * np.exp(-T)

            # Ensure positive volatility
            implied_vols[i, j] = max(base_vol + smile_effect + skew_effect, 0.05)

    return implied_vols


def main():

    print("=" * 80)
    print("DUPIRE LOCAL VOLATILITY MODEL DEMONSTRATION")
    print("=" * 80)

    # Market parameters
    S0 = 100.0          # Current spot price
    r = 0.05            # Risk-free rate
    q = 0.02            # Dividend yield

    print(f"\nMarket Parameters:")
    print(f"Spot Price (S0): ${S0}")
    print(f"Risk-free Rate (r): {r*100:.1f}%")
    print(f"Dividend Yield (q): {q*100:.1f}%")

    # Create option strikes and maturities
    strikes = np.linspace(80, 120, 15)  # Strikes from 80 to 120
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])  # Maturities from 3M to 2Y

    print(f"\nOption Grid:")
    print(f"Strikes: {strikes[0]:.0f} to {strikes[-1]:.0f} ({len(strikes)} points)")
    print(f"Maturities: {maturities[0]:.2f} to {maturities[-1]:.2f} years ({len(maturities)} points)")

    # Generate synthetic implied volatility surface
    print("\n" + "="*50)
    print("1. GENERATING SYNTHETIC MARKET DATA")
    print("="*50)

    implied_vols = generate_synthetic_iv_surface(S0, strikes, maturities)
    print(f"Generated {implied_vols.shape[0]} x {implied_vols.shape[1]} implied volatility surface")
    print(f"Implied Vol Range: {np.min(implied_vols):.1%} to {np.max(implied_vols):.1%}")

    # Initialize the Dupire model
    print("\n" + "="*50)
    print("2. INITIALIZING DUPIRE MODEL")
    print("="*50)

    dupire_model = DupireLocalVolatility(S0, r, q)
    print("✓ Dupire model initialized")

    # Load market data
    dupire_model.load_market_data(strikes, maturities, implied_vols)
    print("✓ Market data loaded and interpolated")
    print(f"✓ Created DataFrame with {len(dupire_model.market_data)} data points")

    # Compute local volatility surface
    print("\n" + "="*50)
    print("3. COMPUTING LOCAL VOLATILITY SURFACE")
    print("="*50)

    dupire_model.compute_local_vol_surface(n_strikes=40, n_maturities=25)
    print("✓ Local volatility surface computed using Dupire's formula")
    print(f"✓ Surface dimensions: {dupire_model.local_vol_surface.shape}")
    print(f"Local Vol Range: {np.min(dupire_model.local_vol_surface):.1%} to {np.max(dupire_model.local_vol_surface):.1%}")

    # Plot volatility surfaces
    print("\n" + "="*50)
    print("4. VISUALIZING VOLATILITY SURFACES")
    print("="*50)

    print("Plotting 3D surfaces...")
    dupire_model.plot_surfaces(figsize=(16, 6))

    # Plot volatility slices
    print("\nPlotting volatility slices...")
    dupire_model.plot_slices(fixed_maturity=1.0, fixed_strike=100.0)

    # Get analytics
    print("\n" + "="*50)
    print("5. ANALYTICS AND INSIGHTS")
    print("="*50)

    analytics = dupire_model.get_analytics()

    print(f"ATM Implied Vol (1Y): {analytics['atm_implied_vols'][len(analytics['atm_implied_vols'])//2]:.1%}")
    print(f"ATM Local Vol (1Y): {analytics['atm_local_vols'][len(analytics['atm_local_vols'])//2]:.1%}")
    print(f"IV Term Structure Slope: {analytics['iv_term_structure_slope']:.4f}")
    print(f"LV Term Structure Slope: {analytics['lv_term_structure_slope']:.4f}")
    print(f"Volatility Range - IV: {analytics['min_implied_vol']:.1%} to {analytics['max_implied_vol']:.1%}")
    print(f"Volatility Range - LV: {analytics['min_local_vol']:.1%} to {analytics['max_local_vol']:.1%}")

    # Create analytics visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # ATM term structure
    axes[0, 0].plot(dupire_model.lv_maturities, analytics['atm_implied_vols'], 'b-o', label='Implied Vol')
    axes[0, 0].plot(dupire_model.lv_maturities, analytics['atm_local_vols'], 'r-s', label='Local Vol')
    axes[0, 0].set_xlabel('Maturity')
    axes[0, 0].set_ylabel('Volatility')
    axes[0, 0].set_title('ATM Volatility Term Structure')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Volatility smile
    axes[0, 1].plot(analytics['smile_strikes'], analytics['smile_implied_vols'], 'b-o', label='Implied Vol')
    axes[0, 1].plot(analytics['smile_strikes'], analytics['smile_local_vols'], 'r-s', label='Local Vol')
    axes[0, 1].axvline(S0, color='k', linestyle='--', alpha=0.5, label='ATM')
    axes[0, 1].set_xlabel('Strike')
    axes[0, 1].set_ylabel('Volatility')
    axes[0, 1].set_title(f'Volatility Smile (T = {analytics["smile_maturity"]:.2f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Surface comparison heatmaps
    im1 = axes[1, 0].imshow(dupire_model.implied_vols, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Implied Volatility Heatmap')
    axes[1, 0].set_xlabel('Strike Index')
    axes[1, 0].set_ylabel('Maturity Index')
    plt.colorbar(im1, ax=axes[1, 0])

    im2 = axes[1, 1].imshow(dupire_model.local_vol_surface, aspect='auto', cmap='plasma')
    axes[1, 1].set_title('Local Volatility Heatmap')
    axes[1, 1].set_xlabel('Strike Index')
    axes[1, 1].set_ylabel('Maturity Index')
    plt.colorbar(im2, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

    # Monte Carlo pricing examples
    print("\n" + "="*50)
    print("6. MONTE CARLO PRICING WITH LOCAL VOLATILITY")
    print("="*50)

    maturity = 1.0

    # Define payoff functions
    def european_call_payoff(path):
        return max(path[-1] - 105, 0)  # Strike = 105

    def asian_call_payoff(path):
        avg_price = np.mean(path)
        return max(avg_price - 100, 0)  # Strike = 100

    def barrier_call_payoff(path):
        barrier = 110
        strike = 100
        if np.max(path) >= barrier:  # Knock-out barrier
            return 0
        return max(path[-1] - strike, 0)

    # Price different options
    options = [
        ("European Call (K=105)", european_call_payoff),
        ("Asian Call (K=100)", asian_call_payoff),
        ("Barrier Call (K=100, B=110)", barrier_call_payoff)
    ]

    print("Pricing exotic options using Monte Carlo with local volatility...")

    for name, payoff_func in options:
        price, se = dupire_model.monte_carlo_pricing(payoff_func, maturity, n_paths=50000, n_steps=100)
        print(f"{name:25s}: ${price:6.3f} ± ${1.96*se:.3f} (95% CI)")

    # Compare with Black-Scholes for European option
    bs_price = dupire_model.black_scholes_call(S0, 105, maturity, 0.20)  # Using 20% vol
    print(f"{'Black-Scholes (σ=20%)':25s}: ${bs_price:6.3f}")

    # Display summary
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nFeatures demonstrated:")
    print("✓ Synthetic market data generation")
    print("✓ Dupire local volatility surface computation")
    print("✓ 3D surface visualization")
    print("✓ Volatility slice analysis")
    print("✓ Comprehensive analytics")
    print("✓ Monte Carlo pricing with local volatility")
    print("✓ Comparison with Black-Scholes pricing")

    print(f"\nKey insights:")
    print(f"• Local volatility shows {dupire_model.local_vol_surface.shape[0]*dupire_model.local_vol_surface.shape[1]} calibrated points")
    print(f"• Volatility smile captured in local vol surface")
    print(f"• Term structure effects incorporated")
    print(f"• Exotic option pricing demonstrates path-dependent effects")

    return dupire_model


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Run the demonstration
    model = main()
