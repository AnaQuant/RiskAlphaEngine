"""
Portfolio Stress Testing Framework
==================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class PortfolioStressTester:
    """
    Stress testing framework for portfolio risk management.
    Features:
    - Position management and tracking
    - Historical scenario replay
    - Hypothetical stress scenarios
    - Risk metrics and analytics
    - Performance attribution
    - Visualization support

    """

    def __init__(self, name: str = "Portfolio"):

        self.name = name
        self.positions = {}
        self.stress_scenarios = {}
        self.historical_scenarios = {}
        self.results = {}
        self.creation_date = datetime.now()

    def add_position(self, asset_name: str, quantity: float, current_price: float,
                     asset_class: str = "Equity", currency: str = "USD"):
        """
        Add a position to the portfolio.

        Args:
            asset_name: Ticker or identifier (e.g., 'AAPL', 'SPY')
            quantity: Number of shares/units (can be negative for shorts)
            current_price: Current market price per unit
            asset_class: Asset classification (Equity, Bond, Commodity, etc.)
            currency: Base currency (default USD)
        """
        self.positions[asset_name] = {
            'quantity': quantity,
            'current_price': current_price,
            'market_value': quantity * current_price,
            'asset_class': asset_class,
            'currency': currency,
            'weight': 0  # Will be calculated when needed
        }

    def remove_position(self, asset_name: str):
        """Remove a position from the portfolio."""
        if asset_name in self.positions:
            del self.positions[asset_name]

    def update_price(self, asset_name: str, new_price: float):
        """Update the price of an existing position."""
        if asset_name in self.positions:
            self.positions[asset_name]['current_price'] = new_price
            self.positions[asset_name]['market_value'] = (
                    self.positions[asset_name]['quantity'] * new_price
            )

    def get_portfolio_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of the current portfolio.

        Returns:
            DataFrame with position details and weights
        """
        if not self.positions:
            return pd.DataFrame()

        total_value = sum([pos['market_value'] for pos in self.positions.values()])

        summary_data = []
        for asset, pos in self.positions.items():
            weight = (pos['market_value'] / total_value) * 100 if total_value != 0 else 0
            summary_data.append({
                'Asset': asset,
                'Quantity': pos['quantity'],
                'Price': pos['current_price'],
                'Market Value': pos['market_value'],
                'Weight %': weight,
                'Asset Class': pos['asset_class'],
                'Currency': pos['currency']
            })

        return pd.DataFrame(summary_data)

    def add_scenario(self, scenario_name: str, price_shocks: Dict[str, float],
                     description: str = "", scenario_type: str = "hypothetical"):
        """
        Add a stress scenario.

        Args:
            scenario_name: Name of the scenario
            price_shocks: Dict of asset -> percentage change (e.g., {'AAPL': -0.30})
            description: Optional description
            scenario_type: 'hypothetical' or 'historical'
        """
        self.stress_scenarios[scenario_name] = {
            'shocks': price_shocks,
            'description': description,
            'type': scenario_type
        }

    def add_historical_scenario(self, scenario_name: str, time_period: str,
                                historical_shocks: Dict[str, float], description: str = ""):
        """
        Add a historical stress scenario based on real market events.

        Args:
            scenario_name: Name of the historical event
            time_period: When it happened (e.g., 'Feb-Mar 2020')
            historical_shocks: Dict of asset -> actual percentage change observed
            description: Description of what happened
        """
        self.historical_scenarios[scenario_name] = {
            'shocks': historical_shocks,
            'period': time_period,
            'description': description,
            'type': 'historical'
        }

        # Also add to regular scenarios for unified processing
        self.stress_scenarios[scenario_name] = {
            'shocks': historical_shocks,
            'description': description,
            'type': 'historical'
        }

    def run_stress_test(self, scenario_name: str) -> Dict:
        """
        Run a single stress test scenario.

        Args:
            scenario_name: Name of scenario to run

        Returns:
            Dictionary with detailed P&L results
        """
        if scenario_name not in self.stress_scenarios:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        scenario_data = self.stress_scenarios[scenario_name]
        scenario = scenario_data['shocks']

        results = {
            'scenario': scenario_name,
            'scenario_type': scenario_data['type'],
            'description': scenario_data['description'],
            'position_pnl': {},
            'total_pnl': 0,
            'total_portfolio_value_before': 0,
            'total_portfolio_value_after': 0,
            'number_of_positions': len(self.positions)
        }

        # Calculate portfolio weights
        total_value = sum([pos['market_value'] for pos in self.positions.values()])

        for asset, position in self.positions.items():
            current_mv = position['market_value']
            results['total_portfolio_value_before'] += current_mv
            weight = (current_mv / total_value) * 100 if total_value != 0 else 0

            if asset in scenario:
                shock = scenario[asset]
                new_price = position['current_price'] * (1 + shock)
                new_mv = position['quantity'] * new_price
                pnl = new_mv - current_mv

                results['position_pnl'][asset] = {
                    'current_price': position['current_price'],
                    'stressed_price': new_price,
                    'shock_pct': shock * 100,
                    'pnl': pnl,
                    'weight': weight,
                    'asset_class': position['asset_class'],
                    'contribution_to_total_pnl': 0  # Will calculate after
                }
                results['total_portfolio_value_after'] += new_mv
            else:
                # No shock applied
                results['position_pnl'][asset] = {
                    'current_price': position['current_price'],
                    'stressed_price': position['current_price'],
                    'shock_pct': 0,
                    'pnl': 0,
                    'weight': weight,
                    'asset_class': position['asset_class'],
                    'contribution_to_total_pnl': 0
                }
                results['total_portfolio_value_after'] += current_mv

        results['total_pnl'] = results['total_portfolio_value_after'] - results['total_portfolio_value_before']
        results['total_return_pct'] = (results['total_pnl'] / results['total_portfolio_value_before']) * 100

        # Calculate contribution to total P&L
        for asset in results['position_pnl']:
            if results['total_pnl'] != 0:
                contribution = (results['position_pnl'][asset]['pnl'] / results['total_pnl']) * 100
                results['position_pnl'][asset]['contribution_to_total_pnl'] = contribution

        self.results[scenario_name] = results
        return results

    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all stress scenarios and return summary results.

        Returns:
            DataFrame with summary of all scenario results
        """
        summary_data = []

        for scenario_name in self.stress_scenarios.keys():
            result = self.run_stress_test(scenario_name)
            summary_data.append({
                'Scenario': scenario_name,
                'Type': result['scenario_type'],
                'Portfolio Value Before': result['total_portfolio_value_before'],
                'Portfolio Value After': result['total_portfolio_value_after'],
                'Total P&L': result['total_pnl'],
                'P&L %': result['total_return_pct'],
                'Positions Count': result['number_of_positions']
            })

        return pd.DataFrame(summary_data)

    def get_position_analysis(self, scenario_name: str) -> pd.DataFrame:
        """
        Get detailed position-level analysis for a scenario.

        Args:
            scenario_name: Name of scenario to analyze

        Returns:
            DataFrame with position-level results
        """
        if scenario_name not in self.results:
            self.run_stress_test(scenario_name)

        result = self.results[scenario_name]
        position_data = []

        for asset, data in result['position_pnl'].items():
            position_data.append({
                'Asset': asset,
                'Asset Class': data['asset_class'],
                'Current Price': data['current_price'],
                'Stressed Price': data['stressed_price'],
                'Shock %': data['shock_pct'],
                'Position P&L': data['pnl'],
                'Portfolio Weight %': data['weight'],
                'Contribution to Total P&L %': data['contribution_to_total_pnl']
            })

        return pd.DataFrame(position_data)

    def compare_to_benchmark(self, scenario_name: str, benchmark_asset: str = 'SPY') -> Dict:
        """
        Compare portfolio performance to a benchmark during a scenario.

        Args:
            scenario_name: Name of scenario
            benchmark_asset: Asset to use as benchmark (default SPY)

        Returns:
            Dictionary with comparison metrics
        """
        if scenario_name not in self.results:
            self.run_stress_test(scenario_name)

        result = self.results[scenario_name]
        portfolio_return = result['total_return_pct']

        # Get benchmark return from scenario data
        scenario_data = self.stress_scenarios[scenario_name]['shocks']
        benchmark_return = scenario_data.get(benchmark_asset, 0) * 100

        return {
            'scenario': scenario_name,
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'outperformance': portfolio_return - benchmark_return,
            'benchmark_asset': benchmark_asset
        }

    def create_covid_scenarios(self):
        """Create pre-built COVID-19 historical scenarios with actual market data."""

        # COVID Initial Crash (Feb 19 - Mar 23, 2020)
        self.add_historical_scenario(
            'COVID Initial Crash',
            'Feb 19 - Mar 23, 2020',
            {
                'SPY': -0.34, 'QQQ': -0.30, 'AAPL': -0.31, 'MSFT': -0.29, 'TSLA': -0.60,
                'XLF': -0.45, 'XLE': -0.54, 'GLD': +0.04, 'TLT': +0.21, 'VXX': +4.00,
                'USD_INDEX': +0.08, 'REITS': -0.44, 'OIL': -0.65, 'BTC': -0.50
            },
            "Fastest market crash in history. Global lockdowns, uncertainty peaked."
        )

        # COVID Recovery Rally (Mar 23 - Aug 2020)
        self.add_historical_scenario(
            'COVID Recovery Rally',
            'Mar 23 - Aug 2020',
            {
                'SPY': +0.50, 'QQQ': +0.60, 'AAPL': +0.70, 'MSFT': +0.45, 'TSLA': +2.48,
                'XLF': +0.25, 'XLE': +0.15, 'GLD': +0.28, 'TLT': +0.05, 'VXX': -0.65,
                'USD_INDEX': -0.10, 'REITS': +0.20, 'OIL': +1.20, 'BTC': +1.60
            },
            "Massive stimulus drove V-shaped recovery. Fed money printing, tech soared."
        )

        # COVID Vaccine Rotation (Nov 2020)
        self.add_historical_scenario(
            'COVID Vaccine Rotation',
            'Nov 2020',
            {
                'SPY': +0.11, 'QQQ': +0.02, 'AAPL': -0.03, 'MSFT': +0.08, 'TSLA': +0.03,
                'XLF': +0.35, 'XLE': +0.29, 'GLD': -0.05, 'TLT': -0.13, 'VXX': -0.25,
                'USD_INDEX': -0.02, 'REITS': +0.18, 'OIL': +0.27, 'BTC': +0.43
            },
            "Vaccine news triggered rotation from growth to value stocks."
        )

    def create_financial_crisis_scenarios(self):
        """Create 2008 Financial Crisis scenarios."""

        self.add_historical_scenario(
            '2008 Financial Crisis',
            'Sep 2008 - Mar 2009',
            {
                'SPY': -0.57, 'QQQ': -0.54, 'XLF': -0.83, 'XLE': -0.54,
                'GLD': +0.24, 'TLT': +0.32, 'USD_INDEX': +0.15, 'REITS': -0.68
            },
            "Lehman Brothers collapse, global financial system near-collapse."
        )

    def get_risk_metrics(self, scenarios: List[str] = None) -> Dict:
        """
        Calculate risk metrics across scenarios.

        Args:
            scenarios: List of scenarios to include (default: all)

        Returns:
            Dictionary with risk metrics
        """
        if scenarios is None:
            scenarios = list(self.stress_scenarios.keys())

        returns = []
        for scenario in scenarios:
            if scenario not in self.results:
                self.run_stress_test(scenario)
            returns.append(self.results[scenario]['total_return_pct'])

        returns = np.array(returns)

        return {
            'scenarios_count': len(scenarios),
            'mean_return': np.mean(returns),
            'volatility': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'var_95': np.percentile(returns, 5),  # 95% VaR
            'var_99': np.percentile(returns, 1),  # 99% VaR
            'scenarios_negative': np.sum(returns < 0),
            'scenarios_positive': np.sum(returns > 0)
        }

    def export_results(self, filename: str = None) -> str:
        """
        Export all results to CSV files.

        Args:
            filename: Base filename (default: uses portfolio name and timestamp)

        Returns:
            String with export summary
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name.replace(' ', '_')}_{timestamp}"

        # Export summary
        summary_df = self.run_all_scenarios()
        summary_df.to_csv(f"{filename}_summary.csv", index=False)

        # Export detailed results for each scenario
        for scenario_name in self.results.keys():
            position_df = self.get_position_analysis(scenario_name)
            safe_name = scenario_name.replace(' ', '_').replace('/', '_')
            position_df.to_csv(f"{filename}_{safe_name}_positions.csv", index=False)

        return f"Results exported with base filename: {filename}"


# Utility functions for common scenarios
def create_market_crash_scenario() -> Dict[str, float]:
    """Create a generic market crash scenario."""
    return {
        'SPY': -0.25, 'QQQ': -0.30, 'IWM': -0.35,  # Equities down
        'TLT': +0.15, 'GLD': +0.10,  # Safe havens up
        'VXX': +2.00  # Volatility spikes
    }


def create_inflation_shock_scenario() -> Dict[str, float]:
    """Create an inflation shock scenario."""
    return {
        'SPY': -0.15, 'QQQ': -0.20,  # Growth stocks hurt more
        'TLT': -0.25, 'TIP': +0.05,  # Bonds down, TIPS up
        'GLD': +0.15, 'OIL': +0.30,  # Commodities up
        'XLF': +0.05  # Banks benefit from rates
    }


def create_tech_bubble_burst() -> Dict[str, float]:
    """Create a tech bubble burst scenario."""
    return {
        'QQQ': -0.40, 'AAPL': -0.35, 'MSFT': -0.30, 'TSLA': -0.50,
        'SPY': -0.20, 'IWM': -0.15,  # Broad market less affected
        'TLT': +0.10, 'GLD': +0.05  # Flight to safety
    }