# Trend Following Strategy Notebook — Reference Plan

Persistent reference for the `notebooks/_fx/trend strategy.ipynb` notebook.
Mirrors the structure of `risk_reversal_improvement.md`.

---

## Critical Files

| File | Role |
|---|---|
| `notebooks/_fx/trend strategy.ipynb` | Main notebook (42 cells, 4 stages) |
| `FX_strategies/sma_backtest.py` | `SMAVectorBacktester` — price-level SMA crossover |
| `FX_strategies/momentum_backtest.py` | `MomVectorBacktester` — time-series momentum (TSMOM) |
| `FX_strategies/ewma_crossover_backtest.py` | `EWMACrossoverBacktester` — EWMA vol-normalised crossover (wraps `Strategy`) |
| `FX_strategies/crossover_momentum.py` | `Strategy` / `BaseMomentum` — pure signal generator (unchanged) |
| `data/spots_currencies_universe.csv` | 28 FX spot pairs, daily, from 12 Dec 2005 |

---

## Style Contract

- No emojis; professional, academic register
- Colour palette: `#1F77B4`, `#D62728`, `#2CA02C`, `#FF7F0E`, `#9467BD`
- Summary stats per strategy: annualised return, annualised volatility, Sharpe ratio, max drawdown
- LaTeX in every derivation cell; one-day execution lag on all strategies
- `get_summary_stats(results, label, trading_days=252)` — notebook-level helper, defined in Cell 15

---

## Bug Fixes Applied

| Bug | Fix |
|---|---|
| Project root resolved to `notebooks/` | `os.path.abspath('../..')` (notebook is at `notebooks/_fx/`) |
| Wrong module import prefix (`FX.`) | `from FX_strategies.sma_backtest import ...` |
| Hardcoded relative data path in `.py` files | Added `data_path` kwarg (backward-compatible default) to both backtester `__init__` |
| No EURUSD-compatible interface for `Strategy` | Created `EWMACrossoverBacktester` with `annual=252`, `vol_window=60` daily defaults |

---

## Stage 1 — Foundation (Cells 1–10)

**Status:** Complete

| # | Type | Content |
|---|---|---|
| 1 | MD | Title, Abstract, Table of Contents |
| 2 | Code | Imports: `pandas`, `numpy`, `matplotlib`, `warnings` |
| 3 | Code | Display config: retina, `plt.rcParams`, `PALETTE` |
| 4 | Code | Path setup + `DATA_PATH` + all four strategy imports — two assertions |
| 5 | MD | §1 Background: trend following theory, Moskowitz et al. 2012, FX persistence, 3-strategy taxonomy table |
| 6 | Code | Load 28-pair universe; print shape, date range; `.head()` |
| 7 | Code | `raw.describe().round(4)` |
| 8 | Code | EURUSD 2-panel: spot price (top) + log return (bottom) |
| 9 | MD | §1.1 Serial autocorrelation — ACF formula in LaTeX |
| 10 | Code | EURUSD ACF bar chart (lags 1–60, `statsmodels.tsa.stattools.acf`, 95% CI) |

**Checkpoint:** Both asserts pass; data `(~3723, 28)`; ACF chart renders.

---

## Stage 2 — SMA Crossover Deep Dive (Cells 11–20)

**Status:** Complete

| # | Type | Content |
|---|---|---|
| 11 | MD | §2 SMA Crossover — full LaTeX: $\text{SMA}_t^{(n)}$, $\delta_t$, $p_t$, $\pi_t$, $\Pi_T$ |
| 12 | Code | `SMAVectorBacktester(EURUSD, 42, 252, data_path=DATA_PATH)` — run + `.results.tail()` |
| 13 | Code | 2-panel: spot + SMA overlays (top); position step ±1 (bottom) |
| 14 | Code | Styled cumulative chart: blue=BnH, red=SMA |
| 15 | Code | `get_summary_stats()` helper defined here; called on baseline results |
| 16 | MD | §2.1 Parameter Sensitivity — optimisation objective in LaTeX; overfitting warning |
| 17 | Code | `optimize_parameters((10,80,5), (100,300,10))` — 280 evaluations |
| 18 | Code | 8×11 grid → `imshow` heatmap (RdYlGn; NaN where SMA1≥SMA2) |
| 19 | Code | Optimal params run + 3-line chart (baseline / optimal / BnH) + summary stats |
| 20 | MD | §2.2 Interpretation — pre/post 2015 regime; 4 numbered limitations |

**Checkpoint:** `run_strategy()` returns `(1.4, 0.58)` matching original notebook; heatmap renders.

---

## Stage 3 — Return-Based Momentum: TSMOM and EWMA (Cells 21–32)

**Status:** Complete

### Part A — TSMOM (Cells 21–26)

| # | Type | Content |
|---|---|---|
| 21 | MD | §3 header + §3.1 TSMOM — $\bar{r}_t^{(m)}$, $p_t$, $\pi_t$ with TC indicator in LaTeX |
| 22 | Code | Baseline loop `m ∈ {1,2,5,10,20,60}` at TC=0; formatted table |
| 23 | Code | 5-line chart `m ∈ {1,5,20,60,120}` (one per PALETTE colour) + gray BnH |
| 24 | MD | §3.1.1 TC sensitivity — annual drag formula; realistic FX pip cost reference |
| 25 | Code | TC grid: `tc ∈ 7 values` × `m ∈ {5,20}` → two declining curves |
| 26 | Code | TSMOM(m=20) + SMA(42/252) comparison table via `get_summary_stats` |

### Part B — EWMA Vol-Normalised Crossover (Cells 27–32)

| # | Type | Content |
|---|---|---|
| 27 | MD | §3.1.2 TSMOM interpretation + §3.2 EWMA 4-step derivation in LaTeX: $\sigma_t$, $\tilde{S}_t$, $\tilde{f}_t^{(s,\ell)}$, clipping |
| 28 | Code | `Strategy(annual=252, vol_window=60, longonly=False)` inline; signal stats |
| 29 | Code | Apply signal with 1-day lag; cumulative chart vs BnH |
| 30 | Code | Three window configs (short/medium/long); identify best → `ewma_best_results` |
| 31 | Code | 3-strategy head-to-head: SMA / TSMOM / EWMA best + BnH |
| 32 | Code | 4-row summary table: BnH, SMA, TSMOM, EWMA |

**Checkpoint:** `compute_signal` runs with `annual=252`; `run_strategy(momentum=2)` returns `(18224.48, 9274.48)`.

---

## Stage 4 — Multi-Currency Portfolio Analysis (Cells 33–42)

**Status:** Complete

**Selected pairs (8):** `EURUSD, GBPUSD, AUDUSD, USDCAD, USDJPY, USDCHF, USDNOK, USDSEK`

**Shared parameters:**

| Parameter | Value |
|---|---|
| SMA windows | `SMA1=42`, `SMA2=252` |
| TSMOM look-back | `m=20` |
| TSMOM TC | `1 bp` one-way |
| EWMA windows | `[(16,64), (32,128), (64,256)]` |
| EWMA vol_window | `60` trading days |
| EWMA annual | `252` |

| # | Type | Content |
|---|---|---|
| 33 | MD | §4 header — pair universe table, sign convention, equal-weight formula |
| 34 | Code | Loop 8 pairs × 3 strategies → `sma_strategies`, `mom_strategies`, `ewma_strategies` dicts |
| 35 | Code | Three 2×4 subplot grids (one per strategy) |
| 36 | Code | 3×8 annotated Sharpe heatmap (RdYlGn, symmetric colormap) |
| 37 | Code | Equal-weight portfolio construction — inner-join alignment; print date ranges |
| 38 | Code | 4-line portfolio chart: SMA / TSMOM / EWMA portfolios + EURUSD BnH |
| 39 | Code | 3-panel diversification bar chart: pair Sharpe (blue) + portfolio (green) |
| 40 | Code | **27-row** final summary table — MultiIndex `(Pair, Strategy)` |
| 41 | MD | §4.1 Cross-pair analysis: G10 majors, commodity pairs, European crosses, diversification benefit |
| 42 | MD | §4.2 Limitations: TC asymmetry, walk-forward, equal-weight, vol targeting, carry correlation, base class refactor |

**Checkpoint:** 27-row summary table renders; portfolio Sharpe > average individual pair Sharpe.

---

## Architectural Note — VectorBacktesterBase (Proposed)

The three backtester classes share identical `get_data()` logic and structurally similar `run_strategy()` and `plot_results()` methods. A future refactor should create:

**`FX_strategies/base_backtester.py`**

```python
class VectorBacktesterBase(metaclass=ABCMeta):
    def __init__(self, symbol, start, end, data_path): ...
    def get_data(self): ...            # shared: CSV read, date filter, log return
    def get_summary_stats(self): ...   # shared: ann. return, vol, Sharpe, max drawdown
    def plot_results(self, ...): ...   # shared: styled matplotlib, project palette
    @abstractmethod
    def run_strategy(self): ...
    @abstractmethod
    def set_parameters(self, **kw): ...
    @abstractmethod
    def _plot_title(self): ...
```

Concrete classes then inherit from this base and implement only `run_strategy()`, `set_parameters()`, and `_plot_title()`. The notebook-level `get_summary_stats()` function (Cell 15) would be replaced by `backtester_obj.get_summary_stats()`.

---

## Key Parameters Quick Reference

| Strategy | Class | Key params | Data source |
|---|---|---|---|
| SMA Crossover | `SMAVectorBacktester` | `SMA1`, `SMA2` | `data_path` kwarg |
| TSMOM | `MomVectorBacktester` | `momentum`, `tc`, `amount` | `data_path` kwarg |
| EWMA Vol-Norm. | `EWMACrossoverBacktester` | `windows_lst`, `vol_window=60`, `annual=252` | `data_path` kwarg |
| Signal only | `Strategy` | `windows_lst`, `vol_window`, `vol_scale`, `longonly`, `annual` | Accepts `pd.Series` directly |
