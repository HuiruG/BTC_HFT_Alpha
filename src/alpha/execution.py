
import polars as pl
import numpy as np
from alpha.base import BaseAlpha

# ==============================================================================
# GROUP 1: COST ESTIMATORS (Spread & Slippage)
# ==============================================================================

class RollSpread(BaseAlpha):
    """
    [1] Roll (1984) Effective Spread
    [Logic] 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1}))
    [Utility] Measures Spread in ranging/choppy markets (Bid-Ask Bounce).
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)

        dp = pl.col("close").forward_fill().diff().fill_null(0)

        # Rolling Covariance
        x = dp
        y = dp.shift(1).fill_null(0)

        mean_x = x.rolling_mean(window)
        mean_y = y.rolling_mean(window)
        mean_xy = (x * y).rolling_mean(window)

        cov = mean_xy - (mean_x * mean_y)

        return (2 * (-cov).clip(0, 1e9).sqrt()).alias("alpha_roll_spread")

class CorwinSchultzSpread(BaseAlpha):
    """
    [2] Corwin-Schultz (2012) High-Low Spread
    [Logic] Derived from High/Low geometric properties.
    [Utility] Measures Spread in trending markets (complements Roll Spread).
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 20)

        high = pl.col("high").forward_fill()
        low = pl.col("low").forward_fill()

        # Simplified Robust Proxy: Excess High-Low Range
        # Real spread is the stable part of H-L range minus volatility component
        hl_ratio = (high / (low + 1e-9)).log()

        avg_hl = hl_ratio.rolling_mean(window)
        std_hl = hl_ratio.rolling_std(window)

        return (avg_hl - std_hl).clip(0, 1.0).alias("alpha_cs_spread")

# ==============================================================================
# GROUP 2: ADVERSE SELECTION (Toxicity & Urgency)
# ==============================================================================

class FlowToxicity(BaseAlpha):
    """
    [3] Flow Toxicity (VPIN Proxy)
    [Logic] Smoothed |Buy-Sell| / Total Vol
    [Utility] Identifies Toxic Flow (Informed Traders).
              High Value -> Don't provide liquidity (Limit Orders will be run over).
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 20)

        diff_vol = (pl.col("buy_vol") - pl.col("sell_vol")).abs().fill_null(0)
        total_vol = (pl.col("buy_vol") + pl.col("sell_vol")).fill_null(0)

        vpin = diff_vol.rolling_mean(window) / (total_vol.rolling_mean(window) + 1e-9)
        return vpin.alias("alpha_flow_toxicity")

class TrendDeviation(BaseAlpha):
    """
    [4] Trend Deviation (Urgency)
    [Logic] (Price - EMA) / Volatility
    [Utility] Measures price over-extension.
              High deviation -> Reversion risk is high -> Passive execution preferred.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        span = self.params.get("span", 20)

        close = pl.col("close").forward_fill()
        ema = close.ewm_mean(span=span, adjust=False)
        vol = close.diff().abs().rolling_mean(span).fill_null(1e-9)

        return ((close - ema) / vol).alias("alpha_trend_deviation")

# ==============================================================================
# GROUP 3: MARKET STATE & TIMING (Regime & Entropy)
# ==============================================================================

class OrderFlowEntropy(BaseAlpha):
    """
    [5] Order Flow Entropy (Crowding)
    [Logic] Shannon Entropy of Buy/Sell Ratios
    [Utility] High Entropy -> Healthy/Balanced Flow. Low Entropy -> One-sided Panic.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)

        b = pl.col("buy_vol").fill_null(0)
        s = pl.col("sell_vol").fill_null(0)
        total = b + s + 1e-9

        p_b = (b / total).clip(1e-6, 1.0)
        p_s = (s / total).clip(1e-6, 1.0)

        entropy = -1 * (p_b * p_b.log(2) + p_s * p_s.log(2))
        return entropy.rolling_mean(window).alias("alpha_flow_entropy")

class VolatilitySignature(BaseAlpha):
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        fast_window = self.params.get("fast_window", 10)
        slow_window = self.params.get("slow_window", 100)
        
        ret = pl.col("close").forward_fill().log().diff().fill_null(0)
        
        vol_fast = ret.rolling_std(fast_window)
        vol_slow = ret.rolling_std(slow_window)
        
        scaling = np.sqrt(fast_window / slow_window)
        return (vol_fast / (vol_slow * scaling + 1e-9)).alias("alpha_vol_signature")

class VolatilityRegime(BaseAlpha):
    """
    [7] Volatility Regime (Expansion/Contraction)
    [Logic] Current Vol / Historical Avg Vol
    [Utility] Risk Gate.
              High Value -> Volatility Expanding -> Widen Spreads / Reduce Size.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50) # Current Vol Window
        history = self.params.get("history", 300) # Baseline Window

        ret = pl.col("close").forward_fill().log().diff().fill_null(0)

        current_vol = ret.rolling_std(window)
        baseline_vol = ret.rolling_std(history)

        # Z-Score of Volatility or Simple Ratio
        return (current_vol / (baseline_vol + 1e-9)).alias("alpha_vol_regime")

class QueuePositionEstimate(BaseAlpha):
    """
    [Signal] Theoretical Queue Advancement
    [Logic] Cumulative Volume since 'Lookback' normalized by Median Depth.
    [Formula] sum(Volume_since_T) / Avg_Volume_at_Level
    [Predicts] Close to 1.0 means your theoretical order is about to be filled.
               Useful for determining 'Urgency' in Execution.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 20)
        
        total_vol = (pl.col("buy_vol") + pl.col("sell_vol")).fill_null(0)
        
        accumulated_vol = total_vol.rolling_sum(window)
        
        avg_depth = total_vol.rolling_median(window * 5).fill_null(1e-9)
        
        return (accumulated_vol / avg_depth).alias("alpha_queue_pos")        
