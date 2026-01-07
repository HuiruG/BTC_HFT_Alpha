
import polars as pl
import numpy as np
from alpha.base import BaseAlpha

# ==============================================================================
# 1. Self-Exciting Process (The "True" Hawkes Proxy)
# ==============================================================================
class SelfExcitingBurst(BaseAlpha):
    """
    [Signal] Trade Arrival Burstiness (Hawkes Proxy)
    [Logic] Intensity_Fast / Intensity_Slow (using Exponential Decay)
    [Upgrade] Using EWM (Exponential Kernel) instead of SMA to mimic
              Hawkes decay kernel: lambda(t) ~ exp(-beta * t).
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        # Fast span mimics high beta (quick decay), Slow span mimics baseline
        fast_span = self.params.get("fast_span", 10)
        slow_span = self.params.get("slow_span", 100)

        count = pl.col("trade_count").fill_null(0).cast(pl.Float64)

        # Exponential decay kernels
        # EWM is mathematically closer to the memory kernel of a Hawkes process
        intensity_fast = count.ewm_mean(span=fast_span, adjust=False)
        intensity_slow = count.ewm_mean(span=slow_span, adjust=False)

        # Burst Ratio: > 1.0 means we are in a self-exciting cascade
        std_slow = count.rolling_std(window_size=slow_span).fill_null(1e-6)
        return ((intensity_fast - intensity_slow) / std_slow).alias("alpha_self_exciting")

# ==============================================================================
# 2. Psychological Barriers (The "Real" Clustering)
# ==============================================================================
class PsychologicalBarrier(BaseAlpha):
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        level = self.params.get("level", 100.0)
        window = self.params.get("window", 20)
        c = pl.col("close").cast(pl.Float64)
        _dist = pl.min_horizontal(c % level, level - (c % level))
        return (-0.1 * _dist).exp().rolling_mean(window).alias("alpha_psych_barrier")

        
# ==============================================================================
# 3. De-Biased Flow Correlation (The "Robust" Persistence)
# ==============================================================================
class DeBiasedFlowCorr(BaseAlpha):
    """
    [Signal] Order Flow Autocorrelation Decay (De-meaned)
    [Logic] Corr(Sign_t, Sign_{t-k}) ratio.
    [Upgrade] Uses Pearson Correlation (rolling_corr) to handle Mean-Shifts
              during strong trends. Solves the 'Trend Bias' issue.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        short_lag = self.params.get("short_lag", 1)
        long_lag = self.params.get("long_lag", 5)
        window = self.params.get("window", 50)

        # 1. Trade Sign (+1 / -1)
        # Using volume weighted sign is even better, but let's stick to sign persistence
        # as a measure of 'Information Assymetry'
        sign = (pl.col("buy_vol") - pl.col("sell_vol")).sign().fill_null(0)

        # 2. Rolling Pearson Correlation
        # This automatically subtracts the mean (De-trending) inside the window
        corr_short = pl.rolling_corr(sign, sign.shift(short_lag), window_size=window)
        corr_long = pl.rolling_corr(sign, sign.shift(long_lag), window_size=window)

        # 3. Decay Ratio
        # High Ratio (~1.0) -> Memory is long (Iceberg / Splitting)
        # Low Ratio (->0.0) -> Memory implies noise (Efficiency)
        # We clamp denominator to avoid exploding ratios on zero correlation
        return (corr_short - corr_long).fill_null(0).alias("alpha_flow_decay")

# ==============================================================================
# 4. Trade Size Entropy (The "Smart Money" Detector)
# ==============================================================================
class TradeSizeEntropy(BaseAlpha):
    """
    [Signal] Entropy of Trade Sizes (Coefficient of Variation)
    [Logic] Std(Vol) / Mean(Vol)
    [Predicts] Low Entropy = Algo (Iceberg); High Entropy = Retail/Chaos.
    [Status] Kept as is (Logic is solid).
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)
        vol = pl.col("volume").fill_null(0)

        roll_mean = vol.rolling_mean(window)
        roll_std = vol.rolling_std(window)

        # CV calculation
        return (roll_std / (roll_mean + 1e-9)).alias("alpha_size_entropy")

# ==============================================================================
# 5. Fractal Dimension (The "Regime" Switch)
# ==============================================================================
class FractalDimension(BaseAlpha):
    """
    [Signal] Variance Ratio (Hurst Proxy)
    [Logic] Var(2-period) / (2 * Var(1-period))
    [Predicts] Trend (>1) vs Mean Reversion (<1).
    [Status] Kept as is (Classic Lo-MacKinlay).
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)

        r1 = pl.col("close").forward_fill().log().diff().fill_null(0)
        r2 = pl.col("close").forward_fill().log().diff(n=2).fill_null(0)

        var1 = r1.rolling_var(window)
        var2 = r2.rolling_var(window)

        vr = var2 / (2 * var1 + 1e-9)
        return vr.alias("alpha_fractal_dim")

class LevelCrossCount(BaseAlpha):
    """
    [Signal] Price Revisit Frequency (Cross Count)
    [Logic] Counts how many times price crosses its own rolling mean.
    [Predicts] High Cross -> Mean Reversion/Balance; Low Cross -> Strong Trending.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)
        
        close = pl.col("close").forward_fill()
        mid_line = close.rolling_mean(window)
        
        cross = ((close > mid_line) & (close.shift(1) < mid_line)) | \
                ((close < mid_line) & (close.shift(1) > mid_line))
        
        return cross.cast(pl.Int32).rolling_sum(window).alias("alpha_cross_count")
