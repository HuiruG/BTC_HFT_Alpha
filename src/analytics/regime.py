
import polars as pl
import numpy as np

# 如果你有 BaseAlpha 类，可以继承。为了通用性，这里写成独立的函数/类。
class RegimeFactors:
    
    @staticmethod
    def get_fractal_efficiency(close_col="close", window=30):
        """
        [Trend Efficiency] Kaufman Efficiency Ratio
        1.0 = Perfect Trend, 0.0 = Pure Noise
        """
        change = pl.col(close_col).diff(n=window).abs()
        path = pl.col(close_col).diff().abs().rolling_sum(window)
        return (change / (path + 1e-9)).alias(f"regime_fractal_{window}")

    @staticmethod
    def get_market_temperature(close_col="close", window=50):
        """
        [Phase Transition] Energy (Vol) vs Dissipation (Friction)
        High Temp = Unstable / Crash Prone
        """
        # Energy: Volatility Squared
        ret = pl.col(close_col).log().diff()
        energy = (ret.pow(2)).rolling_mean(window)
        
        # Friction: 1 - Efficiency
        change = pl.col(close_col).diff(n=10).abs()
        path = pl.col(close_col).diff().abs().rolling_sum(10)
        er = change / (path + 1e-9)
        
        # Temp
        temp = energy / (1.0 - er + 0.1)
        # Z-Score Normalization
        return ((temp - temp.rolling_mean(window*5)) / (temp.rolling_std(window*5) + 1e-9)).alias(f"regime_temp_{window}")

    @staticmethod
    def get_info_entropy(close_col="close", window=60):
        """
        [Predictability] Shannon Entropy
        1.0 = Max Randomness (Don't Trade), 0.0 = Deterministic
        """
        is_up = (pl.col(close_col).diff() > 0).cast(pl.Float64)
        p_up = is_up.rolling_mean(window).clip(0.001, 0.999)
        p_down = 1.0 - p_up
        entropy = -1 * (p_up * p_up.log(2) + p_down * p_down.log(2))
        return entropy.alias(f"regime_entropy_{window}")

    @staticmethod
    def get_fisher_proxy(close_col="close", window=100):
        """
        [Tail Risk] Kurtosis Change Rate
        Spike = Regime Shift Imminent
        """
        ret = pl.col(close_col).log().diff()
        z = (ret - ret.rolling_mean(window)) / (ret.rolling_std(window) + 1e-9)
        kurt = z.pow(4).rolling_mean(window)
        return kurt.diff().abs().rolling_mean(window).alias(f"regime_fisher_{window}")
