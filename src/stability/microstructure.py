
import polars as pl
import numpy as np
from alpha.base import BaseAlpha

class MicrostructureGuard:
    """
    [Real-World Robustness]
    1. Signal Half-Life: Filters 'Ghost Signals' that decay faster than latency.
    2. Adaptive Kalman Gain: Dynamic Q/R adjustment based on innovation.
    """

    # ==============================================================================
    # 1. Signal Half-Life Detection (The "Ghost" Filter)
    # ==============================================================================
    @staticmethod
    def calc_signal_half_life(df: pl.LazyFrame, signal_col: str = "kalman_noise", window: int = 100) -> pl.LazyFrame:
        """
        [Logic]
        Model signal as OU Process (AR1): x_t = rho * x_{t-1} + epsilon
        Half-Life = -ln(2) / ln(|rho|)

        [Fix] Added abs() to rho because mean-reverting signals have negative correlation.
        """
        q = df.with_columns([
            pl.col(signal_col).alias("x"),
            pl.col(signal_col).shift(1).alias("x_lag")
        ])

        # 1. Rolling Autocorrelation
        q = q.with_columns([
            pl.when(pl.col("rho") <= 0)
              .then(pl.lit(0.0))
              .otherwise(-np.log(2) / pl.col("rho").clip(1e-4, 0.9999).log())
              .alias("signal_half_life")
        ])

        # 2. Calculate Half-Life (Bars)
        # Use abs(rho) to handle oscillating mean reversion
        # Clip rho to [0.01, 0.99] to prevent div/0 or log(0)
        q = q.with_columns([
            (
                -np.log(2) / pl.col("rho").abs().clip(0.01, 0.99).log()
            ).alias("signal_half_life")
        ])

        return q.drop(["x", "x_lag", "rho"])

    # ==============================================================================
    # 2. Adaptive Kalman Monitor (Innovation-based)
    # ==============================================================================
    @staticmethod
    def adaptive_kalman_monitor(df: pl.LazyFrame, price_col="close", window=50) -> pl.LazyFrame:
        """
        [Logic]
        Monitor 'Innovation' (Volatility of price changes).
        If Market is Turbulent (High Innovation) -> We need faster adaptation (Low Span).
        If Market is Calm (Low Innovation) -> We trust the trend (High Span).
        """
        # 1. Innovation Proxy: Squared Returns
        q = df.with_columns([
            pl.col(price_col).log().diff().pow(2).rolling_mean(window).alias("innovation_var")
        ])

        # 2. Adaptive Span using Z-Score normalization (Faster than Rank)
        # Z-Score the innovation to see if it's anomalous
        q = q.with_columns([
            (
                (pl.col("innovation_var") - pl.col("innovation_var").rolling_mean(window*5)) /
                (pl.col("innovation_var").rolling_std(window*5) + 1e-9)
            ).alias("innovation_z")
        ])

        # 3. Map Z-Score to Span Multiplier
        # High Z (Shock) -> Multiplier < 1 (Fast)
        # Low Z (Calm) -> Multiplier > 1 (Slow)
        # Sigmoid-like mapping can be used, here we use simple inverse clamping
        q = q.with_columns([
            (1.0 / (1.0 + pl.col("innovation_z").clip(0, 3.0))).alias("adaptive_kalman_scalar")
        ])

        return q.drop(["innovation_var", "innovation_z"])

    # ==============================================================================
    # 3. Latency Guard (Execution Feasibility)
    # ==============================================================================
    @staticmethod
    def check_execution_feasibility(df: pl.LazyFrame, latency_bars: int = 2) -> pl.LazyFrame:
        """
        If Signal Half-Life < System Latency, the alpha is unreachable.
        """
        return df.with_columns([
            (
                1.0 / (1.0 + (latency_bars / (pl.col("signal_half_life") + 1e-9)).exp())
            ).alias("execution_confidence_score")
        ])
