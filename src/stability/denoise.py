
import polars as pl
import numpy as np

class NoiseCanceller:
    """
    [HFT Robustness Layer]
    1. Kalman Filter: Separates 'True Price' from 'Microstructure Noise'.
    2. Robust Volatility: Uses MAD (Median Abs Dev) instead of StdDev to ignore outliers.
    3. Structural Break: CUSUM-like detection for regime shifts.
    """

    # ==============================================================================
    # 1. 1-D Kalman Filter (Simplified for Speed)
    # ==============================================================================
    @staticmethod
    def kalman_filter(df: pl.LazyFrame, price_col: str = "close", process_var=1e-5, measure_var=1e-3) -> pl.LazyFrame:
        """
        [Logic]
        State: x_t = x_{t-1} (Random Walk assumption for True Price)
        Observation: z_t = x_t + noise (Microstructure noise)
        
        Using EWM approximation for speed.
        Ratio Q/R determines the 'Trust' in new data vs old model.
        """
        # Q (Process Variance) / R (Measurement Variance)
        ratio = process_var / measure_var
        
        # Solving steady state Riccati equation approximation for span
        # High Ratio -> High Trust in Data -> Short Span
        # Low Ratio -> High Trust in Model -> Long Span
        span_equiv = 2.0 / (np.sqrt(ratio) + 1e-9)

        # Polars EWM only accepts 'span'
        return df.with_columns([
            pl.col(price_col).ewm_mean(span=span_equiv).alias("kalman_price"),
            
            # The Residual is the Microstructure Noise
            (pl.col(price_col) - pl.col(price_col).ewm_mean(span=span_equiv)).alias("kalman_noise")
        ])

    # ==============================================================================
    # 2. Robust Volatility (MAD > StdDev)
    # ==============================================================================
    @staticmethod
    def robust_volatility(df: pl.LazyFrame, col: str = "close", window: int = 50) -> pl.LazyFrame:
        """
        [Logic] Standard Deviation is sensitive to outliers (Flash Crashes).
                MAD (Median Absolute Deviation) is robust.
        [Formula] MAD = Median(|x - Median(x)|) * 1.4826
        """
        # 1. Log Returns
        # Note: Using forward fill to handle potential NaNs from diff
        q = df.with_columns(pl.col(col).log().diff().fill_null(0).alias("log_ret"))

        # 2. Rolling Median of Returns (Center)
        # [FIX] Polars > 0.19 uses 'window_size' instead of 'window'
        q = q.with_columns(
            pl.col("log_ret").rolling_median(window_size=window).alias("med_ret")
        )

        # 3. Rolling Median of Absolute Deviation
        q = q.with_columns(
            (pl.col("log_ret") - pl.col("med_ret")).abs().rolling_median(window_size=window).alias("mad_raw")
        )

        # 4. Scale to Sigma equivalent (Normal Dist assumption for scaling)
        return q.with_columns(
            (pl.col("mad_raw") * 1.4826).alias("vol_robust")
        ).drop(["log_ret", "med_ret", "mad_raw"])

    # ==============================================================================
    # 3. Signal-to-Noise Ratio (SNR) Dynamic
    # ==============================================================================
    @staticmethod
    def calc_snr(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        [Logic] Trend Strength / Noise Level (Kaufman Efficiency)
        """
        window = 50
        
        # [FIX] window_size
        directional = pl.col("close").diff(n=window).abs()
        noise_path = pl.col("close").diff().abs().rolling_sum(window_size=window)

        return df.with_columns(
            (directional / (noise_path + 1e-9)).alias("system_snr")
        )

    # ==============================================================================
    # 4. False Breakout Filter (Kalman Divergence)
    # ==============================================================================
    @staticmethod
    def detect_false_breakout(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        [Logic] A breakout is "False" if Price moves but Kalman Price doesn't follow.
        [Signal] Z-Score of Kalman Noise.
        """
        # We calculate Rolling Std of the Noise for Z-Score normalization
        
        # [FIX] window_size
        q = df.with_columns(
            pl.col("kalman_noise").rolling_std(window_size=100).alias("noise_sigma")
        )

        # Return boolean signal
        return q.with_columns(
            ((pl.col("kalman_noise") / (pl.col("noise_sigma") + 1e-9)).abs() > 2.0).alias("is_overextended")
        )
