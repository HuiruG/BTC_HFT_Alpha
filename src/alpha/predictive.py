
import polars as pl
import numpy as np
from alpha.base import BaseAlpha

# ==============================================================================
# GROUP A: ORDER FLOW & PRESSURE 
# ==============================================================================

class TradeImbalance(BaseAlpha):
    """
    [1] Aggressor Side Imbalance
    [Logic] (Buy - Sell) / (Buy + Sell)
    [Fix] Added fill_null and epsilon for zero volume bars.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 5)

        b = pl.col("buy_vol").fill_null(0)
        s = pl.col("sell_vol").fill_null(0)

        # Smooth lightly to avoid noise
        b_smooth = b.rolling_mean(window)
        s_smooth = s.rolling_mean(window)

        return ((b_smooth - s_smooth) / (b_smooth + s_smooth + 1e-4)).alias("alpha_trade_imbalance")

class VolumeWeightedPersistence(BaseAlpha):
    """
    [2] Flow Persistence (Smart Money Proxy)
    [Logic] Rolling Mean of Signed Flow Ratio
    [Fix] Replaced simple Sign() with Volume Weighted approach.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 20)

        net_vol = (pl.col("buy_vol").fill_null(0) - pl.col("sell_vol").fill_null(0))
        total_vol = (pl.col("buy_vol").fill_null(0) + pl.col("sell_vol").fill_null(0))

        flow_ratio = net_vol / (total_vol + 1e-9)
        return flow_ratio.rolling_mean(window).alias("alpha_flow_persistence")

# ==============================================================================
# GROUP B: TREND & MOMENTUM 
# ==============================================================================

class SmoothedMomentum(BaseAlpha):
    """
    [3] Noise-Filtered Momentum
    [Logic] Diff(EMA(Price))
    [Fix] Used EMA instead of raw price to filter bid-ask bounce.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        period = self.params.get("period", 10)
        source = self.params.get("source", "vwap")

        col_expr = pl.col(source) if source in df.collect_schema().names() else pl.col("close")
        # Forward fill is crucial for VWAP which might be null on no-trade bars
        col_clean = col_expr.forward_fill()

        smooth_price = col_clean.ewm_mean(span=max(2, int(period/2)), adjust=False)
        return smooth_price.log().diff(n=period).alias("alpha_momentum")

class VWAPDeviation(BaseAlpha):
    """
    [4] Mean Reversion Proxy
    [Logic] (Close - VWAP) / VWAP
    [Fix] Added robustness for null VWAP.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        close = pl.col("close").forward_fill()
        vwap = pl.col("vwap").forward_fill()

        return ((close - vwap) / (vwap + 1e-9)).alias("alpha_vwap_deviation")

# ==============================================================================
# GROUP C: VOLATILITY & RISK 
# ==============================================================================

class ParkinsonVolatility(BaseAlpha):
    """
    [5] Parkinson High-Low Volatility
    [Logic] High-Low Range estimator
    [Fix] Added protection against zero-range or null data.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 20)

        high = pl.col("high").fill_null(pl.col("close"))
        low = pl.col("low").fill_null(pl.col("close"))

        # Log range squared
        log_hl = (high / (low + 1e-9)).log()
        sq_range = log_hl ** 2

        const = 1.0 / (4.0 * np.log(2.0))
        return (const * sq_range.rolling_mean(window)).sqrt().alias("alpha_parkinson_vol")

# ==============================================================================
# GROUP D: LIQUIDITY & IMPACT 
# ==============================================================================

class KyleLambda(BaseAlpha):
    """
    [6] Kyle's Lambda (Impact Cost)
    [Logic] Correlation(Flow, Return) * (Sigma_Ret / Sigma_Flow)
    [Fix] Full NaN protection for low volatility periods.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)

        y = pl.col("close").forward_fill().log().diff().fill_null(0)
        x = (pl.col("buy_vol").fill_null(0) - pl.col("sell_vol").fill_null(0))

        corr = pl.rolling_corr(x, y, window_size=window).fill_null(0)
        std_x = x.rolling_std(window)
        std_y = y.rolling_std(window)

        return (corr * (std_y / (std_x + 1e-9))).fill_nan(0).alias("alpha_kyle_lambda")

class AmihudLiquidity(BaseAlpha):
    """
    [7] Amihud Illiquidity
    [Logic] |Return| / Turnover
    [Fix] Used Turnover (Quote Volume) instead of Base Volume for better accuracy.
    """
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)

        abs_ret = pl.col("close").forward_fill().log().diff().abs().fill_null(0)
        turnover = pl.col("turnover").fill_null(0) # Quote volume (USDT)

        ratio = abs_ret / (turnover + 1e-9)
        return ratio.rolling_mean(window).alias("alpha_amihud")

class SpreadResilience(BaseAlpha):
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        window = self.params.get("window", 50)
        h = pl.col("high").cast(pl.Float64)
        l = pl.col("low").cast(pl.Float64)
        c = pl.col("close").cast(pl.Float64)
        _s = (h - l) / (c + 1e-9)
        return pl.rolling_corr(_s, _s.shift(1), window_size=window).fill_null(0).alias("alpha_spread_resilience")
