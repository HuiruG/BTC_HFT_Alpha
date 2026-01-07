
import polars as pl
import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union

class BaseAlpha(ABC):
    

    def __init__(self, name: str, category: str, params: Dict = None):
        
        self.name = name
        self.category = category
        self.params = params if params else {}
        self.logger = logging.getLogger(f"Alpha.{name}")

    @abstractmethod
    def calculate(self, df: pl.LazyFrame) -> pl.Expr:
        
        pass

    def run(self, df: Union[pl.DataFrame, pl.LazyFrame], normalize: bool = True, shift: int = 1) -> pl.LazyFrame:
      
        q = df.lazy() if isinstance(df, pl.DataFrame) else df
        
        raw_factor_expr = self.calculate(q)
        
        target_col = f"{self.name}_raw"
        q = q.with_columns(raw_factor_expr.alias(target_col))

        if normalize:
            window = self.params.get('z_window', 300)
            
            processed_expr = self._robust_zscore(pl.col(target_col), window)
        else:
            processed_expr = pl.col(target_col)

        if shift > 0:
            processed_expr = processed_expr.shift(shift)
        q = q.with_columns(processed_expr.alias(self.name))
        
        return q

    # --------------------------------------------------------------------------
    # Quant Utility Library
    # --------------------------------------------------------------------------

    def _robust_zscore(self, expr: pl.Expr, window: int) -> pl.Expr:
        """
        Formula: (X - Median) / (MAD * 1.4826)
        """
        roll_med = expr.rolling_median(window_size=window)

        roll_mean = expr.rolling_mean(window_size=window)
        roll_std = expr.rolling_std(window_size=window)

        z = (expr - roll_mean) / (roll_std + 1e-8)

        return z.clip(-4.0, 4.0)

    def _decay(self, expr: pl.Expr, alpha: float) -> pl.Expr:
        """
        (EMA)
        """
        return expr.ewm_mean(alpha=alpha, adjust=False)

