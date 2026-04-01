import numpy as np
import polars as pl
import logging
from numba import njit

logger = logging.getLogger("QuantEngine.Backtest")

@njit
def fast_backtest_logic(prices, signals, fee_bps=2.0, horizon=50):
    """
    Numba-accelerated event-driven backtest simulation.
    Handles path dependency, holding periods, and fees.
    """
    n = len(prices)
    pnl = np.zeros(n)
    position = 0 # 0: Neutral, 1: Long, 2: Short
    entry_price = 0.0
    holding_count = 0

    for i in range(n):
        current_price = prices[i]
        signal = signals[i]

        # Check exit conditions for existing position
        if position != 0:
            holding_count += 1
            # Simple horizon exit for this simulation
            if holding_count >= horizon:
                exit_price = current_price
                if position == 1: # Long
                    trade_pnl = (exit_price / entry_price - 1)
                else: # Short
                    trade_pnl = (1 - exit_price / entry_price)

                # Apply fee twice (entry and exit)
                pnl[i] = trade_pnl - 2 * fee_bps / 10000
                position = 0
                holding_count = 0

        # Check entry conditions
        if position == 0:
            if signal == 1: # Buy/Up-move
                position = 1
                entry_price = current_price
                holding_count = 0
            elif signal == 2: # Sell/Down-move
                position = 2
                entry_price = current_price
                holding_count = 0

    return pnl

def run_backtest(df, model, scaler, features, fee_bps=2.0):
    logger.info("Running event-driven backtest...")

    # Inference
    X = df.select(features).to_numpy()
    X_scaled = scaler.transform(X)

    import torch
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(X_scaled)
        logits = model(inputs)
        signals = torch.argmax(logits, dim=1).numpy()

    prices = df["close"].to_numpy()

    pnl = fast_backtest_logic(prices, signals, fee_bps=fee_bps)

    cumulative_pnl = np.cumsum(pnl)
    total_return = cumulative_pnl[-1]

    logger.info(f"Backtest complete. Total Return: {total_return:.4%}")

    return cumulative_pnl
