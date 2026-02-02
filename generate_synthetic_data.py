import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_tick_data(n_ticks=100000):
    """
    Generate realistic synthetic Bitcoin tick data.
    """
    start_time = datetime(2025, 12, 1)

    times = [start_time + timedelta(milliseconds=i*100 + np.random.randint(0, 50)) for i in range(n_ticks)]

    # Random walk for price
    returns = np.random.normal(0, 0.0001, n_ticks)
    price = 60000 * np.exp(np.cumsum(returns))

    # Add microstructure noise (bid-ask bounce)
    noise = np.random.choice([-0.5, 0.5], n_ticks)
    price += noise

    # Volume
    qty = np.random.gamma(2, 0.1, n_ticks)

    # Side
    is_buyer_maker = np.random.choice([True, False], n_ticks)

    df = pd.DataFrame({
        "time": [int(t.timestamp() * 1000) for t in times],
        "price": price,
        "qty": qty,
        "is_buyer_maker": is_buyer_maker
    })

    return df

if __name__ == "__main__":
    print("Generating synthetic tick data...")
    df = generate_tick_data()
    df.to_csv("btc_data.csv", index=False)
    print("btc_data.csv generated.")
