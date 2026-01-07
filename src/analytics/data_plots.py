
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit

def setup_style():
   
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

def plot_volatility_signature(df: pl.DataFrame, output_dir: Path):
    """
     (Volatility Signature)
    """
    

   
    intervals = ["100ms", "500ms", "1s", "5s", "10s", "30s", "1m", "5m"]
    volatilities = []

   
    df = df.sort("dt")

    for interval in intervals:
        
        resampled = (
            df.group_by_dynamic("dt", every=interval)
            .agg(pl.col("price").last().alias("close"))
            .select([
                pl.col("close").log().diff().alias("log_ret") # 对数收益率
            ])
            .drop_nulls()
        )

     
        std_dev = resampled["log_ret"].std()

        # sqrt(Seconds_in_Year / Interval_Seconds)
        seconds_map = {
            "100ms": 0.1, "500ms": 0.5, "1s": 1, "5s": 5, "10s": 10,
            "30s": 30, "1m": 60, "5m": 300
        }
        sec = seconds_map[interval]
        annual_factor = np.sqrt((365 * 24 * 3600) / sec)

      
        if std_dev is not None:
            volatilities.append(std_dev * annual_factor)
        else:
            volatilities.append(0)

    
    plt.figure()
    plt.plot(intervals, volatilities, marker='o', linestyle='-', linewidth=2.5, color='#2c3e50')
    plt.title("Volatility Signature: Realized Vol vs Sampling Freq")
    plt.ylabel("Annualized Volatility (Log Scale)")
    plt.xlabel("Sampling Frequency")
    plt.yscale("log") 
    plt.grid(True, which="both", ls="--", alpha=0.5)

    
    plt.text(0, volatilities[0]*0.95, "Microstructure Noise Zone", fontsize=10, color='red')
    plt.text(len(intervals)-1, volatilities[-1]*1.05, "Fundamental Volatility", fontsize=10, color='green', ha='right')

    out_path = output_dir / "volatility_signature.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Volatility Signature: {out_path}")

def plot_trade_sign_autocorr(df: pl.DataFrame, output_dir: Path):
    """
     (Trade Sign Autocorrelation)
     (Order Splitting)。
    """
    

    # Side: Buy=1, Sell=-1
    subset = df.head(1_000_000)
    signs = subset["side"].to_numpy()

    lags = np.arange(1, 51) 
    acfs = []

    
    # Corr(X_t, X_{t-k})
    y = signs - signs.mean() 
    n = len(y)
    var_y = np.var(y)

    for lag in lags:
        c = np.mean(y[:-lag] * y[lag:]) / var_y
        acfs.append(c)

    # (Power Law Fit) y = A * x^(-gamma)
    def power_law(x, a, gamma):
        return a * np.power(x, -gamma)

    try:
        popt, _ = curve_fit(power_law, lags, acfs, maxfev=2000)
        fitted_y = power_law(lags, *popt)
        gamma = popt[1]
        fit_label = f"Power Law Fit ($\gamma={gamma:.2f}$)"
    except:
        fitted_y = None
        fit_label = "Fit Failed"

   
    plt.figure()
    plt.bar(lags, acfs, color='#3498db', alpha=0.7, label='Autocorrelation', width=0.8)
    if fitted_y is not None:
        plt.plot(lags, fitted_y, 'r--', linewidth=2, label=fit_label)

    plt.title("Trade Sign Autocorrelation (Order Splitting Evidence)")
    plt.xlabel("Lag (Number of Trades)")
    plt.ylabel("Autocorrelation")
    plt.axhline(0, color='black', lw=0.8)
    plt.legend()
    plt.grid(axis='y', ls="--", alpha=0.5)

    out_path = output_dir / "trade_sign_autocorr.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f" Trade Sign Autocorr : {out_path}")

    
    
def analytics(config):
    setup_style()

    input_path = config.processed / "btcusdt_2025_12_cleaned.parquet"
    output_dir = config.outputs / "plots/data_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

  
    df = pl.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows.")


    plot_volatility_signature(df, output_dir)


    plot_trade_sign_autocorr(df, output_dir)
