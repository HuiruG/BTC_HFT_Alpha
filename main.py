import os
import sys
import logging
import polars as pl
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.cleaner import DataCleaner
from data.bars import build_adaptive_hft_bars_v2
from data.zoo_builder import UltimateZooBuilder
from data.labeler import TripleBarrierLabeler
from pipeline.train import run_training_pipeline
from pipeline.backtest import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s | %(name)s | %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("QuantEngine.Main")

class Config:
    def __init__(self):
        self.raw_data_path = PROJECT_ROOT / "btc_data.csv"
        self.processed_dir = PROJECT_ROOT / "data/processed"
        self.feature_zoo_path = PROJECT_ROOT / "data/feature_zoo/ultimate_zoo.parquet"
        self.labeled_data_path = PROJECT_ROOT / "data/cache/labeled_dataset.parquet"

        # Hyperparameters
        self.JUMP_Z_TH = 8.0
        self.JUMP_BPS_TH = 20.0
        self.ROLLING_WINDOW = 500

        # Sampling
        self.TARGET_FREQ_SEC = 10.0
        self.ROLLING_WINDOW_HOURS = 1

        # Training
        self.EPOCHS = 5
        self.BATCH_SIZE = 1024
        self.FEE_BPS = 2.0

def main():
    config = Config()

    # 0. Generate synthetic data if not exists
    if not config.raw_data_path.exists():
        logger.info("Raw data not found. Generating synthetic data...")
        from generate_synthetic_data import generate_tick_data
        df_tick = generate_tick_data(n_ticks=200000)
        df_tick.to_csv(config.raw_data_path, index=False)

    # 1. Cleaning
    logger.info("Step 1: Data Cleaning...")
    # Convert CSV to Parquet first for cleaner pipeline
    raw_parquet = PROJECT_ROOT / "data/raw/btc_raw.parquet"
    raw_parquet.parent.mkdir(parents=True, exist_ok=True)

    pl.scan_csv(config.raw_data_path).with_columns([
        pl.from_epoch("time", time_unit="ms").alias("timestamp"),
        pl.when(pl.col("is_buyer_maker")).then(pl.lit(-1, dtype=pl.Int8)).otherwise(pl.lit(1, dtype=pl.Int8)).alias("side")
    ]).select([
        pl.col("timestamp").alias("dt"),
        pl.col("price").cast(pl.Float32),
        pl.col("qty").cast(pl.Float32),
        pl.col("side")
    ]).sink_parquet(raw_parquet)

    cleaner = DataCleaner(config)
    df_clean = cleaner.execute_pipeline(raw_parquet)

    cleaned_path = config.processed_dir / "btc_cleaned.parquet"
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(cleaned_path)

    # 2. Sampling
    logger.info("Step 2: Adaptive Bar Sampling...")
    bars_path = config.processed_dir / "btc_bars.parquet"
    df_bars = build_adaptive_hft_bars_v2(
        str(cleaned_path),
        str(bars_path),
        target_freq_sec=config.TARGET_FREQ_SEC,
        rolling_window_hours=config.ROLLING_WINDOW_HOURS
    )

    # 3. Feature Zoo
    logger.info("Step 3: Building Feature Zoo...")
    zoo_builder = UltimateZooBuilder()
    df_zoo = zoo_builder.build(str(bars_path), str(config.feature_zoo_path))

    # 4. Labeling
    logger.info("Step 4: Triple-Barrier Labeling...")
    labeler = TripleBarrierLabeler()
    df_labeled = labeler.apply(df_zoo)
    os.makedirs(os.path.dirname(config.labeled_data_path), exist_ok=True)
    df_labeled.write_parquet(config.labeled_data_path)

    # 5. Training
    logger.info("Step 5: Training Deep Alpha Model...")
    features = [c for c in df_labeled.columns if c.startswith("alpha_") or c == "frac_diff_price"]
    model, scaler, features = run_training_pipeline(df_labeled, features, epochs=config.EPOCHS)

    # 6. Backtesting
    logger.info("Step 6: Running Event-Driven Backtest...")
    # Drop rows with null features for backtest consistency
    df_backtest = df_labeled.drop_nulls(subset=features)
    cumulative_pnl = run_backtest(df_backtest, model, scaler, features, fee_bps=config.FEE_BPS)

    logger.info("Pipeline executed successfully.")
    print(f"Final Cumulative Return: {cumulative_pnl[-1]:.4%}")

if __name__ == "__main__":
    main()
