import polars as pl
import os

RAW_DIR = '/btc_quant/data/raw/'
PROCESSED_DIR = '/btc_quant/data/processed/'

def sync_all():
    print("[SYSTEM] Synchronizing Multi-Source Alpha Repository...")

    # 1. NinjaTrader Daily Repair (Fixing the 8-day gaps)
    # Using ';' as delimiter as noted in your file summary
    nt8 = pl.read_csv(RAW_DIR + 'NinjaTrader8DF.csv', separator=';', has_header=False, 
                      new_columns=['Date', 'Open', 'High', 'Low', 'Price', 'Vol'])
    
    # Format YYYYMMDD to Date object
    nt8 = nt8.with_columns(pl.col('Date').cast(pl.String).str.strptime(pl.Date, "%Y%m%d"))
    
    # Gap-Filling Logic
    nt8 = nt8.sort('Date').upsample(time_column="Date", every="1d")
    nt8 = nt8.with_columns([
        pl.col('Price').interpolate(),
        pl.col('Open').fill_null(pl.col('Price')),
        pl.col('High').fill_null(pl.col('Price')),
        pl.col('Low').fill_null(pl.col('Price')),
        pl.col('Vol').fill_null(0.0)
    ])
    print("   -> NinjaTrader Gaps Repaired.")

    # 2. Implied Volatility (DVOL) Integration
    dvol = pl.read_csv(RAW_DIR + 'BTC_Volatility_DVOL_plus.csv').select([
        pl.col('date').str.strptime(pl.Date, "%Y-%m-%d").alias('Date'),
        pl.col('close').alias('Implied_Vol'),
        pl.col('1d_implied_move')
    ])

    # 3. On-Chain Intelligence (Blockchain Blocks)
    # Skipping the URL header line in the CSV
    blocks = pl.read_csv(RAW_DIR + 'Blockchain_BTC_historical_blocks.csv', skip_rows=1).select([
        pl.col('date').str.strptime(pl.Date, "%Y-%m-%d").alias('Date'),
        pl.col('hashrate').alias('Network_Hashrate'),
        pl.col('total_transactions').alias('TX_Density')
    ])

    # 4. Binance Premium Stats (Institutional Imbalance)
    binance = pl.read_csv(RAW_DIR + 'Binance_summary_statistics_BTCUSDT_premium.csv').select([
        pl.col('date').str.strptime(pl.Date, "%Y-%m-%d").alias('Date'),
        (pl.col('buy_total_volume') / (pl.col('sell_total_volume') + 1e-7)).alias('Buy_Sell_Ratio'),
        pl.col('average_usd_size').alias('Inst_Trade_Size')
    ])

    # 5. Master Merge - Creating the Strategic Bias Layer
    strategic_bias = nt8.join(dvol, on='Date', how='left') \
                        .join(blocks, on='Date', how='left') \
                        .join(binance, on='Date', how='left') \
                        .fill_null(strategy='forward')
    
    print(f"[SUCCESS] Strategic Bias Layer Locked: {len(strategic_bias.columns)} Active Factors.")
    strategic_bias.write_parquet(PROCESSED_DIR + 'master_strategic_bias.parquet')

if __name__ == '__main__':
    sync_all()
