import polars as pl
PARQUET_FILE = r'C:\btc_quant\data\processed\quant_data.parquet'
def peek_data():
    print('\n[SYSTEM] Accessing Memory-Mapped Parquet File...')
    df = pl.scan_parquet(PARQUET_FILE).head(5).collect()
    pl.Config.set_tbl_cols(df.width)
    print(f'[INFO] Total Columns Detected: {df.width}')
    print('[INFO] First 5 Rows (Raw Strings):\n')
    print(df)
if __name__ == '__main__':
    peek_data()
