import polars as pl
import time

INPUT = r'C:\btc_quant\data\processed\quant_data.parquet'
OUTPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

def clean_data():
    print('\n[SYSTEM] Initializing Data Scrubbing Protocol on P-Cores...')
    start = time.time()
    
    # Lazy load the messy data
    df = pl.scan_parquet(INPUT)
    
    # 1. Isolate the 7 real columns and assign the true headers
    df = df.select([
        pl.col('column_1').alias('Date'),
        pl.col('column_2').alias('Price'),
        pl.col('column_3').alias('Open'),
        pl.col('column_4').alias('High'),
        pl.col('column_5').alias('Low'),
        pl.col('column_6').alias('Vol'),
        pl.col('column_7').alias('Change')
    ])
    
    # 2. Filter out the header rows that got mixed into the data
    df = df.filter(pl.col('Date') != 'Date')
    
    # 3. Strip string artifacts and cast to Float32 for CUDA
    price_cols = ['Price', 'Open', 'High', 'Low']
    for col in price_cols:
        df = df.with_columns(pl.col(col).str.replace_all(',', '').cast(pl.Float32, strict=False))
        
    df = df.with_columns([
        # Convert K, M, B to scientific notation so Polars can cast it to float
        pl.col('Vol').str.replace_all('K', 'e3').str.replace_all('M', 'e6').str.replace_all('B', 'e9').str.replace_all('-', '0').cast(pl.Float32, strict=False),
        # Remove % and divide by 100 for true mathematical decimals
        (pl.col('Change').str.replace_all('%', '').cast(pl.Float32, strict=False) / 100.0).alias('Change')
    ])
    
    # 4. Stream to the final clean Parquet file
    df.sink_parquet(OUTPUT, compression='snappy')
    
    end = time.time()
    print(f'[SUCCESS] Matrix scrubbed and aligned in {round(end - start, 2)} seconds.')
    print(f'[OUTPUT] Target locked: {OUTPUT}')

if __name__ == '__main__':
    clean_data()
