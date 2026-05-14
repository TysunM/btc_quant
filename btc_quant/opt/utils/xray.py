import polars as pl
import numpy as np

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

def xray_the_math():
    print('\n[SYSTEM] X-Raying the Matrix Math (PowerShell Safe)...')
    df = (pl.read_parquet(INPUT)
          .tail(1000)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.0)
          .with_columns(
              ((pl.col('Price') / pl.col('Price').shift(1)) - 1.0)
              .rolling_std(window_size=100)
              .fill_null(0.0)
              .alias('Volatility')
          ))
    
    np_data = df.to_numpy()
    pt_mult = 2.0
    
    print(f'{'Price':<12} | {'Volatility':<12} | {'Target Price':<12} | {'Req Move':<12}')
    print('-' * 55)
    
    for i in range(100, 110):
        price = np_data[i, 0]
        vol = np_data[i, 6]
        pt_price = price * (1 + (vol * pt_mult))
        req_move = pt_price - price
        
        print(f'{price:<12.2f} | {vol:<12.6f} | {pt_price:<12.2f} | {req_move:<12.2f}')

if __name__ == '__main__':
    xray_the_math()
