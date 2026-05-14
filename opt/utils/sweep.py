import polars as pl
import numpy as np

INPUT = r'C:\btc_quant\data\processed\quant_data_clean.parquet'

def run_parameter_sweep():
    print('\n[SYSTEM] Initiating Triple Barrier Parameter Sweep (Pure Data)...')
    # Taking a 500k row slice so the sweep executes rapidly
    df = (pl.read_parquet(INPUT)
          .tail(500000)
          .select(['Price', 'Open', 'High', 'Low', 'Vol', 'Change'])
          .fill_null(strategy='forward').fill_null(strategy='backward').fill_null(0.0)
          .with_columns(
              ((pl.col('Price') / pl.col('Price').shift(1)) - 1.0)
              .rolling_std(window_size=100)
              .fill_null(0.0)
              .alias('Volatility')
          ))
    
    np_data = df.to_numpy().astype(np.float32)
    np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    profiles = [
        {'name': 'The Original (Wide targets / Short time)', 'horizon': 30, 'pt': 2.0, 'sl': 2.0},
        {'name': 'The Patient (Wide targets / Long time)', 'horizon': 180, 'pt': 2.0, 'sl': 2.0},
        {'name': 'The Day Trader (Mid targets / Mid time)', 'horizon': 60, 'pt': 1.0, 'sl': 1.0},
        {'name': 'The Scalper (Tight targets / Short time)', 'horizon': 30, 'pt': 0.5, 'sl': 0.5}
    ]
    
    seq_len = 60
    
    for p in profiles:
        horizon = p['horizon']
        pt_mult = p['pt']
        sl_mult = p['sl']
        p_name = p['name']
        
        valid_rows = len(np_data) - seq_len - horizon
        y_all = np.zeros((valid_rows,), dtype=np.int64)
        
        for i in range(valid_rows):
            current_price = np_data[i+seq_len - 1, 0]
            current_vol = np_data[i+seq_len - 1, 6]
            if current_vol == 0: current_vol = 0.0001
                
            pt_price = current_price * (1 + (current_vol * pt_mult))
            sl_price = current_price * (1 - (current_vol * sl_mult))
            
            future_window = np_data[i+seq_len : i+seq_len+horizon, 0]
            
            hit_pt = np.argmax(future_window >= pt_price)
            hit_sl = np.argmax(future_window <= sl_price)
            
            pt_valid = future_window[hit_pt] >= pt_price
            sl_valid = future_window[hit_sl] <= sl_price
            
            if pt_valid and sl_valid: y_all[i] = 1 if hit_pt < hit_sl else 2
            elif pt_valid: y_all[i] = 1
            elif sl_valid: y_all[i] = 2
            else: y_all[i] = 0
                
        timeouts = np.sum(y_all == 0)
        profits = np.sum(y_all == 1)
        losses = np.sum(y_all == 2)
        total = len(y_all)
        
        print(f'\n--- {p_name} ---')
        print(f'Horizon: {horizon} ticks | Profit: {pt_mult}x Vol | Loss: {sl_mult}x Vol')
        print(f'   -> [0] Timeouts:   {(timeouts/total)*100:.2f}%')
        print(f'   -> [1] Profits:    {(profits/total)*100:.2f}%')
        print(f'   -> [2] Stop-Loss:  {(losses/total)*100:.2f}%')

if __name__ == '__main__':
    run_parameter_sweep()
