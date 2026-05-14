import polars as pl
import numpy as np

INPUT = '/btc_quant/data/processed/quant_data_clean.parquet'

def run_physics_sweep():
    print('\n[SYSTEM] Initiating Brownian Motion Sweep (Quadratic Time Decay)...')
    print('[SYSTEM] Function Active: Horizon = k * (Multiplier ^ 2)\n')
    
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
    
    seq_len = 60
    max_multiplier = 25
    k_constant = 2  # Base time constant
    
    max_horizon = k_constant * (max_multiplier ** 2)
    valid_rows = len(np_data) - seq_len - max_horizon
    
    print(f"{'Mult':<6} | {'Horizon (m^2)':<14} | {'Timeouts (0)':<15} | {'Profits (1)':<15} | {'Losses (2)':<15}")
    print("-" * 75)

    for m in range(2, max_multiplier + 1):
        pt_mult = float(m)
        sl_mult = float(m)
        
        # THE QUADRATIC SLIDING SCALE
        horizon = int(k_constant * (m ** 2))
        
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
                
        timeouts = (np.sum(y_all == 0) / valid_rows) * 100
        profits = (np.sum(y_all == 1) / valid_rows) * 100
        losses = (np.sum(y_all == 2) / valid_rows) * 100
        
        print(f"{str(m)+'x':<6} | {str(horizon)+'m':<14} | {timeouts:>13.2f}% | {profits:>13.2f}% | {losses:>13.2f}%")

if __name__ == '__main__':
    run_physics_sweep()
