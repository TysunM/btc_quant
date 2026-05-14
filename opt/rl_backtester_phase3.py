import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

PROCESSED_DIR = '/btc_quant/data/processed/'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_rl_backtest():
    print("\n[SYSTEM] Loading test matrix for Phase 3 Validation...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'rl_smart_money_data.parquet'), engine='pyarrow')
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    
    test_df['H-L'] = test_df['High'] - test_df['Low']
    test_df['H-C'] = np.abs(test_df['High'] - test_df['Price'].shift(1))
    test_df['L-C'] = np.abs(test_df['Low'] - test_df['Price'].shift(1))
    test_df['TR'] = test_df[['H-L', 'H-C', 'L-C']].max(axis=1)
    test_df['ATR'] = test_df['TR'].rolling(14).mean().bfill()
    
    features = test_df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date', 'H-L', 'H-C', 'L-C', 'TR', 'ATR'], errors='ignore').values
    prices = test_df['Price'].values
    highs = test_df['High'].values
    lows = test_df['Low'].values
    atrs = test_df['ATR'].values
    
    model = PPO.load(MODEL_PATH)
    
    capital = 10000.0
    position = 0  
    entry_price = 0.0
    bars_held = 0
    trade_log = []
    
    tp_mult = 3.5
    sl_mult = 1.2
    time_limit = 18 # 3-Day Rule
    fee = 0.0012
    
    # Using 0.3 as a balanced threshold for this exploratory model
    THRESHOLD = 0.30 
    
    i = 0
    while i < len(test_df) - time_limit:
        if position == 0:
            obs = features[i]
            action, _states = model.predict(obs, deterministic=True)
            action_value = action[0]
            
            if abs(action_value) >= THRESHOLD:
                position = 1 if action_value > 0 else -1
                position_size = abs(action_value)
                entry_price = prices[i]
                current_atr = atrs[i]
                
                if position == 1:
                    take_profit = entry_price + (current_atr * tp_mult)
                    stop_loss = entry_price - (current_atr * sl_mult)
                else:
                    take_profit = entry_price - (current_atr * tp_mult)
                    stop_loss = entry_price + (current_atr * sl_mult)
                bars_held = 0
                
        if position != 0:
            bars_held += 1
            future_idx = i + bars_held
            high, low, close = highs[future_idx], lows[future_idx], prices[future_idx]
            
            trade_over = False
            trade_return = 0.0
            reason = ""
            
            if (position == 1 and low <= stop_loss) or (position == -1 and high >= stop_loss):
                trade_return = (stop_loss - entry_price) / entry_price * position
                trade_over = True
                reason = "SL"
            elif (position == 1 and high >= take_profit) or (position == -1 and low <= take_profit):
                trade_return = (take_profit - entry_price) / entry_price * position
                trade_over = True
                reason = "TP"
            elif bars_held >= time_limit:
                trade_return = (close - entry_price) / entry_price * position
                trade_over = True
                reason = "TIME"
                
            if trade_over:
                net_return = (trade_return - fee) * position_size
                capital += capital * net_return
                trade_log.append({'Pct': net_return, 'Reason': reason})
                position = 0
                i = future_idx
                continue 
        i += 1
        
    wins = [t for t in trade_log if t['Pct'] > 0]
    win_rate = len(wins) / len(trade_log) if trade_log else 0
    
    print(f"\n--- [PHASE 3 SNIPER RESULTS] ---")
    print(f"Final Capital: ${capital:,.2f} | Total Return: {((capital - 10000)/10000)*100:.2f}%")
    print(f"Trades: {len(trade_log)} | Win Rate: {win_rate*100:.2f}%")
    if trade_log:
        tp = len([t for t in trade_log if t['Reason'] == 'TP'])
        sl = len([t for t in trade_log if t['Reason'] == 'SL'])
        tm = len([t for t in trade_log if t['Reason'] == 'TIME'])
        print(f"Exits -> TP: {tp} | SL: {sl} | Time: {tm}")

if __name__ == "__main__":
    run_rl_backtest()
