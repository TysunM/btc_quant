import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

PROCESSED_DIR = '/btc_quant/data/processed/'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_verify():
    print("\n[SYSTEM] Loading Smart Money Matrix for Surgical Verification...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'rl_smart_money_data.parquet'), engine='pyarrow')
    
    # Mirror the training script's indexing exactly
    df = df.dropna().reset_index()
    
    # Test on the last 20% of data (Out of Sample)
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    
    # Calculate ATR for the exit barriers only (not as a feature)
    test_df['H-L'] = test_df['High'] - test_df['Low']
    test_df['H-C'] = np.abs(test_df['High'] - test_df['Price'].shift(1))
    test_df['L-C'] = np.abs(test_df['Low'] - test_df['Price'].shift(1))
    test_df['TR'] = test_df[['H-L', 'H-C', 'L-C']].max(axis=1)
    atrs = test_df['TR'].rolling(14).mean().bfill().values
    
    # --- PERFECT FEATURE ALIGNMENT (14 COLUMNS) ---
    # We drop the exact same columns as the training script
    features_df = test_df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date', 'index', 'H-L', 'H-C', 'L-C', 'TR'], errors='ignore')
    features = features_df.values
    
    print(f"[SYSTEM] Features aligned: {features.shape[1]} columns detected.")
    
    prices = test_df['Price'].values
    highs = test_df['High'].values
    lows = test_df['Low'].values
    
    print("[SYSTEM] Waking up Titanium RL Agent...")
    model = PPO.load(MODEL_PATH)
    
    capital = 10000.0
    position = 0  
    entry_price = 0.0
    bars_held = 0
    trade_log = []
    
    tp_mult, sl_mult, time_limit, fee = 4.0, 2.0, 18, 0.0012
    SNIPER_THRESHOLD = 0.50 
    
    i = 0
    while i < len(test_df) - time_limit:
        if position == 0:
            obs = features[i]
            action, _ = model.predict(obs, deterministic=True)
            action_value = action[0]
            confidence = abs(action_value)
            
            if confidence >= SNIPER_THRESHOLD:
                position = 1 if action_value > 0 else -1
                position_size = confidence 
                entry_price = prices[i]
                current_atr = atrs[i]
                
                take_profit = entry_price + (current_atr * tp_mult * position)
                stop_loss = entry_price - (current_atr * sl_mult * position)
                bars_held = 0
                
        if position != 0:
            bars_held += 1
            idx = i + bars_held
            h, l, c = highs[idx], lows[idx], prices[idx]
            
            trade_over, trade_return, reason = False, 0.0, ""
            
            if (position == 1 and l <= stop_loss) or (position == -1 and h >= stop_loss):
                trade_return, trade_over, reason = (stop_loss - entry_price) / entry_price * position, True, "SL"
            elif (position == 1 and h >= take_profit) or (position == -1 and l <= take_profit):
                trade_return, trade_over, reason = (take_profit - entry_price) / entry_price * position, True, "TP"
            elif bars_held >= time_limit:
                trade_return, trade_over, reason = (c - entry_price) / entry_price * position, True, "TIME"
                
            if trade_over:
                net_ret = (trade_return - fee) * position_size
                capital += capital * net_ret
                trade_log.append({'Pct': net_ret, 'Reason': reason, 'Conf': confidence})
                position, i = 0, idx
                continue 
        i += 1
        
    wins = [t for t in trade_log if t['Pct'] > 0]
    win_rate = len(wins) / len(trade_log) if trade_log else 0
    elite_trades = [t for t in trade_log if t['Conf'] >= 0.75]
    
    print(f"\n--- [TITANIUM SURGICAL SNIPER: 400K VERIFICATION] ---")
    print(f"Final Balance: ${capital:,.2f} | Total ROI: {((capital - 10000)/10000)*100:.2f}%")
    print(f"Total Trades: {len(trade_log)} | Win Rate: {win_rate*100:.2f}%")
    print(f"Elite Trades (>= 75% Conviction): {len(elite_trades)}")
    
    if trade_log:
        tp = len([t for t in trade_log if t['Reason'] == 'TP'])
        sl = len([t for t in trade_log if t['Reason'] == 'SL'])
        print(f"Outcome Metrics -> Take Profits: {tp} | Stop Losses: {sl}")

if __name__ == "__main__":
    run_verify()
