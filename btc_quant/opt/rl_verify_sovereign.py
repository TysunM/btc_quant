import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

PROCESSED_DIR = '/btc_quant/data/processed/'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_verify():
    print("\n[SYSTEM] Loading Smart Money Matrix for Sovereign Validation...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'rl_smart_money_data.parquet'), engine='pyarrow')
    df = df.dropna().reset_index()
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    
    # 14 Feature Alignment
    features = test_df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date', 'index'], errors='ignore').values
    prices, highs, lows = test_df['Price'].values, test_df['High'].values, test_df['Low'].values
    macro_trend = test_df['Macro_Trend'].values
    
    # ATR for dynamic barriers
    test_df['H-L'] = test_df['High'] - test_df['Low']
    test_df['H-C'] = np.abs(test_df['High'] - test_df['Price'].shift(1))
    test_df['L-C'] = np.abs(test_df['Low'] - test_df['Price'].shift(1))
    atrs = test_df[['H-L', 'H-C', 'L-C']].max(axis=1).rolling(14).mean().bfill().values
    
    model = PPO.load(MODEL_PATH)
    
    # TREASURY PARAMS
    initial_cap = 10000.0
    reserve = 2000.0 # 20% Money Market Lock
    tp_mult, sl_mult = 8.0, 2.5
    min_hold, max_hold, fee = 18, 2190, 0.0012
    
    # Test across different confidence tiers to find the Alpha Peak
    for threshold in [0.45, 0.60, 0.75]:
        balance = initial_cap
        trade_log = []
        i = 0
        
        while i < len(test_df) - 500: # Buffer for end of data
            obs = features[i]
            action, _ = model.predict(obs, deterministic=True)
            conf = abs(action[0])
            
            if conf >= threshold:
                direction = 1 if action[0] > 0 else -1
                
                # Treasury Lock
                tradable = balance - reserve
                pos_size_pct = (tradable * conf) / balance
                
                entry, current_atr = prices[i], atrs[i]
                tp = entry + (current_atr * tp_mult * direction)
                sl = entry - (current_atr * sl_mult * direction)
                
                trade_ret = 0.0
                bars_held = 0
                
                for hold in range(1, max_hold + 1):
                    f_idx = i + hold
                    if f_idx >= len(test_df): break
                    h, l, c = highs[f_idx], lows[f_idx], prices[f_idx]
                    bars_held = hold
                    
                    if hold < min_hold:
                        continue
                        
                    # Physical Exits
                    if (direction == 1 and l <= sl) or (direction == -1 and h >= sl):
                        trade_ret = (sl - entry) / entry * direction
                        break
                    elif (direction == 1 and h >= tp) or (direction == -1 and l <= tp):
                        trade_ret = (tp - entry) / entry * direction
                        break
                        
                    # Macro Trend Exit
                    if macro_trend[f_idx] != direction:
                        trade_ret = (c - entry) / entry * direction
                        break
                
                net_ret = (trade_ret - fee) * pos_size_pct
                balance += balance * net_ret
                trade_log.append(net_ret)
                i += bars_held
                continue
            i += 1
        
        wr = len([t for t in trade_log if t > 0]) / len(trade_log) if trade_log else 0
        print(f"--- [Conviction Tier: {threshold}] ROI: {((balance-initial_cap)/initial_cap)*100:.2f}% | Trades: {len(trade_log)} | WinRate: {wr*100:.2f}%")

if __name__ == "__main__":
    run_verify()
