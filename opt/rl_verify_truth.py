import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

PROCESSED_DIR = '/btc_quant/data/processed/'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_verify():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'rl_smart_money_data.parquet'), engine='pyarrow')
    df = df.dropna().reset_index()
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    
    test_df['H-L'] = test_df['High'] - test_df['Low']
    test_df['H-C'] = np.abs(test_df['High'] - test_df['Price'].shift(1))
    test_df['L-C'] = np.abs(test_df['Low'] - test_df['Price'].shift(1))
    test_df['TR'] = test_df[['H-L', 'H-C', 'L-C']].max(axis=1)
    atrs = test_df['TR'].rolling(14).mean().bfill().values
    
    # 14 Feature Columns exactly as trained
    features = test_df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date', 'index', 'H-L', 'H-C', 'L-C', 'TR'], errors='ignore').values
    prices, highs, lows = test_df['Price'].values, test_df['High'].values, test_df['Low'].values
    
    model = PPO.load(MODEL_PATH)
    
    balance = 10000.0
    reserve = 2000.0 # 20% Money Market Lock
    trade_log = []
    
    tp_mult, sl_mult, time_limit, fee = 4.0, 2.5, 18, 0.0012
    
    i = 0
    while i < len(test_df) - time_limit:
        obs = features[i]
        action, _ = model.predict(obs, deterministic=True)
        conf = abs(action[0])
        
        if conf >= 0.5: # Sniper trigger
            direction = 1 if action[0] > 0 else -1
            
            # Treasury Management: Only trade with 80% of funds, scaled by confidence
            tradable = balance - reserve
            pos_size_pct = (tradable * conf) / balance
            
            entry = prices[i]
            current_atr = atrs[i]
            tp = entry + (current_atr * tp_mult * direction)
            sl = entry - (current_atr * sl_mult * direction)
            
            trade_ret = 0.0
            for hold in range(1, time_limit + 1):
                f_idx = i + hold
                h, l, c = highs[f_idx], lows[f_idx], prices[f_idx]
                
                if (direction == 1 and l <= sl) or (direction == -1 and h >= sl):
                    trade_ret = (sl - entry) / entry * direction
                    break
                elif (direction == 1 and h >= tp) or (direction == -1 and l <= tp):
                    trade_ret = (tp - entry) / entry * direction
                    break
                elif hold == time_limit:
                    trade_ret = (c - entry) / entry * direction
                    break
            
            net_ret = (trade_ret - fee) * pos_size_pct
            balance += balance * net_ret
            trade_log.append({'Pct': net_ret, 'Conf': conf})
            i += hold
            continue
        i += 1
        
    wins = [t for t in trade_log if t['Pct'] > 0]
    wr = len(wins) / len(trade_log) if trade_log else 0
    
    print(f"\n--- [PHASE 4: TRUTH & TREASURY REPORT] ---")
    print(f"Final Balance: ${balance:,.2f} (Reserve of ${reserve} intact)")
    print(f"Total Trades: {len(trade_log)} | Win Rate: {wr*100:.2f}%")
    if trade_log:
        avg_conf = sum([t['Conf'] for t in trade_log]) / len(trade_log)
        print(f"Average Sniper Confidence: {avg_conf:.2%}")

if __name__ == "__main__":
    run_verify()
