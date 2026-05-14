import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

# --- PATHS ---
DATA_PATH = '/btc_quant/data/processed/rl_smart_money_data.parquet'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_master_analysis():
    print("[SYSTEM] Loading Master Render for Playback Analysis...")
    try:
        df = pd.read_parquet(DATA_PATH)
    except FileNotFoundError:
        print(f"[CRITICAL] Missing V7 Matrix at {DATA_PATH}")
        return
        
    feature_cols = ['Volume', 'EMA_200', 'Macro_Trend', 'RSI_4H', '+DM', '-DM', 'UpMove', 'DownMove', 'VWAP_Dev', 'ATR', 'H-L', 'H-C', 'L-C', 'TR']
    X = df[feature_cols].values
    prices = df['Price'].values
    
    print("[SYSTEM] Loading Titanium Brain V7...")
    model = PPO.load(MODEL_PATH)
    
    capital = 10000.0
    initial_capital = 10000.0
    equity_curve = [capital]
    trade_log = []
    
    print("[SYSTEM] Bouncing track... checking phase and dynamic range...")
    
    for i in range(len(df) - 1):
        obs = X[i].reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        ai_signal = action[0]
        
        # 1. Recreate the EQ/Confluence Curve
        score = 0
        rsi = df.iloc[i]['RSI_4H']
        macro = df.iloc[i]['Macro_Trend']
        vwap = df.iloc[i]['VWAP_Dev']
        
        if 30 <= rsi <= 45 or 55 <= rsi <= 70: score += 25
        if macro == 1: score += 25
        if -0.02 <= vwap <= 0.01: score += 15
        score += 35 # Core tactical/liquidity base
        
        if macro == 1 and (55 <= rsi <= 68): score += 5.0
        elif rsi > 75 or rsi < 35: score -= 5.0
            
        wcs = min(max(score, 0), 100)
        
        # 2. The 72% Noise Gate
        if ai_signal > 0.15 and wcs >= 72.0:
            # 80/20 Headroom scaling
            alloc = (capital * 0.80) * (wcs / 100)
            ret = (prices[i+1] - prices[i]) / prices[i]
            net_ret = ret - 0.0012 # Slippage/Fee
            
            capital += alloc * net_ret
            trade_log.append(net_ret)
            
        equity_curve.append(capital)
        
    eq_series = pd.Series(equity_curve)
    pct_change = eq_series.pct_change().dropna()
    
    # 2190 4H bars in a year
    sharpe = (pct_change.mean() / pct_change.std()) * np.sqrt(2190) if pct_change.std() != 0 else 0
    
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    print("\n--- [MASTERING CHAIN RESULTS] ---")
    print(f"Total Trades Printed: {len(trade_log)}")
    print(f"Ending Capital:       ${capital:,.2f}")
    print(f"Total Return:         {((capital - initial_capital)/initial_capital)*100:.2f}%")
    print(f"Max Drawdown:         {max_dd*100:.2f}%")
    print(f"Sharpe Ratio:         {sharpe:.2f}")

if __name__ == "__main__":
    run_master_analysis()
