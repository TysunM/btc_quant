import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

PROCESSED_DIR = '/btc_quant/data/processed/'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_scanner():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'rl_smart_money_data.parquet'), engine='pyarrow')
    df = df.dropna().reset_index()
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    
    # 14 Feature Alignment
    features = test_df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date', 'index', 'H-L', 'H-C', 'L-C', 'TR'], errors='ignore').values
    
    model = PPO.load(MODEL_PATH)
    
    print("[SYSTEM] Scanning hardened agent for its 'Truth Ceiling'...")
    all_conf = []
    for i in range(len(features)):
        obs = features[i]
        action, _ = model.predict(obs, deterministic=True)
        all_conf.append(abs(action[0]))
    
    max_c = max(all_conf)
    avg_c = sum(all_conf) / len(all_conf)
    
    print(f"\n--- [PHASE 4 INTELLIGENCE REPORT] ---")
    print(f"Hardened Max Conviction: {max_c:.4f}")
    print(f"Hardened Avg Conviction: {avg_c:.4f}")
    
    # Calculate a threshold that would allow at least some trades
    suggested = max_c * 0.9
    print(f"Suggested 'Truth Threshold': {suggested:.4f}")

if __name__ == "__main__":
    run_scanner()
