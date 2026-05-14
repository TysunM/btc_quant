import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import os

PROCESSED_DIR = '/btc_quant/data/processed/'
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'

def run_scanner():
    print("\n[SYSTEM] Loading test matrix...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'rl_smart_money_data.parquet'), engine='pyarrow')
    test_split = int(len(df) * 0.8)
    test_df = df.iloc[test_split:].reset_index(drop=True)
    
    features = test_df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date'], errors='ignore').values
    
    print("[SYSTEM] Scanning RL Agent conviction levels...")
    model = PPO.load(MODEL_PATH)
    
    all_actions = []
    for i in range(len(features)):
        obs = features[i]
        action, _ = model.predict(obs, deterministic=True)
        all_actions.append(abs(action[0]))
    
    max_conviction = max(all_actions)
    avg_conviction = sum(all_actions) / len(all_actions)
    
    print(f"\n--- [AGENT INTELLIGENCE REPORT] ---")
    print(f"Maximum Conviction Found: {max_conviction:.4f}")
    print(f"Average Conviction Found: {avg_conviction:.4f}")
    
    # Auto-adjust threshold to 80% of max conviction to see what it wants to trade
    suggested_threshold = max_conviction * 0.8
    print(f"Suggested Threshold for testing: {suggested_threshold:.4f}")

if __name__ == "__main__":
    run_scanner()
