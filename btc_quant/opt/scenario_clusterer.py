import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import joblib  # Fixed import
import os

# --- CONFIG ---
LOOKBACK = 66
INPUT_FILE = '/btc_quant/data/processed/rl_smart_money_data.parquet'
GENRE_MODEL_PATH = '/btc_quant/opt/market_genres.pkl'

def forge_scenarios():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    print("[SYSTEM] Loading 15-Channel Matrix for Scenario Analysis...")
    df = pd.read_parquet(INPUT_FILE)
    
    # Filter to only the features used for decision making
    feature_cols = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Price']]
    data = df[feature_cols].values
    
    # 1. Scaling (Essential for distance-based clustering)
    print("[INFO] Normalizing signal for pattern recognition...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 2. Windowing (3D 'Shapes')
    print(f"[INFO] Sculpting {LOOKBACK}-bar temporal windows...")
    windows = []
    # Sampling every 10th window to fit in RAM while maintaining the 'rhythm'
    for i in range(0, len(scaled_data) - LOOKBACK, 10):
        windows.append(scaled_data[i : i + LOOKBACK].flatten())
    
    window_matrix = np.array(windows)
    print(f"[INFO] Window matrix forged: {window_matrix.shape}")
    
    # 3. K-Means 'MiniBatch' (Institutional Speed)
    # 6 Clusters = 2 Trend types, 2 Range types, 1 Breakout, 1 Noise
    print("[SYSTEM] Engaging MiniBatch K-Means. Identifying 6 Market Genres...")
    kmeans = MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=2048, n_init=3)
    kmeans.fit(window_matrix)
    
    # 4. Export the Genre Model
    os.makedirs(os.path.dirname(GENRE_MODEL_PATH), exist_ok=True)
    joblib.dump({'model': kmeans, 'scaler': scaler}, GENRE_MODEL_PATH)
    
    print(f"\n[SUCCESS] Market Genre Library Forged: {GENRE_MODEL_PATH}")
    print(f"   -> Cluster Centers: {kmeans.cluster_centers_.shape}")
    print(f"   -> READY FOR TRAINING INTEGRATION.")

if __name__ == "__main__":
    forge_scenarios()
