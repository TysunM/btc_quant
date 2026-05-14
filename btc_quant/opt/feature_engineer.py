import pandas as pd
import numpy as np
import os

INPUT_FILE = '/btc_quant/data/processed/quant_data_clean.parquet' 
OUTPUT_FILE = '/btc_quant/data/processed/rl_smart_money_data.parquet'

def build_v7_matrix():
    print("\n[SYSTEM] Restoring Old Brain Core... Injecting 15th Channel (FVG Structure).")
    df = pd.read_parquet(INPUT_FILE)

    if 'Vol' in df.columns: df = df.rename(columns={'Vol': 'Volume'})
    if 'Price' in df.columns: df = df.rename(columns={'Price': 'Close'})
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df = df.resample('4h', label='right', closed='right').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    # --- THE ORIGINAL 14 CORE ---
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0.0)
    
    # ATR & TR
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Volume Dynamics
    df['Volume_Z'] = (df['Volume'] - df['Volume'].rolling(50).mean()) / (df['Volume'].rolling(50).std() + 1e-10)
    
    # Momentum (Old Brain RSI + DM)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI_14'] = 100 - (100 / (1 + (gain / (loss + 1e-10))))
    df['+DM'] = df['High'].diff().clip(lower=0).rolling(14).mean()
    df['-DM'] = (-df['Low'].diff()).clip(lower=0).rolling(14).mean()
    
    # Liquidity & Macro
    df['VWAP'] = (((df['High'] + df['Low'] + df['Close']) / 3) * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Dev'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['Macro_Trend'] = np.where(df['Close'] > df['Close'].ewm(span=200, adjust=False).mean(), 1, -1)
    
    # Geometry
    roll_max, roll_min = df['High'].rolling(21).max(), df['Low'].rolling(21).min()
    swing = roll_max - roll_min + 1e-10
    df['Fib_618'] = (df['Close'] - (roll_max - (swing * 0.618))) / swing
    df['Fib_382'] = (df['Close'] - (roll_max - (swing * 0.382))) / swing
    df['Fib_236'] = (df['Close'] - (roll_max - (swing * 0.236))) / swing
    
    # Oscillators
    df['EMA_Spread'] = (df['Close'].ewm(span=10).mean() - df['Close'].ewm(span=30).mean()) / df['Close']
    df['Momentum_ROC'] = df['Close'].pct_change(10)

    # --- THE 15TH CHANNEL: FAIR VALUE GAP (FVG) ---
    # Unifying Bull and Bear FVG into a single structural signal
    bull_fvg = np.where(df['Low'] > df['High'].shift(2), 1, 0)
    bear_fvg = np.where(df['High'] < df['Low'].shift(2), -1, 0)
    df['FVG_Signal'] = bull_fvg + bear_fvg

    df = df.dropna()

    feature_cols = [
        'Log_Returns', 'ATR', 'TR', 'Volume_Z', 'RSI_14', '+DM', '-DM', 
        'VWAP_Dev', 'Macro_Trend', 'Fib_618', 'Fib_382', 'Fib_236', 
        'EMA_Spread', 'Momentum_ROC', 'FVG_Signal'
    ]
    
    df = df.rename(columns={'Close': 'Price'}).reset_index()
    final_cols = ['Date', 'Open', 'High', 'Low', 'Price'] + feature_cols
    df[final_cols].to_parquet(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] 15-Channel SMC Matrix Forged. Shape: {df[feature_cols].shape}")

if __name__ == "__main__":
    build_v7_matrix()
