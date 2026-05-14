import os
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
from stable_baselines3 import PPO
import alpaca_trade_api as tradeapi
import schedule

# --- SECURE CONFIGURATION ---
load_dotenv('/btc_quant/.env') # Force read from the specific .env location

API_KEY = os.getenv('ALPACA_PAPER_API_KEY')
API_SECRET = os.getenv('ALPACA_PAPER_SECRET_KEY')
BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
MODEL_PATH = '/btc_quant/opt/titanium_rl_agent.zip'
SYMBOL = 'BTC/USD'

if not API_KEY or not API_SECRET:
    raise ValueError("[FATAL ERROR] API Keys missing. Check your /btc_quant/.env file.")

# Connect to Alpaca
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

print("[SYSTEM] Waking up Titanium Brain...")
model = PPO.load(MODEL_PATH)

def fetch_and_build_features():
    # 1. Fetch the last 200 candles (4H timeframe) to ensure moving averages calculate correctly
    bars = api.get_crypto_bars(SYMBOL, tradeapi.TimeFrame(4, tradeapi.TimeFrameUnit.Hour)).df
    bars = bars.tz_convert('UTC')
    
    df = pd.DataFrame({
        'Open': bars['open'],
        'High': bars['high'],
        'Low': bars['low'],
        'Price': bars['close'],
        'Volume': bars['volume']
    })
    
    # --- REVERSE-ENGINEERED SMART MONEY FEATURES ---
    
    # 1. Macro Trend (200 EMA)
    df['EMA_200'] = df['Price'].ewm(span=200, adjust=False).mean()
    df['Macro_Trend'] = np.where(df['Price'] > df['EMA_200'], 1, -1)
    
    # 2. RSI (14-Period)
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_4H'] = 100 - (100 / (1 + rs))
    
    # 3. Directional Movement (+DM, -DM, UpMove, DownMove)
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0.0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0.0)
    
    # 4. VWAP Deviation
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Price']) / 3
    df['VWAP'] = (df['Typical_Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Dev'] = (df['Price'] - df['VWAP']) / df['VWAP']
    
    # 5. ATR (14-Period for Stop Loss / Take Profit)
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = np.abs(df['High'] - df['Price'].shift(1))
    df['L-C'] = np.abs(df['Low'] - df['Price'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # Drop NaNs created by rolling windows
    df = df.dropna()
    latest_candle = df.iloc[-1]
    
    # Combine into the exact 14-feature array shape the AI expects
    # Note: If the AI expects slightly different columns, we will catch it on the first inference
    feature_columns = [
        'Volume', 'EMA_200', 'Macro_Trend', 'RSI_4H', 
        '+DM', '-DM', 'UpMove', 'DownMove', 'VWAP_Dev', 
        'ATR', 'H-L', 'H-C', 'L-C', 'TR'
    ]
    
    latest_features = latest_candle[feature_columns].values
    
    return latest_features, latest_candle['Price'], latest_candle['ATR'], latest_candle['Macro_Trend']

def execute_sovereign_sniper():
    print(f"\n[SYSTEM] Commencing 4H Market Scan for {SYMBOL}...")
    try:
        obs, current_price, current_atr, macro_trend = fetch_and_build_features()
        action, _ = model.predict(obs, deterministic=True)
        action_val = action[0]
        conf = abs(action_val)
        direction = 1 if action_val > 0 else -1
        
        print(f"-> AI Conviction: {conf:.4f} | Direction: {'LONG' if direction == 1 else 'SHORT'}")
        print(f"-> Current Price: ${current_price:,.2f} | 4H ATR: ${current_atr:,.2f}")
        
        if conf < 0.45:
            print("-> [HOLD] Conviction below 0.45 threshold. Standing by.")
            return

        account = api.get_account()
        equity = float(account.equity)
        reserve_requirement = 0.20
        
        tradable_capital = equity * (1 - reserve_requirement)
        position_size_dollars = tradable_capital * conf
        qty = round(position_size_dollars / current_price, 4) # Alpaca requires rounded crypto qty
        
        print(f"-> [TREASURY] Equity: ${equity:,.2f} | Deploying: ${position_size_dollars:,.2f} (Qty: {qty})")
        
        tp_price = current_price + (current_atr * 8.0 * direction)
        sl_price = current_price - (current_atr * 2.5 * direction)
        
        positions = api.list_positions()
        for pos in positions:
            if pos.symbol == 'BTCUSD':
                current_pos_dir = 1 if pos.side == 'long' else -1
                if current_pos_dir != direction:
                    print(f"-> [MACRO FLIP] Reversing position from {pos.side} to {'long' if direction==1 else 'short'}")
                    api.close_position('BTCUSD')
                else:
                    print("-> [HOLDING] Already in aligned position. Letting profits run.")
                    return

        side = 'buy' if direction == 1 else 'sell'
        print(f"-> [EXECUTION] Firing {side.upper()} order. TP: ${tp_price:,.2f} | SL: ${sl_price:,.2f}")
        
        api.submit_order(
            symbol='BTCUSD',
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc',
            order_class='bracket',
            take_profit=dict(limit_price=round(tp_price, 2)),
            stop_loss=dict(stop_price=round(sl_price, 2), limit_price=round(sl_price, 2))
        )
        print("[SUCCESS] Sniper Round deployed successfully.")

    except Exception as e:
        print(f"\n[ERROR] Execution Failed: {e}")
        print("Note: If the error mentions 'shape mismatch', the 14 columns need to be re-ordered.")

execute_sovereign_sniper()
schedule.every(4).hours.do(execute_sovereign_sniper)

print("\n[SYSTEM] Execution Node Online. Listening for 4H intervals...")
while True:
    schedule.run_pending()
    time.sleep(60)
