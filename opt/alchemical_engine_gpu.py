import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import joblib
import os

# ====================== HARMONIC CONFIG (3-6-9) ======================
GENRE_MODEL_PATH = '/btc_quant/opt/market_genres.pkl'
DATA_PATH = '/btc_quant/data/processed/rl_smart_money_data.parquet'
INITIAL_CAPITAL = 10000
TOTAL_STEPS = 3000000
LOOKBACK = 66           # Harmonized window
LEARNING_RATE = 6e-5    # The "6" harmonic
N_STEPS = 2016          # SB3 rollout buffer
BATCH_SIZE = 126        # Harmonic of 3
COMMISSION = 0.0005     # 5 bps institutional fee
SLIPPAGE = 0.0002       # 2 bps slippage buffer

# --- ANSI VISUALS ---
C, W, G, R, Y = "\033[96m", "\033[0m", "\033[92m", "\033[91m", "\033[93m"

# ====================== VERTICAL TELEMETRY BOX ======================
class SovereignGroundedCallback(BaseCallback):
    """Vertical Scannable Telemetry: No Resets, No Hallucinations."""
    def _on_step(self) -> bool:
        if self.n_calls % 1200 == 0:
            env = self.training_env.envs[0].unwrapped
            initial = env.initial_capital
            final = env.global_balance
            roi = ((final - initial) / initial) * 100
            
            # ep_rew_mean will move now due to 5,000-step force-resets
            ep_rew = self.model.logger.name_to_value.get('rollout/ep_rew_mean', 0.0)
            iteration = self.num_timesteps // N_STEPS
            
            # Accurate Time Sync: 6 bars = 1 day (4H timeframe)
            total_days = env.global_steps // 6
            y, m, d = total_days // 365, (total_days % 365) // 30, (total_days % 30)

            print(f"\n{C}=================================================={W}")
            print(f" Initial Capital           ${initial:,.0f}")
            print(f"--------------------------------------------------")
            print(f" ROI                       {G if roi >=0 else R}{roi:>+7.2f}%{W}")
            print(f" ep_rew_mean_              {Y}{ep_rew:>7.2f}{W}") 
            print(f" Iteration                 {iteration}")
            print(f" Step count                {self.num_timesteps}")
            print(f" Current Genre             {C}GENRE_{env.current_genre}{W}")
            print(f"")
            print(f" total trades              {env.global_trades}")
            print(f" profit (wins)             {env.global_wins}")
            print(f" Losses                    {env.global_losses}")
            print(f" win %                     {(env.global_wins/max(1, env.global_trades))*100:.2f}%")
            print(f" total time elapsed        {y}y, {m}m, {d}d")
            print(f"--------------------------------------------------")
            print(f" {G if final >= initial else R}Final Capital             ${final:,.2f}{W}")
            print(f"{C}=================================================={W}\n")
        return True

# ====================== SOVEREIGN ENVIRONMENT ======================
class BTCSovereignEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.dropna().reset_index(drop=True)
        # 15 channels (Price + Indicators + EMF Frequency)
        self.features = self.df.drop(columns=['Open', 'High', 'Low', 'Price', 'Date'], errors='ignore').values
        
        # Scenario Library: Loads K-Means for real-time Genre detection
        genre_data = joblib.load(GENRE_MODEL_PATH)
        self.genre_model = genre_data['model']
        self.genre_scaler = genre_data['scaler']
        
        self.action_space = spaces.Discrete(3) # 0=Hold, 1=Long, 2=Short
        
        # Obs = 66 bars of (15 features + Pos + PnL + GenreID) = 18 total channels
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(LOOKBACK, self.features.shape[1] + 3), dtype=np.float32
        )
        
        # Global Master Tape (Never Resets)
        self.initial_capital = INITIAL_CAPITAL
        self.global_balance = INITIAL_CAPITAL
        self.global_trades = 0
        self.global_wins = 0
        self.global_losses = 0
        self.global_steps = 0
        self.current_genre = 0
        
        self.reset()

    def _get_obs(self):
        start = max(0, self.current_step - LOOKBACK)
        raw_window = self.features[start:self.current_step]
        if len(raw_window) < LOOKBACK:
            raw_window = np.vstack([np.zeros((LOOKBACK - len(raw_window), self.features.shape[1])), raw_window])
        
        # ID Current Market Genre
        scaled_window = self.genre_scaler.transform(raw_window)
        self.current_genre = self.genre_model.predict(scaled_window.flatten().reshape(1, -1))[0]
        
        # Internal State Features
        pos_feat = np.full((LOOKBACK, 1), self.position)
        pnl_feat = np.full((LOOKBACK, 1), (self.ep_balance - self.initial_capital) / self.initial_capital)
        genre_feat = np.full((LOOKBACK, 1), self.current_genre)
        
        return np.hstack([raw_window, pos_feat, pnl_feat, genre_feat]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Randomized starts for deep history training
        self.current_step = np.random.randint(LOOKBACK, len(self.df) - 5001)
        self.ep_balance = 10000.0
        self.ep_length = 0
        self.position = 0 
        self.entry_price = 0
        return self._get_obs(), {}

    def step(self, action):
        price = self.df.iloc[self.current_step]['Price']
        reward = 0
        
        # 1. Position Management & "Grounded" Stop Loss
        if self.position != 0:
            pnl = (price / self.entry_price - 1) * self.position
            # Hard 1% Stop Loss as per Brun & Pereira 2025
            if pnl <= -0.01 or (action == 0) or (action == 1 and self.position == -1) or (action == 2 and self.position == 1):
                net_pnl = pnl - COMMISSION - SLIPPAGE
                
                # Fixed $10k Block impact to prevent runaway hallucinations
                impact = 10000 * net_pnl
                self.global_balance += impact
                self.ep_balance += impact
                
                if net_pnl > 0: self.global_wins += 1
                else: self.global_losses += 1
                
                reward = net_pnl * 100
                self.position = 0
                self.global_trades += 1

        # 2. Entry Logic
        if action == 1 and self.position == 0: # LONG
            self.position = 1
            self.entry_price = price * (1 + SLIPPAGE)
        elif action == 2 and self.position == 0: # SHORT
            self.position = -1
            self.entry_price = price * (1 - SLIPPAGE)

        self.current_step += 1
        self.global_steps += 1
        self.ep_length += 1
        
        # Done Logic: 1. Liquidation floor  2. End of data  3. Life cycle reset (5000 bars)
        done = (self.ep_balance <= 2100.0 or 
                self.current_step >= len(self.df) - 1 or 
                self.ep_length >= 5000)
        
        return self._get_obs(), reward, done, False, {}

# ====================== MASTER TRAINING SESSION ======================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Signal data not found at {DATA_PATH}")
    elif not os.path.exists(GENRE_MODEL_PATH):
        print(f"[ERROR] Genre model not found. Run scenario_clusterer.py first.")
    else:
        df = pd.read_parquet(DATA_PATH)
        env = DummyVecEnv([lambda: BTCSovereignEnv(df)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        model = PPO("MlpPolicy", env, verbose=0, 
                    learning_rate=LEARNING_RATE, 
                    n_steps=N_STEPS, 
                    batch_size=BATCH_SIZE)
        
        model.set_logger(configure('./logs/', ["tensorboard", "csv"]))

        print(f"\n{G}[SYSTEM] FINAL RENDER INITIATED. 3M Step Render Active.{W}")
        print(f"{G}[SYSTEM] GROUNDED ROI & SCENARIO-AWARENESS ENGAGED.{W}")
        model.learn(total_timesteps=TOTAL_STEPS, callback=SovereignGroundedCallback())
        
        model.save("/btc_quant/opt/titanium_final_agent")
        print(f"\n{G}[SUCCESS] Sovereign Protocol Rendered. Model Saved.{W}")
