
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

        # Done Logic: 1. Liquidation floor  2. End of data  3. Life cycle r>
        done = (self.ep_balance <= 2100.0 or
                self.current_step >= len(self.df) - 1 or
                self.ep_length >= 5000)

        return self._get_obs(), reward, done, False, {}

# ====================== MASTER TRAINING SESSION ======================
if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Signal data not found at {DATA_PATH}")
    elif not os.path.exists(GENRE_MODEL_PATH):
        print(f"[ERROR] Genre model not found. Run scenario_clusterer.py fi>
    else:
        df = pd.read_parquet(DATA_PATH)
        env = DummyVecEnv([lambda: BTCSovereignEnv(df)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        model = PPO("MlpPolicy", env, verbose=0,
                    learning_rate=LEARNING_RATE,
                    n_steps=N_STEPS,
                    batch_size=BATCH_SIZE)

        model.set_logger(configure('./logs/', ["tensorboard", "csv"]))

        print(f"\n{G}[SYSTEM] FINAL RENDER INITIATED. 3M Step Render Active>
        print(f"{G}[SYSTEM] GROUNDED ROI & SCENARIO-AWARENESS ENGAGED.{W}")
        model.learn(total_timesteps=TOTAL_STEPS, callback=SovereignGrounded>

        model.save("/btc_quant/opt/titanium_final_agent")
        print(f"\n{G}[SUCCESS] Sovereign Protocol Rendered. Model Saved.{W}>
