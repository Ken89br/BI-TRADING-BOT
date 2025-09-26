# strategy/wash_signals.py

import pandas as pd
import os

SIGNALS_PATH = "signals.csv"
OUTPUT_PATH = "training_data.csv"

def is_good_signal(row):
    # AI-style filtering logic
    try:
        if row["confidence"] < 60:
            return False
        if row["strength"].lower() == "weak":
            return False
        if float(row["high"]) - float(row["low"]) < 0.0001:
            return False
        if float(row["volume"]) < 1:
            return False
        return True
    except Exception as e:
        print(f"⚠️ Skipping row due to error: {e}")
        return False

def wash_signals():
    if not os.path.exists(SIGNALS_PATH):
        print("⚠️ No signals.csv file found.")
        return

    df = pd.read_csv(SIGNALS_PATH)
    print(f"📥 Loaded {len(df)} signals.")

    df_cleaned = df[df.apply(is_good_signal, axis=1)]

    if df_cleaned.empty:
        print("⚠️ No valid signals after filtering.")
        return

    df_cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Cleaned data saved to {OUTPUT_PATH} ({len(df_cleaned)} rows)")

if __name__ == "__main__":
    wash_signals()
