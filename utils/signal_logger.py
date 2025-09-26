#signal_logger.py
import csv
import os
from datetime import datetime

SIGNAL_LOG_PATH = "signals.csv"

def log_signal(chat_id, symbol, timeframe, signal_data):
    fields = [
        "timestamp", "user", "symbol", "timeframe",
        "signal", "confidence", "strength",
        "price", "recommend_entry", "high", "low", "volume"
    ]

    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": chat_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal_data.get("signal"),
        "confidence": signal_data.get("confidence"),
        "strength": signal_data.get("strength"),
        "price": signal_data.get("price"),
        "recommend_entry": signal_data.get("recommend_entry"),
        "high": signal_data.get("high"),
        "low": signal_data.get("low"),
        "volume": signal_data.get("volume")
    }

    file_exists = os.path.isfile(SIGNAL_LOG_PATH)
    with open(SIGNAL_LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    print("📦 Signal logged to CSV.")
      
