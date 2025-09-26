# data/dukascopy_data.py
import dayjs
import pandas as pd
from dukascopy_node import getHistoricalRates

class DukascopyClient:
    def fetch_candles(self, symbol, interval="1min", limit=300):
        try:
            now = pd.Timestamp.utcnow()
            from_time = now - pd.Timedelta(minutes=limit)
            tf_map = {
                "1s": "s1", "1min": "m1", "5min": "m5", "15min": "m15",
                "30min": "m30", "1h": "h1", "4h": "h4", "1day": "d1"
            }
            candles = getHistoricalRates({
                "instrument": symbol.lower(),
                "dates": {
                    "from": from_time.to_pydatetime(),
                    "to": now.to_pydatetime()
                },
                "timeframe": tf_map.get(interval, "m1"),
                "format": "json",
                "volumes": True
            })
            if not candles:
                return None
            return {
                "symbol": symbol,
                "interval": interval,
                "history": candles,
                "close": candles[-1]["close"]
            }
        except Exception as e:
            print(f"❌ DukascopyClient failed: {e}")
            return None
