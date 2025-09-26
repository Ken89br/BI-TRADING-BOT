import numpy as np
from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class RSIStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.overbought = config.get('overbought', 70)
        self.oversold = config.get('oversold', 30)
        self.min_confidence = config.get('min_confidence', 65)
        self.pattern_boost = config.get('pattern_boost', 0.2)
        self.candle_lookback = config.get('candle_lookback', 3)
        self.volume_threshold = config.get('volume_threshold', 1.5)

    def generate_signal(self, features_df):
        """
        Estratégia RSI compatível com o DataFrame universal de features.
        """
        try:
            if features_df is None or features_df.empty or len(features_df) < max(2, self.candle_lookback):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            rsi = last["rsi_value"]
            volume = last["volume"]
            patterns = last.get("patterns", [])
            pattern_strength = last.get("pattern_strength", 0)
            close = last["close"]

            if self.candle_lookback > 1:
                prev_vols = features_df.iloc[-self.candle_lookback:-1]["volume"]
                avg_prev_vol = prev_vols.mean() if not prev_vols.empty else 0
            else:
                avg_prev_vol = prev["volume"]

            volume_ok = volume > avg_prev_vol * self.volume_threshold if avg_prev_vol else False

            signal = None
            if rsi < self.oversold:
                signal = {
                    "signal": "up",
                    "rsi": rsi,
                    "confidence": 60 + min(30, (self.oversold - rsi) / 2),
                    "price": close,
                    "volume_ok": volume_ok
                }
            elif rsi > self.overbought:
                signal = {
                    "signal": "down",
                    "rsi": rsi,
                    "confidence": 65 + min(30, (rsi - self.overbought) / 2),
                    "price": close,
                    "volume_ok": volume_ok
                }

            # Aplica boost por padrões de vela
            if signal and patterns:
                direction = signal["signal"]
                if direction == "up":
                    confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"]
                elif direction == "down":
                    confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"]
                else:
                    confirm_patterns = []
                boost = sum(
                    PATTERN_STRENGTH.get(p, 0)
                    for p in patterns if p in confirm_patterns
                )
                if boost > 0:
                    signal["confidence"] = min(95, signal["confidence"] + int(boost * 15 * self.pattern_boost))
                    signal["patterns"] = patterns
                    signal["pattern_strength"] = pattern_strength

            if signal and signal.get("confidence", 0) >= self.min_confidence:
                return signal
            return None
        except Exception as e:
            print(f"RSIStrategy error: {e}")
            return None
