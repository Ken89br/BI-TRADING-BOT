from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class ATRStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.atr_period = config.get('atr_period', 14)
        self.multiplier = config.get('multiplier', 1.2)
        self.require_volume = config.get('require_volume', True)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.min_confidence = config.get('min_confidence', 65)
        self.candle_lookback = config.get('candle_lookback', 3)
        self.pattern_boost = config.get('pattern_boost', 0.2)

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(self.atr_period+1, self.candle_lookback):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            atr = last["atr_value"]
            body_size = abs(last["close"] - last["open"])
            is_bullish = last["close"] > last["open"]

            volume = last["volume"]
            if self.atr_period > 1:
                prev_vols = features_df.iloc[-self.atr_period:-1]["volume"]
                avg_volume = prev_vols.mean() if not prev_vols.empty else 0
            else:
                avg_volume = prev["volume"]
            volume_ok = not self.require_volume or (volume > avg_volume * self.volume_threshold if avg_volume else False)

            patterns = last.get("patterns", [])
            signal = None

            if body_size > atr * self.multiplier:
                direction = "up" if is_bullish else "down"
                base_confidence = 70 if direction == "up" else 75
                size_factor = min(20, (body_size / atr - 1) * 10)
                volume_factor = 10 if volume_ok else 0

                signal = {
                    "signal": direction,
                    "atr": round(atr, 5),
                    "body_ratio": round(body_size / atr, 2),
                    "confidence": min(95, base_confidence + size_factor + volume_factor),
                    "volume_ok": volume_ok,
                    "price": last["close"]
                }

                signal = self._apply_pattern_boost(signal, patterns, direction)

                if direction == "up":
                    signal.update({
                        "recommend_entry": last["close"],
                        "recommend_stop": last["low"] - atr * 0.5
                    })
                else:
                    signal.update({
                        "recommend_entry": last["close"],
                        "recommend_stop": last["high"] + atr * 0.5
                    })

            return signal if (signal and signal["confidence"] >= self.min_confidence) else None

        except Exception as e:
            print(f"ATRStrategy error: {e}")
            return None

    def _apply_pattern_boost(self, signal, patterns, direction):
        if not patterns:
            return signal
        if direction == "up":
            relevant_patterns = CONFIG["candlestick_patterns"]["reversal_up"]
        elif direction == "down":
            relevant_patterns = CONFIG["candlestick_patterns"]["reversal_down"]
        else:
            relevant_patterns = []
        pattern_strength = sum(
            PATTERN_STRENGTH.get(p, 0)
            for p in patterns
            if p in relevant_patterns
        )
        if pattern_strength > 0:
            signal["confidence"] = min(
                95,
                signal.get("confidence", 65) + int(pattern_strength * 20 * self.pattern_boost)
            )
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal
