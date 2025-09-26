from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class BollingerStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.min_confidence = config.get('min_confidence', 70)
        self.pattern_boost = config.get('pattern_boost', 0.2)
        self.candle_lookback = config.get('candle_lookback', 3)

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(2, self.candle_lookback):
                return None

            last = features_df.iloc[-1]
            current_close = last["close"]
            upper = last["bb_upper"]
            lower = last["bb_lower"]
            sma = (upper + lower) / 2
            patterns = last.get("patterns", [])
            band_width = upper - lower

            signal = None
            if current_close < lower:
                signal = {
                    "signal": "up",
                    "price": current_close,
                    "confidence": 75,
                    "distance_from_band": lower - current_close,
                    "band_width": band_width
                }
                signal = self._apply_pattern_boost(signal, patterns, "up")
            elif current_close > upper:
                signal = {
                    "signal": "down",
                    "price": current_close,
                    "confidence": 80,
                    "distance_from_band": current_close - upper,
                    "band_width": band_width
                }
                signal = self._apply_pattern_boost(signal, patterns, "down")
            if signal:
                signal.update({
                    "upper_band": upper,
                    "lower_band": lower,
                    "sma": sma,
                    "recommend_entry": current_close,
                    "recommend_stop": sma,
                    "volume": last["volume"]
                })
                if signal["confidence"] >= self.min_confidence:
                    return signal
            return None
        except Exception as e:
            print(f"BollingerStrategy error: {e}")
            return None

    def _apply_pattern_boost(self, signal, patterns, direction):
        if not patterns or not signal:
            return signal
        if direction == "up":
            relevant_patterns = CONFIG["candlestick_patterns"]["reversal_up"]
        elif direction == "down":
            relevant_patterns = CONFIG["candlestick_patterns"]["reversal_down"]
        else:
            relevant_patterns = []
        pattern_strength = sum(
            PATTERN_STRENGTH.get(p, 0) for p in patterns if p in relevant_patterns
        )
        if pattern_strength > 0:
            signal["confidence"] = min(
                95,
                signal.get("confidence", 70) + int(pattern_strength * 20 * self.pattern_boost)
            )
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal
