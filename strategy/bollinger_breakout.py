from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class BollingerBreakoutStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.pattern_boost = config.get('pattern_boost', 0.2)
        self.candle_lookback = config.get('candle_lookback', 3)
        self.min_confidence = config.get('min_confidence', 65)

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(2, self.candle_lookback):
                return None

            last = features_df.iloc[-1]
            current_close = last["close"]
            upper = last["bb_upper"]
            lower = last["bb_lower"]
            ma = (upper + lower) / 2
            patterns = last.get("patterns", [])
            band_width = upper - lower
            band_pct = band_width / ma if ma > 0 else 0

            signal = None
            if current_close < lower and band_pct > 0.01:
                confidence = 70 + min(20, int((lower - current_close) / lower * 1000))
                signal = {
                    "signal": "up",
                    "price": current_close,
                    "confidence": confidence,
                    "band_width": band_width,
                    "distance": lower - current_close
                }
            elif current_close > upper and band_pct > 0.01:
                confidence = 75 + min(20, int((current_close - upper) / upper * 1000))
                signal = {
                    "signal": "down",
                    "price": current_close,
                    "confidence": confidence,
                    "band_width": band_width,
                    "distance": current_close - upper
                }
            if signal:
                signal = self._apply_pattern_boost(signal, patterns)
                signal.update({
                    "upper_band": upper,
                    "lower_band": lower,
                    "middle_band": ma,
                    "recommend_entry": (last["high"] + last["low"]) / 2,
                    "recommend_stop": ma,
                    "volume": last["volume"]
                })
                if signal["confidence"] >= self.min_confidence:
                    return signal
            return None
        except Exception as e:
            print(f"BollingerBreakout error: {e}")
            return None

    def _apply_pattern_boost(self, signal, patterns):
        if not signal or not patterns:
            return signal
        direction = signal["signal"]

        if direction == "up":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"] + CONFIG["candlestick_patterns"]["trend_up"]
        elif direction == "down":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"] + CONFIG["candlestick_patterns"]["trend_down"]
       else:
            confirm_patterns = []
        pattern_strength = sum(PATTERN_STRENGTH.get(p, 0.1) for p in patterns if p in confirm_patterns)
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            signal["confidence"] = min(95, signal["confidence"] + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal
