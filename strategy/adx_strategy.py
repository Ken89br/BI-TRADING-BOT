from strategy.config_strategy import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class ADXStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.adx_threshold = config.get('adx_threshold', 25)
        self.candle_lookback = config.get('candle_lookback', 3)
        self.pattern_boost = config.get('pattern_boost', 0.2)
        self.min_confidence = config.get('min_confidence', 70)
        self.volume_threshold = config.get('volume_threshold', 1.3)
        self.require_trend_confirmation = config.get('trend_confirmation', True)

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(3, self.candle_lookback):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            adx = last["adx_value"]
            plus_di = last["adx_di_plus"]
            minus_di = last["adx_di_minus"]
            patterns = last.get("patterns", [])
            volume_ok = last.get("volume", 0) > prev.get("volume", 0) * self.volume_threshold
            trend_up = plus_di > minus_di
            price_up = last["close"] > prev["close"]

            signal = None
            if adx > self.adx_threshold:
                if trend_up and (price_up or not self.require_trend_confirmation):
                    signal = {
                        "signal": "up",
                        "adx": round(adx, 2),
                        "plus_di": round(plus_di, 2),
                        "minus_di": round(minus_di, 2),
                        "confidence": 70 + min(20, int((adx - 25) / 2)),  # 70-90
                        "volume_ok": volume_ok
                    }
                    signal = self._apply_pattern_boost(signal, patterns, "up")
                elif not trend_up and (not price_up or not self.require_trend_confirmation):
                    signal = {
                        "signal": "down",
                        "adx": round(adx, 2),
                        "plus_di": round(plus_di, 2),
                        "minus_di": round(minus_di, 2),
                        "confidence": 75 + min(20, int((adx - 25) / 2)),  # 75-95
                        "volume_ok": volume_ok
                    }
                    signal = self._apply_pattern_boost(signal, patterns, "down")

            if signal and volume_ok:
                signal["confidence"] = min(95, signal["confidence"] + 10)

            return signal if (signal and signal["confidence"] >= self.min_confidence) else None

        except Exception as e:
            print(f"ADXStrategy error: {e}")
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
                signal.get("confidence", 70) + int(pattern_strength * 20 * self.pattern_boost)
            )
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal
