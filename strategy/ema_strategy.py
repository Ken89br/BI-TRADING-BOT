from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class EMAStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.short_period = config.get("short_period", 9)
        self.long_period = config.get("long_period", 21)
        self.candle_lookback = config.get("candle_lookback", 3)
        self.pattern_boost = config.get("pattern_boost", 0.2)
        self.min_confidence = config.get("min_confidence", 70)

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(self.short_period, self.long_period, self.candle_lookback, 3):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2]

            short_ema_col = f"ema_{self.short_period}"
            long_ema_col = f"ema_{self.long_period}"

            short_ema_prev = prev.get(short_ema_col, None)
            long_ema_prev = prev.get(long_ema_col, None)
            short_ema_last = last.get(short_ema_col, None)
            long_ema_last = last.get(long_ema_col, None)

            if None in [short_ema_prev, long_ema_prev, short_ema_last, long_ema_last]:
                return None

            prev_cross = short_ema_prev - long_ema_prev
            current_cross = short_ema_last - long_ema_last

            signal = None
            if prev_cross < 0 and current_cross > 0:
                signal = {
                    "signal": "up",
                    "indicator": "ema_crossover",
                    "short_ema": short_ema_last,
                    "long_ema": long_ema_last,
                    "confidence": self._calculate_confidence(abs(current_cross))
                }
            elif prev_cross > 0 and current_cross < 0:
                signal = {
                    "signal": "down",
                    "indicator": "ema_crossover",
                    "short_ema": short_ema_last,
                    "long_ema": long_ema_last,
                    "confidence": self._calculate_confidence(abs(current_cross))
                }

            if signal:
                patterns = last.get("patterns", [])
                signal = self._apply_pattern_boost(signal, patterns)
                if signal.get("confidence", 0) >= self.min_confidence:
                    return signal

            return None

        except Exception as e:
            print(f"Erro em EMAStrategy: {e}")
            return None

    def _apply_pattern_boost(self, signal, patterns):
        if not signal or not patterns:
            return signal
        direction = signal["signal"]
        if direction == "up":
            confirm_patterns = CONFIG["candlestick_patterns"]["trend_up"] + CONFIG["candlestick_patterns"]["neutral"]
        else:
            confirm_patterns = CONFIG["candlestick_patterns"]["trend_down"] + CONFIG["candlestick_patterns"]["neutral"]

        pattern_strength = sum(PATTERN_STRENGTH.get(pattern, 0.1) for pattern in patterns if pattern in confirm_patterns)

        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            signal["confidence"] = min(100, signal.get("confidence", 70) + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength

        return signal

    def _calculate_confidence(self, spread):
        normalized_spread = min(spread / (self.short_period * 0.1), 1.0)
        return int(50 + 50 * normalized_spread)
