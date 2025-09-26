from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class SMACrossStrategy:
    def __init__(self, short_period=5, long_period=10, min_history=20, confirmation_candles=3, candle_lookback=3, pattern_boost=0.2):
        self.short_period = short_period
        self.long_period = long_period
        self.min_history = max(min_history, long_period)
        self.confirmation_candles = confirmation_candles
        self.candle_lookback = candle_lookback
        self.pattern_boost = pattern_boost

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(self.min_history, self.candle_lookback, 5):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            sma_short_col = f"sma_{self.short_period}"
            sma_long_col = f"sma_{self.long_period}"
            sma_short = last.get(sma_short_col, None)
            sma_long = last.get(sma_long_col, None)
            if sma_short is None or sma_long is None:
                return None

            current_cross = sma_short - sma_long
            signal = None

            if current_cross > 0 and (getattr(self, 'trend', None) != "up" or not getattr(self, 'trend', None)):
                if len(features_df) >= self.confirmation_candles:
                    prev_sma_short = features_df.iloc[-self.confirmation_candles-1:-1][sma_short_col].mean()
                    if all(features_df.iloc[-self.confirmation_candles:][sma_short_col] > prev_sma_short):
                        signal = {"signal": "up", "type": "sma_cross"}
                        self.trend = "up"

            elif current_cross < 0 and (getattr(self, 'trend', None) != "down" or not getattr(self, 'trend', None)):
                if len(features_df) >= self.confirmation_candles:
                    prev_sma_short = features_df.iloc[-self.confirmation_candles-1:-1][sma_short_col].mean()
                    if all(features_df.iloc[-self.confirmation_candles:][sma_short_col] < prev_sma_short):
                        signal = {"signal": "down", "type": "sma_cross"}
                        self.trend = "down"

            if signal:
                patterns = last.get("patterns", [])
                signal = self._apply_pattern_boost(signal, patterns)
                signal.update({
                    "sma_short": sma_short,
                    "sma_long": sma_long,
                    "spread": abs(sma_short - sma_long),
                    "confidence": self._calculate_confidence(features_df),
                    "price": last["close"],
                    "volume": last.get("volume", 0)
                })
                return signal

            return None

        except Exception as e:
            print(f"Error in SMACrossStrategy: {e}")
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

    def _calculate_confidence(self, features_df):
        closes = features_df["close"]
        price_change = closes.iloc[-1] - closes.iloc[-5]
        sma_short = features_df.iloc[-1][f"sma_{self.short_period}"]
        sma_long = features_df.iloc[-1][f"sma_{self.long_period}"]
        spread = abs(sma_short - sma_long)
        price_factor = min(max(price_change / (closes.iloc[-5] * 0.01), -2), 2)
        spread_factor = spread / (closes.iloc[-1] * 0.01)
        confidence = 50 + (20 * price_factor) + (30 * min(spread_factor, 1))
        return min(max(int(confidence), 0), 100)
