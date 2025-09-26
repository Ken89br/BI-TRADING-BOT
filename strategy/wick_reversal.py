from strategy.config_strategy import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class WickReversalStrategy:
    def __init__(self, config=None):
        cfg = None
        if config and "wick_reversal" in config:
            cfg = config["wick_reversal"]
        else:
            cfg = config or {}
        self.wick_ratio = cfg.get('wick_ratio', 2.0)
        self.min_body_ratio = cfg.get('min_body_ratio', 0.1)
        self.volume_multiplier = cfg.get('volume_multiplier', 1.5)
        self.trend_confirmation = cfg.get('trend_confirmation', True)
        self.pattern_boost = cfg.get("pattern_boost", 0.2)
        self.candle_lookback = cfg.get("candle_lookback", 3)

    def generate_signal(self, features_df):
        try:
            if (
                features_df is None
                or features_df.empty
                or len(features_df) < max(3, self.candle_lookback)
            ):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            open_ = last["open"]
            close = last["close"]
            high = last["high"]
            low = last["low"]
            volume = last["volume"]
            prev_volume = prev["volume"]
            body_size = abs(close - open_)
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low

            if body_size == 0 or (upper_wick + lower_wick) == 0:
                return None

            volume_ok = volume > prev_volume * self.volume_multiplier

            trend_aligned = True
            if self.trend_confirmation:
                prev_trend = prev["close"] - prev["open"]
                trend_aligned = (
                    (lower_wick > body_size * self.wick_ratio and prev_trend < 0)
                    or (upper_wick > body_size * self.wick_ratio and prev_trend > 0)
                )

            patterns = last.get("patterns", [])

            wick_signal = None
            if (
                lower_wick > body_size * self.wick_ratio
                and body_size / (upper_wick + 1e-8) > self.min_body_ratio
            ):
                wick_signal = "up"
            elif (
                upper_wick > body_size * self.wick_ratio
                and body_size / (lower_wick + 1e-8) > self.min_body_ratio
            ):
                wick_signal = "down"

            if wick_signal:
                strength = "high" if (volume_ok and trend_aligned) else "medium"
                signal = self._package(wick_signal, last, prev, strength)
                signal = self._apply_pattern_boost(signal, patterns)
                return signal

            return None

        except Exception as e:
            print(f"Error in WickReversalStrategy: {e}")
            return None

    def _apply_pattern_boost(self, signal, patterns):
        if not signal or not patterns:
            return signal
        direction = signal["signal"]
        if direction == "up":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"] + CONFIG["candlestick_patterns"]["neutral"]
        else:
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"] + CONFIG["candlestick_patterns"]["neutral"]

        pattern_strength = sum(
            PATTERN_STRENGTH.get(pattern, 0.1) for pattern in patterns if pattern in confirm_patterns
        )
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            signal["confidence"] = min(100, signal["confidence"] + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal

    def _package(self, direction, last, prev, strength):
        price = last["close"]
        high = last["high"]
        low = last["low"]
        volume = last["volume"]
        base_confidence = {"high": 90, "medium": 75, "low": 60}.get(strength, 50)
        # Volume boost nos últimos 4 candles
        volume_boost = 0
        if "volume" in last and "volume" in prev:
            if prev["volume"] > 0:
                volume_boost = min(10, max(0, (volume - prev["volume"]) / (prev["volume"] + 1e-8) * 10))
        confidence = min(100, base_confidence + volume_boost)
        return {
            "signal": direction,
            "price": price,
            "high": high,
            "low": low,
            "volume": volume,
            "recommend_entry": (high + low) / 2,
            "recommend_stop": low if direction == "up" else high,
            "strength": strength,
            "confidence": confidence,
            "wick_ratio": self.wick_ratio,
            "candle_size": high - low,
            "context": {
                "prev_trend": prev["close"] - prev["open"] if prev is not None else 0,
                "volume_change": volume / prev["volume"] if prev is not None and prev["volume"] > 0 else 1
            }
            }
