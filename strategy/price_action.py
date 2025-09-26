import numpy as np
from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class EnhancedPriceActionStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.min_wick_ratio = config.get('min_wick_ratio', 2.0)
        self.volume_threshold = config.get('volume_multiplier', 1.5)
        self.trend_confirmation = config.get('confirmation', True)
        self.trend_lookback = config.get('trend_lookback', 3)
        self.candle_lookback = config.get('candle_lookback', 5)
        self.pattern_boost = config.get('pattern_boost', 0.2)

        self.pattern_config = {
            'doji': {'max_body_ratio': 0.1},
            'hammer': {'max_body_ratio': 0.3, 'min_wick_ratio': 2.0},
        }

    def _analyze_trend(self, closes):
        if len(closes) < self.trend_lookback:
            return None
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        return 'up' if slope > 0 else 'down'

    def _apply_pattern_boost(self, signal, patterns):
        if not signal or not patterns:
            return signal
        confirm_patterns = (
            CONFIG["candlestick_patterns"]["reversal_up"] +
            CONFIG["candlestick_patterns"]["reversal_down"] +
            CONFIG["candlestick_patterns"]["trend_up"] +
            CONFIG["candlestick_patterns"]["trend_down"] +
            CONFIG["candlestick_patterns"]["neutral"]
        )
        pattern_strength = sum(PATTERN_STRENGTH.get(pattern, 0.1) for pattern in patterns if pattern in confirm_patterns)
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            signal["confidence"] = min(100, signal.get("confidence", 70) + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(5, self.candle_lookback):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            # Para padrões de 3 candles, pega os últimos 3 registros como dicionários
            candles = features_df.iloc[-5:].to_dict("records")
            closes = features_df["close"].values

            avg_volume = features_df.iloc[-5:-1]["volume"].mean() if len(features_df) > 5 else 1

            patterns = last.get("patterns", [])

            # ==== Morning Star ====
            if (
                len(candles) >= 3 and
                candles[-3]["close"] < candles[-3]["open"] and
                (candles[-3]["open"] - candles[-3]["close"]) / (candles[-3]["high"] - candles[-3]["low"] + 1e-8) > 0.6 and
                abs(candles[-2]["close"] - candles[-2]["open"]) / (candles[-2]["high"] - candles[-2]["low"] + 1e-8) < 0.3 and
                candles[-1]["close"] > candles[-1]["open"] and
                (candles[-1]["close"] - candles[-1]["open"]) / (candles[-1]["high"] - candles[-1]["low"] + 1e-8) > 0.6 and
                candles[-1]["close"] > candles[-3]["close"]
            ):
                trend = self._analyze_trend(closes[:-3])
                volume_ok = candles[-1]['volume'] > avg_volume * self.volume_threshold
                if not self.trend_confirmation or (trend == 'down'):
                    signal = {
                        "signal": "up",
                        "pattern": "morning_star",
                        "confidence": 85 if volume_ok else 70,
                        "context": {
                            "trend": trend,
                            "volume_ratio": candles[-1]['volume'] / avg_volume if avg_volume else 0
                        }
                    }
                    return self._apply_pattern_boost(signal, patterns)

            # ==== Evening Star ====
            # (simples reverso do morning_star)
            if (
                len(candles) >= 3 and
                candles[-3]["close"] > candles[-3]["open"] and
                (candles[-3]["close"] - candles[-3]["open"]) / (candles[-3]["high"] - candles[-3]["low"] + 1e-8) > 0.6 and
                abs(candles[-2]["close"] - candles[-2]["open"]) / (candles[-2]["high"] - candles[-2]["low"] + 1e-8) < 0.3 and
                candles[-1]["close"] < candles[-1]["open"] and
                (candles[-1]["open"] - candles[-1]["close"]) / (candles[-1]["high"] - candles[-1]["low"] + 1e-8) > 0.6 and
                candles[-1]["close"] < candles[-3]["close"]
            ):
                trend = self._analyze_trend(closes[:-3])
                volume_ok = candles[-1]['volume'] > avg_volume * self.volume_threshold
                if not self.trend_confirmation or (trend == 'up'):
                    signal = {
                        "signal": "down",
                        "pattern": "evening_star",
                        "confidence": 85 if volume_ok else 70,
                        "context": {
                            "trend": trend,
                            "volume_ratio": candles[-1]['volume'] / avg_volume if avg_volume else 0
                        }
                    }
                    return self._apply_pattern_boost(signal, patterns)

            # ==== Three White Soldiers ====
            if (
                len(candles) >= 3 and
                all(c["close"] > c["open"] and (c["close"] - c["open"]) / (c["high"] - c["low"] + 1e-8) > 0.7 for c in candles[-3:]) and
                candles[-2]["open"] > candles[-3]["open"] and candles[-2]["open"] < candles[-3]["close"] and
                candles[-1]["open"] > candles[-2]["open"] and candles[-1]["open"] < candles[-2]["close"] and
                candles[-1]["close"] > candles[-2]["close"] > candles[-3]["close"]
            ):
                trend = self._analyze_trend(closes[:-3])
                if not self.trend_confirmation or (trend != 'down'):
                    signal = {
                        "signal": "up",
                        "pattern": "three_white_soldiers",
                        "confidence": 90,
                        "context": {
                            "trend": trend,
                            "consecutive_bodies": 3
                        }
                    }
                    return self._apply_pattern_boost(signal, patterns)

            # ==== Three Black Crows ====
            if (
                len(candles) >= 3 and
                all(c["close"] < c["open"] and (c["open"] - c["close"]) / (c["high"] - c["low"] + 1e-8) > 0.7 for c in candles[-3:]) and
                candles[-2]["open"] < candles[-3]["open"] and candles[-2]["open"] > candles[-3]["close"] and
                candles[-1]["open"] < candles[-2]["open"] and candles[-1]["open"] > candles[-2]["close"] and
                candles[-1]["close"] < candles[-2]["close"] < candles[-3]["close"]
            ):
                trend = self._analyze_trend(closes[:-3])
                if not self.trend_confirmation or (trend != 'up'):
                    signal = {
                        "signal": "down",
                        "pattern": "three_black_crows",
                        "confidence": 90,
                        "context": {
                            "trend": trend,
                            "consecutive_bodies": 3
                        }
                    }
                    return self._apply_pattern_boost(signal, patterns)

            # ==== DOJI: Corpo muito pequeno, indecisão ====
            open_ = last["open"]
            close = last["close"]
            high = last["high"]
            low = last["low"]
            body = abs(close - open_)
            total_range = high - low if high != low else 0.0001
            if body / total_range < self.pattern_config['doji']['max_body_ratio']:
                signal = {"signal": None, "pattern": "doji"}
                return self._apply_pattern_boost(signal, patterns)

            # ==== HAMMER / HANGING MAN ====
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            if body / total_range < self.pattern_config['hammer']['max_body_ratio'] and lower_wick / (body + 1e-8) > self.pattern_config['hammer']['min_wick_ratio']:
                direction = "up" if close > open_ else "down"
                pattern = "hammer" if direction == "up" else "hanging_man"
                signal = {"signal": "up" if pattern == "hammer" else "down", "pattern": pattern}
                return self._apply_pattern_boost(signal, patterns)

            # ==== ENGULFING BULLISH ====
            prev_open = prev["open"]
            prev_close = prev["close"]
            if close > open_ and prev_close < prev_open and close > prev_open and open_ < prev_close:
                signal = {"signal": "up", "pattern": "bullish_engulfing"}
                return self._apply_pattern_boost(signal, patterns)

            # ==== ENGULFING BEARISH ====
            if close < open_ and prev_close > prev_open and close < prev_open and open_ > prev_close:
                signal = {"signal": "down", "pattern": "bearish_engulfing"}
                return self._apply_pattern_boost(signal, patterns)

            # ==== PIN BAR (rejeição de preço) ====
            if upper_wick > body * self.min_wick_ratio and lower_wick < body * 0.3:
                signal = {"signal": "down", "pattern": "pinbar_top"}
                return self._apply_pattern_boost(signal, patterns)
            elif lower_wick > body * self.min_wick_ratio and upper_wick < body * 0.3:
                signal = {"signal": "up", "pattern": "pinbar_bottom"}
                return self._apply_pattern_boost(signal, patterns)

            return None

        except Exception as e:
            print(f"Error in EnhancedPriceAction: {e}")
            return None
