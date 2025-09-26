import numpy as np
from collections import deque
from config import CONFIG
from strategy.candlestick_patterns import detect_patterns, PATTERN_STRENGTH

class EnhancedPriceActionStrategy:
    def __init__(self, config=None):
        self.min_wick_ratio = config.get('min_wick_ratio', 2.0) if config else 2.0
        self.volume_threshold = config.get('volume_multiplier', 1.5) if config else 1.5
        self.trend_confirmation = config.get('confirmation', True) if config else True
        self.trend_lookback = config.get('trend_lookback', 3) if config else 3
        self.candle_lookback = config.get('candle_lookback', 5) if config else 5
        self.pattern_boost = config.get('pattern_boost', 0.2) if config else 0.2

        self.pattern_config = {
            'doji': {'max_body_ratio': 0.1},
            'hammer': {'max_body_ratio': 0.3, 'min_wick_ratio': 2.0},
            'engulfing': {'min_body_ratio': 0.3},
            'star': {'max_body_ratio': 0.2, 'min_gap_ratio': 0.5},
            'soldiers_crows': {'min_body_ratio': 0.7, 'min_consecutive': 3}
        }
        self.candle_buffer = deque(maxlen=5)

    def _analyze_trend(self, candles):
        lookback = min(self.trend_lookback, len(candles))
        if lookback < 3:
            return None
        closes = [c['close'] for c in candles[-lookback:]]
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]
        return 'up' if slope > 0 else 'down'

    def _is_volume_confirmed(self, current, reference):
        return current['volume'] > reference * self.volume_threshold

    def _detect_morning_star(self, candles):
        if len(candles) < 3:
            return False
        first, second, third = candles[-3], candles[-2], candles[-1]
        cond1 = (first['close'] < first['open'] and
                 (first['open'] - first['close']) / (first['high'] - first['low'] + 1e-8) > 0.6)
        cond2 = abs(second['close'] - second['open']) / (second['high'] - second['low'] + 1e-8) < 0.3
        cond3 = (third['close'] > third['open'] and
                 (third['close'] - third['open']) / (third['high'] - third['low'] + 1e-8) > 0.6 and
                 third['close'] > first['close'])
        return cond1 and cond2 and cond3

    def _detect_three_soldiers(self, candles):
        if len(candles) < 3:
            return False
        candles = candles[-3:]
        bodies = [(c['close'] - c['open']) / (c['high'] - c['low'] + 1e-8) for c in candles]
        cond1 = all(c['close'] > c['open'] and b > 0.7 for c, b in zip(candles, bodies))
        cond2 = all(candles[i]['open'] > candles[i-1]['open'] and
                    candles[i]['open'] < candles[i-1]['close'] for i in range(1, 3))
        cond3 = candles[-1]['close'] > candles[-2]['close'] > candles[-3]['close']
        return cond1 and cond2 and cond3

    def _apply_pattern_boost(self, signal, patterns):
        if not signal or not patterns:
            return signal
        # Price Action aceita todos (reversão, tendência, neutro)
        confirm_patterns = (
            CONFIG["candlestick_patterns"]["reversal_up"] +
            CONFIG["candlestick_patterns"]["reversal_down"] +
            CONFIG["candlestick_patterns"]["trend_up"] +
            CONFIG["candlestick_patterns"]["trend_down"] +
            CONFIG["candlestick_patterns"]["neutral"]
        )
        pattern_strength = 0
        for pattern in patterns:
            if pattern in confirm_patterns:
                pattern_strength += PATTERN_STRENGTH.get(pattern, 0.1)
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            # se já tiver confiança, soma, senão usa 70 como base
            signal["confidence"] = min(100, signal.get("confidence", 70) + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal

    def generate_signal(self, data):
        try:
            candles = data.get("history", [])
            if len(candles) < 5:
                return None
            self.candle_buffer.extend(candles[-5:])
            avg_volume = np.mean([c.get('volume', 0) for c in candles[-5:-1]] or [1])

            # Padrões de 3 candles
            morning_star = self._detect_morning_star(candles)
            evening_star = self._detect_morning_star(candles[::-1])
            three_soldiers = self._detect_three_soldiers(candles)
            three_crows = self._detect_three_soldiers(candles[::-1])

            # Morning Star
            if morning_star:
                trend = self._analyze_trend(candles[:-3])
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
                    # BOOST
                    patterns = detect_patterns(candles[-self.candle_lookback:])
                    return self._apply_pattern_boost(signal, patterns)

            # Evening Star
            if evening_star:
                trend = self._analyze_trend(candles[:-3])
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
                    patterns = detect_patterns(candles[-self.candle_lookback:])
                    return self._apply_pattern_boost(signal, patterns)

            # Three White Soldiers
            if three_soldiers:
                trend = self._analyze_trend(candles[:-3])
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
                    patterns = detect_patterns(candles[-self.candle_lookback:])
                    return self._apply_pattern_boost(signal, patterns)

            # Three Black Crows
            if three_crows:
                trend = self._analyze_trend(candles[:-3])
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
                    patterns = detect_patterns(candles[-self.candle_lookback:])
                    return self._apply_pattern_boost(signal, patterns)

            # ==== Padrões básicos (como na versão clássica) ====
            current = candles[-1]
            prev = candles[-2]

            open_ = current["open"]
            close = current["close"]
            high = current["high"]
            low = current["low"]

            body = abs(close - open_)
            upper_wick = high - max(open_, close)
            lower_wick = min(open_, close) - low
            total_range = high - low if high != low else 0.0001

            # DOJI: Corpo muito pequeno, indecisão
            if body / total_range < self.pattern_config['doji']['max_body_ratio']:
                signal = {"signal": None, "pattern": "doji"}
                patterns = detect_patterns(candles[-self.candle_lookback:])
                return self._apply_pattern_boost(signal, patterns)

            # HAMMER / HANGING MAN
            if body / total_range < self.pattern_config['hammer']['max_body_ratio'] and lower_wick / (body + 1e-8) > self.pattern_config['hammer']['min_wick_ratio']:
                direction = "up" if close > open_ else "down"
                pattern = "hammer" if direction == "up" else "hanging_man"
                signal = {"signal": "up" if pattern == "hammer" else "down", "pattern": pattern}
                patterns = detect_patterns(candles[-self.candle_lookback:])
                return self._apply_pattern_boost(signal, patterns)

            # ENGULFING BULLISH
            if close > open_ and prev["close"] < prev["open"] and close > prev["open"] and open_ < prev["close"]:
                signal = {"signal": "up", "pattern": "bullish_engulfing"}
                patterns = detect_patterns(candles[-self.candle_lookback:])
                return self._apply_pattern_boost(signal, patterns)

            # ENGULFING BEARISH
            if close < open_ and prev["close"] > prev["open"] and close < prev["open"] and open_ > prev["close"]:
                signal = {"signal": "down", "pattern": "bearish_engulfing"}
                patterns = detect_patterns(candles[-self.candle_lookback:])
                return self._apply_pattern_boost(signal, patterns)

            # PIN BAR (forte rejeição de preço)
            if upper_wick > body * self.min_wick_ratio and lower_wick < body * 0.3:
                signal = {"signal": "down", "pattern": "pinbar_top"}
                patterns = detect_patterns(candles[-self.candle_lookback:])
                return self._apply_pattern_boost(signal, patterns)
            elif lower_wick > body * self.min_wick_ratio and upper_wick < body * 0.3:
                signal = {"signal": "up", "pattern": "pinbar_bottom"}
                patterns = detect_patterns(candles[-self.candle_lookback:])
                return self._apply_pattern_boost(signal, patterns)

            return None

        except Exception as e:
            print(f"Error in EnhancedPriceAction: {e}")
            return None
