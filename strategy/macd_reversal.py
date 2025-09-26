import numpy as np
from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class MACDReversalStrategy:
    def __init__(self, config=None):
        config = config or {}
        self.threshold = config.get('threshold', 0.1)
        self.candle_lookback = config.get('candle_lookback', 3)
        self.pattern_boost = config.get('pattern_boost', 0.2)

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < 3:
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2]

            macd = last["macd_line"]
            signal_line = last["macd_signal_line"]
            hist = last["macd_histogram"]
            prev_hist = prev["macd_histogram"]

            price = last["close"]
            high = last["high"]
            low = last["low"]
            volume = last["volume"]

            result_signal = None
            strength = "medium"
            price_above_ma = price > last.get("sma_20", price)
            if (prev_hist < 0 < hist) or (prev_hist < -self.threshold and hist > -self.threshold/2):
                if price_above_ma:
                    strength = "high"
                result_signal = self._package("up", last, strength, macd, signal_line, hist)
            elif (prev_hist > 0 > hist) or (prev_hist > self.threshold and hist < self.threshold/2):
                if not price_above_ma:
                    strength = "high"
                result_signal = self._package("down", last, strength, macd, signal_line, hist)

            # BOOST: padrões de velas
            if result_signal:
                patterns = last.get("patterns", [])
                result_signal = self._apply_pattern_boost(result_signal, patterns)

            return result_signal
        except Exception as e:
            print(f"MACDReversal error: {str(e)}")
            return None

    def _apply_pattern_boost(self, signal, patterns):
        if not signal or not patterns:
            return signal
        direction = signal["signal"]
        if direction == "up":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"]
        elif direction == "down":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"]
        else:
            confirm_patterns = []
        pattern_strength = sum(PATTERN_STRENGTH.get(p, 0.1) for p in patterns if p in confirm_patterns)
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            signal["confidence"] = min(100, signal.get("confidence", 70) + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal

    def _package(self, signal, last, strength, macd, signal_line, hist):
        price = float(last['close'])
        high = float(last['high'])
        low = float(last['low'])
        base_conf = {"high": 85, "medium": 70, "low": 55}.get(strength, 50)
        hist_boost = min(20, abs(hist) * 100)
        ma_boost = 10 if signal == "up" and price > signal_line else -10 if signal == "down" and price < signal_line else 0
        confidence = min(100, base_conf + hist_boost + ma_boost)
        return {
            "signal": signal,
            "price": price,
            "high": high,
            "low": low,
            "volume": last.get('volume', 0),
            "macd": macd,
            "signal_line": signal_line,
            "histogram": hist,
            "recommend_entry": (high + low) / 2,
            "recommend_stop": low if signal == "up" else high,
            "strength": strength,
            "confidence": int(confidence),
            "indicators": {
                "price_vs_signal": price / signal_line if signal_line else 1,
                "macd_cross": "golden" if signal == "up" else "death",
                "volatility": last.get("volatility", 0)
            }
        }
