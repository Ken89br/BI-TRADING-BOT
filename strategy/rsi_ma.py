from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class AggressiveRSIMA:
    def __init__(self, config=None):
        config = config or {}
        self.rsi_period = config.get('rsi_period', 14)
        self.ma_period = config.get('ma_period', 5)
        self.overbought = config.get('overbought', 65)
        self.oversold = config.get('oversold', 35)
        self.candle_lookback = config.get('candle_lookback', 3)
        self.pattern_boost = config.get('pattern_boost', 0.2)
        self.require_confirmation = config.get('confirmation', True)
        self.volume_threshold = config.get('volume_threshold', 1.2)
        self.min_confidence = config.get('min_confidence', 65)

    def generate_signal(self, features_df):
        try:
            # Checa se há dados suficientes
            if features_df is None or features_df.empty or len(features_df) < max(2, self.candle_lookback, self.ma_period):
                return None

            last = features_df.iloc[-1]
            prev = features_df.iloc[-2] if len(features_df) > 1 else last

            # RSI e MA (ajuste o nome da coluna se mudar o período no pipeline universal!)
            rsi_col = "rsi_value"
            ma_col = f"sma_{self.ma_period}"

            rsi = last[rsi_col]
            ma = last[ma_col] if ma_col in last else None
            close = last["close"]
            volume = last["volume"]
            prev_close = prev["close"]
            prev_rsi = prev[rsi_col]
            prev_ma = prev[ma_col] if ma_col in prev else None
            patterns = last.get("patterns", [])
            pattern_strength = last.get("pattern_strength", 0)

            # Confirmação de volume
            avg_vol = features_df.iloc[-5:]["volume"].mean()
            volume_ok = volume > avg_vol * self.volume_threshold if avg_vol else False

            signal = None
            strength = "medium"
            if ma is not None:
                if (rsi < self.oversold and close > ma and
                    (not self.require_confirmation or prev_rsi < rsi)):
                    strength = "high" if volume_ok else "medium"
                    signal = self._package("up", last, strength, rsi, ma)
                elif (rsi > self.overbought and close < ma and
                      (not self.require_confirmation or prev_rsi > rsi)):
                    strength = "high" if volume_ok else "medium"
                    signal = self._package("down", last, strength, rsi, ma)

            # BOOST: padrões de vela
            if signal and patterns:
                direction = signal["signal"]
                if direction == "up":
                    confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"]
                elif direction == "down":
                    confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"]
                else:
                    confirm_patterns = []
                boost = sum(PATTERN_STRENGTH.get(p, 0) for p in patterns if p in confirm_patterns)
                if boost > 0:
                    signal["confidence"] = min(100, signal["confidence"] + int(boost * 20 * self.pattern_boost))
                    signal["patterns"] = patterns
                    signal["pattern_strength"] = pattern_strength

            # Só retorna se atingir confiança mínima
            if signal and signal.get("confidence", 0) >= self.min_confidence:
                return signal
            return None
        except Exception as e:
            print(f"Error in AggressiveRSIMA: {e}")
            return None

    def _package(self, direction, last, strength, rsi, ma):
        price = last["close"]
        high = last["high"]
        low = last["low"]
        volume = last["volume"]
        base_conf = {"high": 85, "medium": 70, "low": 55}.get(strength, 50)
        ma_distance = abs(price - ma) / ma if ma else 0
        volume_factor = min(1, volume / (last["volume"] + 1e-10))
        confidence = min(100, base_conf + (10 * ma_distance * 100) + (5 * volume_factor))
        return {
            "signal": direction,
            "price": price,
            "high": high,
            "low": low,
            "volume": volume,
            "rsi": rsi,
            "ma": ma,
            "recommend_entry": (high + low) / 2,
            "recommend_stop": low if direction == "up" else high,
            "strength": strength,
            "confidence": int(confidence),
            "indicators": {
                "price_ma_ratio": price / ma if ma else None,
            }
            }
