import pandas as pd
import numpy as np
from typing import Dict, Union
from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class RSIStrategy:
    """
    RSI profissional:
    - RSI suavizado (EMA + MA)
    - Bandas dinâmicas OB/OS (adapta à volatilidade)
    - Divergências (bullish/bearish)
    - Cruzamento linha de sinal
    - Confirmação por volume
    - Força dinâmica (strong_volume, strong_divergence, strong_extreme, weak)
    - Boost dinâmico por padrões de vela
    - Retorno universalizado
    """

    def __init__(self, config=None):
        config = config or {}
        self.rsi_period = config.get("rsi_period", 14)
        self.smooth_period = config.get("smooth_period", 3)
        self.atr_period = config.get("atr_period", 14)
        self.vol_window = config.get("vol_window", 20)
        self.min_confidence = config.get("min_confidence", 65)
        self.pattern_boost = config.get("pattern_boost", 0.2)
        self.volume_multiplier = config.get("volume_multiplier", 1.5)
        self.divergence_lookback = config.get("divergence_lookback", 14)

    def _calculate_rsi(self, close: pd.Series, period: int, smooth: int):
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_smoothed = rsi.rolling(window=smooth, min_periods=1).mean()
        signal_line = rsi_smoothed.rolling(window=smooth, min_periods=1).mean()
        return rsi, rsi_smoothed, signal_line

    def _identify_divergence(self, close: pd.Series, rsi: pd.Series, lookback: int):
        price = close.iloc[-lookback:]
        rsi_sub = rsi.iloc[-lookback:]

        price_max_idx = price[price == price.max()].index[-1]
        price_min_idx = price[price == price.min()].index[-1]
        rsi_max_idx = rsi_sub[rsi_sub == rsi_sub.max()].index[-1]
        rsi_min_idx = rsi_sub[rsi_sub == rsi_sub.min()].index[-1]

        bullish = (price.iloc[-1] < price.loc[price_min_idx]) and (rsi.iloc[-1] > rsi.loc[rsi_min_idx])
        bearish = (price.iloc[-1] > price.loc[price_max_idx]) and (rsi.iloc[-1] < rsi.loc[rsi_max_idx])
        return {"bullish": bullish, "bearish": bearish}

    def _pattern_boost(self, signal: Dict, patterns: list):
        if not signal or not patterns:
            return signal
        direction = signal.get("signal")
        if direction == "up":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"] + CONFIG["candlestick_patterns"]["trend_up"]
        elif direction == "down":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"] + CONFIG["candlestick_patterns"]["trend_down"]
        else:
            confirm_patterns = CONFIG["candlestick_patterns"]["neutral"]
        pattern_strength = sum(PATTERN_STRENGTH.get(p, 0.1) for p in patterns if p in confirm_patterns)
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * self.pattern_boost)
            signal["confidence"] = min(100, signal.get("confidence", 70) + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal

    def generate_signal(self, features_df):
        try:
            if features_df is None or features_df.empty or len(features_df) < max(self.rsi_period+2, self.divergence_lookback+2):
                return None

            close = features_df["close"]
            volume = features_df["volume"]
            high = features_df["high"] if "high" in features_df else None
            low = features_df["low"] if "low" in features_df else None
            last = features_df.iloc[-1]
            patterns = last.get("patterns", [])

            # RSI institucional
            rsi_raw, rsi_smoothed, signal_line = self._calculate_rsi(close, self.rsi_period, self.smooth_period)
            current_rsi = rsi_smoothed.iloc[-1]
            previous_rsi = rsi_smoothed.iloc[-2]
            current_signal = signal_line.iloc[-1]
            previous_signal = signal_line.iloc[-2]

            # Volatilidade (ATR)
            if high is not None and low is not None:
                tr = pd.DataFrame({
                    'hl': high - low,
                    'hc': (high - close.shift(1)).abs(),
                    'lc': (low - close.shift(1)).abs()
                }).max(axis=1)
                volatility = tr.rolling(window=self.atr_period).mean()
            else:
                volatility = close.rolling(window=self.vol_window).std()

            # Bandas dinâmicas
            overbought = 70 - (volatility.iloc[-1] * 0.5 if not np.isnan(volatility.iloc[-1]) else 0)
            oversold = 30 + (volatility.iloc[-1] * 0.5 if not np.isnan(volatility.iloc[-1]) else 0)

            # Sinal base e tipo de evento
            signal = None
            event = ""
            if current_rsi > overbought and previous_rsi <= overbought:
                signal = "down"
                event = "overbought"
            elif current_rsi < oversold and previous_rsi >= oversold:
                signal = "up"
                event = "oversold"
            elif current_rsi > current_signal and previous_rsi <= previous_signal:
                signal = "up"
                event = "bullish_crossover"
            elif current_rsi < current_signal and previous_rsi >= previous_signal:
                signal = "down"
                event = "bearish_crossover"

            # Divergência (sobrescreve outros sinais)
            divergence = self._identify_divergence(close, rsi_smoothed, self.divergence_lookback)
            if divergence["bullish"]:
                signal = "up"
                event = "bullish_divergence"
            elif divergence["bearish"]:
                signal = "down"
                event = "bearish_divergence"

            # Força dinâmica (detalhada)
            avg_vol = volume.rolling(window=20).mean().iloc[-1]
            vol_ok = volume.iloc[-1] > avg_vol * self.volume_multiplier if avg_vol > 0 else False
            strength = "weak"
            if signal:
                if vol_ok:
                    strength = "strong_volume"
                elif divergence["bullish"] or divergence["bearish"]:
                    strength = "strong_divergence"
                elif (event in ["overbought", "oversold"] and abs(current_rsi - (overbought if event == "overbought" else oversold)) > 5):
                    strength = "strong_extreme"

            # Confiança
            confidence = 65
            if strength.startswith("strong"):
                confidence = 80
            elif signal is not None:
                confidence = 70

            # Retorno universalizado
            result = {
                "signal": signal,
                "event": event,
                "confidence": confidence,
                "strength": strength,
                "rsi_value": round(current_rsi, 2),
                "rsi_raw": round(rsi_raw.iloc[-1], 2),
                "rsi_signal_line": round(signal_line.iloc[-1], 2),
                "overbought": round(overbought, 2),
                "oversold": round(oversold, 2),
                "divergence": divergence,
                "price": close.iloc[-1],
                "volume": volume.iloc[-1],
                "vol_ok": vol_ok,
            }

            # Boost dinâmico por padrões de vela
            result = self._pattern_boost(result, patterns)

            if signal is not None and result["confidence"] >= self.min_confidence:
                result["risk_management"] = {
                    "stop_loss": round(close.iloc[-1]*0.98, 4) if signal == "up" else round(close.iloc[-1]*1.02, 4),
                    "take_profit": round(close.iloc[-1]*1.03, 4) if signal == "up" else round(close.iloc[-1]*0.97, 4),
                    "risk_reward_ratio": "1:2"
                }
                return result
            return None
        except Exception as e:
            print(f"RSIStrategyerror: {e}")
            return None
