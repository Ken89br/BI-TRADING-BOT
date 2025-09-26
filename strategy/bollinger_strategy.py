
import pandas as pd
import numpy as np
from typing import Dict, Union
from config import CONFIG
from strategy.candlestick_patterns import PATTERN_STRENGTH

class BollingerBandsStrategy:
    """
    Estratégia Bollinger Bands com:
    - Cálculo das bandas (SMA/EMA)
    - Detecção de reversões e breakouts
    - Confirmações por RSI e volume
    - Recomendação de entrada baseada na volatilidade
    - Integração com padrões de velas para boost de confiança
    """

    @staticmethod
    def calculate_bollinger_bands(close: pd.Series,
                                  period: int = 20,
                                  std_dev: float = 2.0,
                                  ma_type: str = 'sma') -> Dict[str, pd.Series]:
        if ma_type == 'sma':
            middle_band = close.rolling(window=period).mean()
        elif ma_type == 'ema':
            middle_band = close.ewm(span=period, adjust=False).mean()
        else:
            raise ValueError("ma_type must be 'sma' or 'ema'")

        std = close.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        bandwidth = (upper_band - lower_band) / middle_band * 100

        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'bandwidth': bandwidth,
            'std_dev': std
        }

    @staticmethod
    def identify_conditions(close: pd.Series,
                            upper_band: pd.Series,
                            middle_band: pd.Series,
                            lower_band: pd.Series,
                            bandwidth: pd.Series) -> Dict[str, Union[bool, float]]:
        current_price = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = middle_band.iloc[-1]
        bandwidth_pct = bandwidth.rank(pct=True).iloc[-1] * 100
        squeeze = bandwidth_pct < 20

        return {
            'above_upper': current_price >= current_upper,
            'below_lower': current_price <= current_lower,
            'squeeze': squeeze,
            'bandwidth_pct': bandwidth_pct,
            'distance_to_middle': ((current_price - current_middle) / current_middle) * 100,
        }

    @staticmethod
    def _apply_pattern_boost(signal: Dict, patterns: list) -> Dict:
        if not signal or not patterns:
            return signal

        direction = signal.get("signal")
        if direction == "up":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_up"] + CONFIG["candlestick_patterns"]["trend_up"]
        elif direction == "down":
            confirm_patterns = CONFIG["candlestick_patterns"]["reversal_down"] + CONFIG["candlestick_patterns"]["trend_down"]
        else:
            confirm_patterns = []

        if signal.get("signal", "").endswith("breakout"):
            confirm_patterns += CONFIG["candlestick_patterns"]["neutral"]
            
        pattern_strength = sum(PATTERN_STRENGTH.get(p, 0.1) for p in patterns if p in confirm_patterns)
        if pattern_strength > 0:
            boost = int(pattern_strength * 20 * 0.2)
            signal["confidence"] = min(95, signal.get("confidence", 70) + boost)
            signal["patterns"] = patterns
            signal["pattern_strength"] = pattern_strength
        return signal

    @staticmethod
    def generate_signals(close: pd.Series,
                         upper_band: pd.Series,
                         middle_band: pd.Series,
                         lower_band: pd.Series,
                         conditions: Dict,
                         volume: pd.Series = None,
                         rsi: pd.Series = None,
                         patterns: list = None) -> Dict[str, Union[str, float]]:
        current_price = close.iloc[-1]
        previous_price = close.iloc[-2]
        signal = None

        if conditions['above_upper'] and rsi is not None and rsi.iloc[-1] > 70:
            signal = 'bearish_reversal'
        elif conditions['below_lower'] and rsi is not None and rsi.iloc[-1] < 30:
            signal = 'bullish_reversal'

        if conditions['squeeze'] and signal is None:
            vol_ok = volume is not None and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.2
            if current_price > previous_price and vol_ok:
                signal = 'bullish_breakout'
            elif current_price < previous_price and vol_ok:
                signal = 'bearish_breakout'

        strength = 'weak'
        if signal:
            if volume is not None and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                strength = 'strong_volume'
            elif conditions['squeeze'] and signal.endswith('breakout'):
                strength = 'strong_squeeze'
            elif signal == 'bullish_reversal' and rsi.iloc[-1] < 25:
                strength = 'strong_rsi'
            elif signal == 'bearish_reversal' and rsi.iloc[-1] > 75:
                strength = 'strong_rsi'

        direction = 'up' if 'bullish' in signal else 'down' if 'bearish' in signal else None
        result = {
            'signal': signal,
            'strength': strength,
            'current_price': current_price,
            'middle_band': middle_band.iloc[-1],
            'bandwidth_pct': round(conditions['bandwidth_pct'], 2),
            'recommend_entry': (close.iloc[-1] + close.iloc[-2]) / 2
        }
        return BollingerBandsStrategy._apply_pattern_boost(result, patterns or [])

    @staticmethod
    def risk_management(close: pd.Series,
                        middle_band: pd.Series,
                        signal: str) -> Dict[str, float]:
        current_price = close.iloc[-1]
        current_middle = middle_band.iloc[-1]

        if signal in ['bullish_reversal', 'bullish_breakout']:
            stop_loss = min(current_middle, close.iloc[-3:].min())
            take_profit = current_price + (current_price - stop_loss) * 1.5
        elif signal in ['bearish_reversal', 'bearish_breakout']:
            stop_loss = max(current_middle, close.iloc[-3:].max())
            take_profit = current_price - (stop_loss - current_price) * 1.5
        else:
            stop_loss = take_profit = None

        return {
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'take_profit': round(take_profit, 2) if take_profit else None,
            'risk_reward_ratio': 1.5
        }

    @staticmethod
    def full_analysis(close: pd.Series,
                      high: pd.Series = None,
                      low: pd.Series = None,
                      volume: pd.Series = None,
                      rsi: pd.Series = None,
                      patterns: list = None,
                      period: int = 20,
                      std_dev: float = 2.0,
                      ma_type: str = 'sma') -> Dict[str, Union[Dict, str]]:
        bands = BollingerBandsStrategy.calculate_bollinger_bands(close, period, std_dev, ma_type)
        conditions = BollingerBandsStrategy.identify_conditions(
            close, bands['upper_band'], bands['middle_band'], bands['lower_band'], bands['bandwidth'])
        signals = BollingerBandsStrategy.generate_signals(
            close, bands['upper_band'], bands['middle_band'], bands['lower_band'], conditions, volume, rsi, patterns)
        risk = BollingerBandsStrategy.risk_management(close, bands['middle_band'], signals['signal'])

        return {
            'bands': {
                'upper': round(bands['upper_band'].iloc[-1], 5),
                'middle': round(bands['middle_band'].iloc[-1], 5),
                'lower': round(bands['lower_band'].iloc[-1], 5),
                'bandwidth': round(bands['bandwidth'].iloc[-1], 2),
                'std_dev': round(bands['std_dev'].iloc[-1], 5)
            },
            'conditions': conditions,
            'signals': signals,
            'risk_management': risk
        }

    @staticmethod
    def optimize_parameters(close: pd.Series,
                            min_period: int = 10,
                            max_period: int = 50,
                            min_std: float = 1.5,
                            max_std: float = 3.0) -> Dict[str, float]:
        """
        Otimiza os parâmetros (período e desvio padrão) para:
        - Maximizar o número de toques nas bandas
        - Minimizar falsos breakouts
        """
        def objective(params):
            period, std = params
            period = int(round(period))
            bands = BollingerBandsStrategy.calculate_bollinger_bands(close, period, std)
            touches = 0
            for i in range(1, len(close)):
                if (close.iloc[i] >= bands['upper_band'].iloc[i] and close.iloc[i - 1] < bands['upper_band'].iloc[i - 1]) or \
                   (close.iloc[i] <= bands['lower_band'].iloc[i] and close.iloc[i - 1] > bands['lower_band'].iloc[i - 1]):
                    touches += 1
            penalty = abs(period - 20) * 0.1 + abs(std - 2.0) * 0.5
            return -(touches - penalty)

        from scipy.optimize import differential_evolution
        bounds = [(min_period, max_period), (min_std, max_std)]
        result = differential_evolution(objective, bounds)

        return {
            'optimal_period': int(round(result.x[0])),
            'optimal_std_dev': round(result.x[1], 2)
                                      }
                                
