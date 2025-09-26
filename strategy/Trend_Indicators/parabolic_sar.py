import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class ParabolicSARStrategy:
    """
    Implementação profissional do Parabolic SAR com:
    - Cálculo preciso do SAR com ajuste dinâmico de fator
    - Identificação de reversões de tendência
    - Confirmação com volume e ADX
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_sar(high: pd.Series,
                    low: pd.Series,
                    close: pd.Series,
                    acceleration_factor: float = 0.02,
                    max_acceleration: float = 0.2) -> Dict[str, pd.Series]:
        """
        Calcula o Parabolic SAR:
        - Inicia com tendência de alta se close[1] > close[0]
        - SAR = Prior SAR + AF * (EP - Prior SAR)
        - AF aumenta em incrementos até atingir o máximo
        """
        sar = pd.Series(np.nan, index=close.index)
        trend = pd.Series(0, index=close.index)
        ep = pd.Series(np.nan, index=close.index)  # Extreme Point
        af = pd.Series(np.nan, index=close.index)  # Acceleration Factor
        
        # Determina a tendência inicial
        initial_trend = 1 if close.iloc[1] > close.iloc[0] else -1
        sar.iloc[1] = high.iloc[0] if initial_trend == 1 else low.iloc[0]
        trend.iloc[1] = initial_trend
        ep.iloc[1] = high.iloc[1] if initial_trend == 1 else low.iloc[1]
        af.iloc[1] = acceleration_factor
        
        for i in range(2, len(close)):
            # Tendência anterior
            prev_trend = trend.iloc[i-1]
            prev_sar = sar.iloc[i-1]
            prev_ep = ep.iloc[i-1]
            prev_af = af.iloc[i-1]
            
            # SAR provisório
            current_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            
            # Verifica reversão
            if prev_trend == 1:
                current_sar = min(current_sar, low.iloc[i-1], low.iloc[i-2])
                if low.iloc[i] < current_sar:
                    # Reversão para baixa
                    trend.iloc[i] = -1
                    sar.iloc[i] = max(high.iloc[i-1], high.iloc[i-2])
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = acceleration_factor
                else:
                    # Continua tendência de alta
                    trend.iloc[i] = 1
                    sar.iloc[i] = current_sar
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(prev_af + acceleration_factor, max_acceleration)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
            else:
                current_sar = max(current_sar, high.iloc[i-1], high.iloc[i-2])
                if high.iloc[i] > current_sar:
                    # Reversão para alta
                    trend.iloc[i] = 1
                    sar.iloc[i] = min(low.iloc[i-1], low.iloc[i-2])
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = acceleration_factor
                else:
                    # Continua tendência de baixa
                    trend.iloc[i] = -1
                    sar.iloc[i] = current_sar
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(prev_af + acceleration_factor, max_acceleration)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
        
        return {
            'sar': sar,
            'trend': trend,
            'ep': ep,
            'af': af
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                       sar: pd.Series,
                       trend: pd.Series,
                       adx: pd.Series = None,
                       volume: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento do preço com o SAR
        - Força da tendência (ADX)
        - Confirmação de volume
        """
        current_close = close.iloc[-1]
        current_sar = sar.iloc[-1]
        current_trend = trend.iloc[-1]
        previous_trend = trend.iloc[-2]
        
        # Sinal básico
        signal = None
        if current_trend == 1 and previous_trend == -1:
            signal = 'bullish_reversal'
        elif current_trend == -1 and previous_trend == 1:
            signal = 'bearish_reversal'
        
        # Força do sinal
        strength = 'weak'
        if signal:
            if adx is not None and adx.iloc[-1] > 25:
                strength = 'strong_trend'
            elif volume is not None and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                strength = 'strong_volume'
            elif abs(current_close - current_sar) / current_close > 0.01:
                strength = 'strong_distance'
        
        return {
            'signal': signal,
            'strength': strength,
            'current_sar': current_sar,
            'current_trend': 'up' if current_trend == 1 else 'down',
            'distance_pct': abs(current_close - current_sar) / current_close * 100
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     volume: pd.Series = None,
                     acceleration_factor: float = 0.02,
                     max_acceleration: float = 0.2,
                     adx_period: int = 14) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do Parabolic SAR:
        - Valores do SAR e tendência
        - Sinais de trading
        - Gerenciamento de risco
        """
        # Calcula SAR
        sar_data = ParabolicSARStrategy.calculate_sar(
            high, low, close, acceleration_factor, max_acceleration
        )
        
        # Calcula ADX se volume for fornecido
        adx = None
        if volume is not None:
            adx = ParabolicSARStrategy.calculate_adx(high, low, close, adx_period)
        
        # Gera sinais
        signals = ParabolicSARStrategy.generate_signals(
            close, sar_data['sar'], sar_data['trend'], adx, volume
        )
        
        return {
            'sar': {
                'current': round(sar_data['sar'].iloc[-1], 5),
                'trend': sar_data['trend'].iloc[-1],
                'acceleration': round(sar_data['af'].iloc[-1], 3)
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(
                    sar_data['sar'].iloc[-1], 5
                ),
                'take_profit': round(
                    close.iloc[-1] + (close.iloc[-1] - sar_data['sar'].iloc[-1]) if signals['signal'] == 'bullish_reversal'
                    else close.iloc[-1] - (sar_data['sar'].iloc[-1] - close.iloc[-1]), 5
                ),
                'risk_reward_ratio': '1:2'  # Padrão institucional
            }
        }

    @staticmethod
    def calculate_adx(high: pd.Series,
                    low: pd.Series,
                    close: pd.Series,
                    period: int = 14) -> pd.Series:
        """
        Calcula o ADX para confirmação de tendência
        """
        # Cálculo do Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Suavização
        tr = pd.DataFrame({
            'high_low': high - low,
            'high_close': abs(high - close.shift(1)),
            'low_close': abs(low - close.shift(1))
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx
