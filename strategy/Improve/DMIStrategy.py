import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class DMIStrategy:
    """
    Implementação profissional do DMI (Directional Movement Index) com:
    - Cálculo preciso do +DI, -DI e ADX
    - Identificação de tendências fortes e fracas
    - Sinais de crossover (+DI/-DI)
    - Filtro de confirmação com volume
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_dmi(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = 14) -> Dict[str, pd.Series]:
        """
        Calcula os componentes do DMI:
        - +DI (Positive Directional Indicator)
        - -DI (Negative Directional Indicator)
        - ADX (Average Directional Movement Index)
        """
        # Cálculo do Directional Movement (+DM e -DM)
        up_move = high.diff()
        down_move = low.diff().abs()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Suavização com média móvel
        plus_dm_smoothed = pd.Series(plus_dm).rolling(window=period).mean()
        minus_dm_smoothed = pd.Series(minus_dm).rolling(window=period).mean()
        
        # True Range (TR)
        tr = pd.DataFrame({
            'high_low': high - low,
            'high_close': abs(high - close.shift(1)),
            'low_close': abs(low - close.shift(1))
        }).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Cálculo do +DI e -DI
        plus_di = 100 * (plus_dm_smoothed / atr)
        minus_di = 100 * (minus_dm_smoothed / atr)
        
        # Cálculo do ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return {
            'plus_di': plus_di,
            'minus_di': minus_di,
            'adx': adx,
            'atr': atr
        }

    @staticmethod
    def generate_signals(plus_di: pd.Series,
                        minus_di: pd.Series,
                        adx: pd.Series,
                        close: pd.Series,
                        volume: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Crossover +DI/-DI
        - Força da tendência (ADX > 25)
        - Confirmação de volume
        """
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        previous_plus_di = plus_di.iloc[-2]
        previous_minus_di = minus_di.iloc[-2]
        current_adx = adx.iloc[-1]
        
        # Sinal de crossover
        signal = None
        if current_plus_di > current_minus_di and previous_plus_di <= previous_minus_di:
            signal = 'bullish_crossover'
        elif current_plus_di < current_minus_di and previous_plus_di >= previous_minus_di:
            signal = 'bearish_crossover'
        
        # Força da tendência (ADX > 25 = tendência forte)
        trend_strength = 'strong' if current_adx > 25 else 'weak'
        
        # Confirmação de volume
        confirmation = None
        if volume is not None:
            if signal == 'bullish_crossover' and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
            elif signal == 'bearish_crossover' and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
        
        return {
            'signal': signal,
            'trend_strength': trend_strength,
            'confirmation': confirmation,
            'plus_di': current_plus_di,
            'minus_di': current_minus_di,
            'adx': current_adx
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series = None,
                      period: int = 14) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do DMI:
        - Componentes (+DI, -DI, ADX)
        - Sinais de trading
        - Força da tendência
        - Gerenciamento de risco
        """
        dmi = DMIStrategy.calculate_dmi(high, low, close, period)
        signals = DMIStrategy.generate_signals(
            dmi['plus_di'],
            dmi['minus_di'],
            dmi['adx'],
            close,
            volume
        )
        
        return {
            'components': {
                'plus_di': round(dmi['plus_di'].iloc[-1], 2),
                'minus_di': round(dmi['minus_di'].iloc[-1], 2),
                'adx': round(dmi['adx'].iloc[-1], 2),
                'atr': round(dmi['atr'].iloc[-1], 2)
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(close.iloc[-1] * 0.98 if signals['signal'] == 'bullish_crossover' else close.iloc[-1] * 1.02, 2),
                'take_profit': round(close.iloc[-1] * 1.04 if signals['signal'] == 'bullish_crossover' else close.iloc[-1] * 0.96, 2)
            }
        }