import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class KeltnerStrategy:
    """
    Implementação profissional dos Keltner Channels com:
    - Cálculo dos canais usando EMA + ATR
    - Identificação de squeezes (compressão de volatilidade)
    - Breakouts com confirmação de volume
    - Reversões nos limites do canal
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_keltner(high: pd.Series,
                         low: pd.Series,
                         close: pd.Series,
                         ema_period: int = 20,
                         atr_period: int = 10,
                         multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calcula os Keltner Channels:
        - Linha Central: EMA(close, ema_period)
        - Bandas: EMA ± (ATR(atr_period) * multiplier)
        """
        ema = close.ewm(span=ema_period, adjust=False).mean()
        
        # Cálculo do ATR
        tr = pd.DataFrame({
            'high_low': high - low,
            'high_close': abs(high - close.shift(1)),
            'low_close': abs(low - close.shift(1))
        }).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        upper_band = ema + (atr * multiplier)
        lower_band = ema - (atr * multiplier)
        
        return {
            'ema': ema,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr
        }

    @staticmethod
    def identify_squeeze(close: pd.Series,
                         upper_band: pd.Series,
                         lower_band: pd.Series,
                         lookback: int = 20) -> bool:
        """
        Identifica compressão de volatilidade (squeeze):
        - Quando o preço fica contido em uma faixa estreita entre os canais.
        - Condição: Largura dos canais < 50% da média móvel da largura.
        """
        channel_width = upper_band - lower_band
        avg_width = channel_width.rolling(window=lookback).mean()
        return channel_width.iloc[-1] < 0.5 * avg_width.iloc[-1]

    @staticmethod
    def generate_signals(high: pd.Series,
                        low: pd.Series,
                        close: pd.Series,
                        ema: pd.Series,
                        upper_band: pd.Series,
                        lower_band: pd.Series,
                        volume: pd.Series = None,
                        atr: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Breakout dos canais (com confirmação de volume)
        - Reversão nos limites do canal
        - Squeeze (potencial explosão de volatilidade)
        """
        current_close = close.iloc[-1]
        previous_close = close.iloc[-2]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        signal = None
        confirmation = None
        
        # Breakout de alta
        if current_close > current_upper and previous_close <= current_upper:
            signal = 'breakout_bullish'
            if volume is not None and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
        
        # Breakout de baixa
        elif current_close < current_lower and previous_close >= current_lower:
            signal = 'breakout_bearish'
            if volume is not None and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
        
        # Reversão na banda superior (bearish rejection)
        elif (high.iloc[-1] >= current_upper and 
              close.iloc[-1] < (high.iloc[-1] + low.iloc[-1]) / 2):
            signal = 'reversal_bearish'
        
        # Reversão na banda inferior (bullish rejection)
        elif (low.iloc[-1] <= current_lower and 
              close.iloc[-1] > (high.iloc[-1] + low.iloc[-1]) / 2):
            signal = 'reversal_bullish'
        
        # Squeeze (potencial breakout futuro)
        squeeze = KeltnerStrategy.identify_squeeze(close, upper_band, lower_band)
        if squeeze:
            signal = 'squeeze'
        
        return {
            'signal': signal,
            'confirmation': confirmation,
            'current_close': current_close,
            'upper_band': current_upper,
            'lower_band': current_lower,
            'squeeze': squeeze
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series = None,
                      ema_period: int = 20,
                      atr_period: int = 10,
                      multiplier: float = 2.0) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa dos Keltner Channels:
        - Componentes dos canais (EMA, bandas, ATR)
        - Sinais de trading
        - Squeeze detection
        - Gerenciamento de risco
        """
        keltner = KeltnerStrategy.calculate_keltner(
            high, low, close, ema_period, atr_period, multiplier
        )
        
        signals = KeltnerStrategy.generate_signals(
            high, low, close,
            keltner['ema'],
            keltner['upper_band'],
            keltner['lower_band'],
            volume,
            keltner['atr']
        )
        
        # Gerenciamento de risco baseado no ATR
        stop_loss = None
        take_profit = None
        
        if signals['signal'] == 'breakout_bullish':
            stop_loss = keltner['ema'].iloc[-1]
            take_profit = close.iloc[-1] + (2 * keltner['atr'].iloc[-1])
        elif signals['signal'] == 'breakout_bearish':
            stop_loss = keltner['ema'].iloc[-1]
            take_profit = close.iloc[-1] - (2 * keltner['atr'].iloc[-1])
        
        return {
            'components': {
                'ema': round(keltner['ema'].iloc[-1], 5),
                'upper_band': round(keltner['upper_band'].iloc[-1], 5),
                'lower_band': round(keltner['lower_band'].iloc[-1], 5),
                'atr': round(keltner['atr'].iloc[-1], 5),
                'channel_width': round(keltner['upper_band'].iloc[-1] - keltner['lower_band'].iloc[-1], 5)
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(stop_loss, 5) if stop_loss else None,
                'take_profit': round(take_profit, 5) if take_profit else None,
                'risk_reward_ratio': round(abs(take_profit - close.iloc[-1]) / abs(stop_loss - close.iloc[-1]), 2) 
                                if stop_loss and take_profit else None
            }
        }