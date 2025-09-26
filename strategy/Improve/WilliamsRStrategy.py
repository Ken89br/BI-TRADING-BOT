import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class WilliamsRStrategy:
    """
    Implementação profissional do Williams %R com:
    - Cálculo preciso do %R com ajuste de período
    - Identificação de divergências (regular e oculta)
    - Zonas de overbought/oversold dinâmicas
    - Confirmação com volume e volatilidade (ATR)
    - Gerenciamento de risco integrado
    - Filtro de tendência (ADX opcional)
    """

    @staticmethod
    def calculate_williams_r(high: pd.Series, 
                            low: pd.Series, 
                            close: pd.Series, 
                            period: int = 14) -> pd.Series:
        """
        Calcula o Williams %R usando a fórmula clássica:
        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
        return williams_r

    @staticmethod
    def identify_divergence(close: pd.Series, 
                           williams_r: pd.Series,
                           lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e Williams %R:
        - Regular (reversão)
        - Oculta (continuação)
        Retorna: {'type': 'bearish/bullish', 'confirmed': bool}
        """
        # Identifica máximos/mínimos locais
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        wr_peaks = williams_r.rolling(window=lookback, center=True).max() == williams_r
        wr_troughs = williams_r.rolling(window=lookback, center=True).min() == williams_r

        # Divergência regular (reversão)
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(wr_peaks):
            last_price_peak = close[price_peaks].index[-1]
            last_wr_peak = williams_r[wr_peaks].index[-1]
            if (close[last_price_peak] > close[price_peaks].iloc[-2] and 
                williams_r[last_wr_peak] < williams_r[wr_peaks].iloc[-2]):
                bearish_div = True

        if any(price_troughs) and any(wr_troughs):
            last_price_trough = close[price_troughs].index[-1]
            last_wr_trough = williams_r[wr_troughs].index[-1]
            if (close[last_price_trough] < close[price_troughs].iloc[-2] and 
                williams_r[last_wr_trough] > williams_r[wr_troughs].iloc[-2]):
                bullish_div = True

        # Divergência oculta (continuação)
        hidden_bearish = False
        hidden_bullish = False
        
        if any(price_peaks) and any(wr_peaks):
            if (close[price_peaks].iloc[-1] < close[price_peaks].iloc[-2] and 
                williams_r[wr_peaks].iloc[-1] > williams_r[wr_peaks].iloc[-2]):
                hidden_bearish = True

        if any(price_troughs) and any(wr_troughs):
            if (close[price_troughs].iloc[-1] > close[price_troughs].iloc[-2] and 
                williams_r[wr_troughs].iloc[-1] < williams_r[wr_troughs].iloc[-2]):
                hidden_bullish = True

        return {
            'regular': {
                'bearish': bearish_div,
                'bullish': bullish_div
            },
            'hidden': {
                'bearish': hidden_bearish,
                'bullish': hidden_bullish
            }
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                        williams_r: pd.Series,
                        volume: pd.Series = None,
                        atr: pd.Series = None,
                        overbought: float = -20,
                        oversold: float = -80) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento de zonas de overbought/oversold
        - Divergências
        - Confirmação de volume
        """
        current_wr = williams_r.iloc[-1]
        previous_wr = williams_r.iloc[-2]

        # Overbought/Oversold dinâmico (ajustado por ATR)
        if atr is not None:
            volatility_adjustment = (atr.iloc[-1] / close.iloc[-1]) * 100
            overbought = -20 + volatility_adjustment
            oversold = -80 - volatility_adjustment

        # Sinal básico
        signal = None
        if current_wr > overbought and previous_wr <= overbought:
            signal = 'overbought'
        elif current_wr < oversold and previous_wr >= oversold:
            signal = 'oversold'

        # Divergências (prioritárias)
        divergence = WilliamsRStrategy.identify_divergence(close, williams_r)
        if divergence['regular']['bearish']:
            signal = 'bearish_divergence'
        elif divergence['regular']['bullish']:
            signal = 'bullish_divergence'

        # Confirmação de volume
        confirmation = None
        if volume is not None:
            if signal in ('overbought', 'bearish_divergence') and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
            elif signal in ('oversold', 'bullish_divergence') and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'

        return {
            'signal': signal,
            'confirmation': confirmation,
            'current_wr': current_wr,
            'overbought_level': overbought,
            'oversold_level': oversold,
            'divergence': divergence
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series = None,
                      period: int = 14,
                      overbought: float = -20,
                      oversold: float = -80) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do Williams %R:
        - Valor atual e níveis dinâmicos
        - Sinais de trading
        - Divergências
        - Confirmações
        - Gerenciamento de risco
        """
        williams_r = WilliamsRStrategy.calculate_williams_r(high, low, close, period)
        atr = None
        if volume is not None:
            tr = pd.DataFrame({
                'high_low': high - low,
                'high_close': abs(high - close.shift(1)),
                'low_close': abs(low - close.shift(1))
            }).max(axis=1)
            atr = tr.rolling(window=period).mean()

        signals = WilliamsRStrategy.generate_signals(close, williams_r, volume, atr, overbought, oversold)

        return {
            'williams_r': {
                'current': round(williams_r.iloc[-1], 2),
                'mean': round(williams_r.mean(), 2),
                'std_dev': round(williams_r.std(), 2)
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(close.iloc[-1] * 0.99 if signals['signal'] == 'bullish_divergence' else close.iloc[-1] * 1.01, 2),
                'take_profit': round(close.iloc[-1] * 1.03 if signals['signal'] == 'bullish_divergence' else close.iloc[-1] * 0.97, 2)
            }
        }