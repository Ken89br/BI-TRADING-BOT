import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class CCIStrategy:
    """
    Implementação profissional do CCI (Commodity Channel Index) com:
    - Cálculo do CCI padrão e adaptativo
    - Identificação de divergências (regular e oculta)
    - Sinais de overbought/oversold dinâmicos
    - Confirmação com volume e volatilidade (ATR)
    - Gerenciamento de risco automático
    """

    @staticmethod
    def calculate_cci(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     period: int = 20,
                     constant: float = 0.015) -> pd.Series:
        """
        Calcula o CCI usando a fórmula clássica:
        CCI = (TP - SMA(TP, period)) / (constant * Mean Deviation)
        Onde TP (Typical Price) = (High + Low + Close) / 3
        """
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mean_deviation = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        
        cci = (tp - sma_tp) / (constant * mean_deviation)
        return cci

    @staticmethod
    def identify_divergence(close: pd.Series,
                           cci: pd.Series,
                           lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e CCI:
        - Regular (reversão)
        - Oculta (continuação)
        Retorna: {'type': 'bearish/bullish', 'confirmed': bool}
        """
        # Encontra máximos/mínimos locais no preço e CCI
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        cci_peaks = cci.rolling(window=lookback, center=True).max() == cci
        cci_troughs = cci.rolling(window=lookback, center=True).min() == cci
        
        # Verifica divergências de reversão (regular)
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(cci_peaks):
            last_price_peak = close[price_peaks].index[-1]
            last_cci_peak = cci[cci_peaks].index[-1]
            if (close[last_price_peak] > close[price_peaks].iloc[-2] and
                cci[last_cci_peak] < cci[cci_peaks].iloc[-2]):
                bearish_div = True
        
        if any(price_troughs) and any(cci_troughs):
            last_price_trough = close[price_troughs].index[-1]
            last_cci_trough = cci[cci_troughs].index[-1]
            if (close[last_price_trough] < close[price_troughs].iloc[-2] and
                cci[last_cci_trough] > cci[cci_troughs].iloc[-2]):
                bullish_div = True
        
        # Verifica divergências de continuação (oculta)
        hidden_bearish_div = False
        hidden_bullish_div = False
        
        if any(price_peaks) and any(cci_peaks):
            if (close[price_peaks].iloc[-1] < close[price_peaks].iloc[-2] and
                cci[cci_peaks].iloc[-1] > cci[cci_peaks].iloc[-2]):
                hidden_bearish_div = True
        
        if any(price_troughs) and any(cci_troughs):
            if (close[price_troughs].iloc[-1] > close[price_troughs].iloc[-2] and
                cci[cci_troughs].iloc[-1] < cci[cci_troughs].iloc[-2]):
                hidden_bullish_div = True
        
        return {
            'regular': {
                'bearish': bearish_div,
                'bullish': bullish_div
            },
            'hidden': {
                'bearish': hidden_bearish_div,
                'bullish': hidden_bullish_div
            }
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                        cci: pd.Series,
                        volume: pd.Series = None,
                        atr: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento de linhas de sinal (±100)
        - Divergências
        - Overbought/Oversold dinâmico
        """
        current_cci = cci.iloc[-1]
        previous_cci = cci.iloc[-2]
        
        # Overbought/Oversold dinâmico (baseado no ATR)
        overbought = 100 + (atr.iloc[-1] / close.iloc[-1] * 100 if atr is not None else 0)
        oversold = -100 - (atr.iloc[-1] / close.iloc[-1] * 100 if atr is not None else 0)
        
        # Sinal básico
        signal = None
        if current_cci > overbought and previous_cci <= overbought:
            signal = 'overbought'
        elif current_cci < oversold and previous_cci >= oversold:
            signal = 'oversold'
        
        # Divergências (prioritárias sobre overbought/oversold)
        divergence = CCIStrategy.identify_divergence(close, cci)
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
            'current_cci': current_cci,
            'overbought_level': overbought,
            'oversold_level': oversold,
            'divergence': divergence
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series = None,
                      period: int = 20,
                      constant: float = 0.015) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do CCI:
        - Valor atual e níveis dinâmicos
        - Sinais de trading
        - Divergências
        - Confirmações
        """
        cci = CCIStrategy.calculate_cci(high, low, close, period, constant)
        atr = None
        if volume is not None:
            tr = pd.DataFrame({
                'high_low': high - low,
                'high_close': abs(high - close.shift(1)),
                'low_close': abs(low - close.shift(1))
            }).max(axis=1)
            atr = tr.rolling(window=period).mean()
        
        signals = CCIStrategy.generate_signals(close, cci, volume, atr)
        
        return {
            'cci': {
                'current': round(cci.iloc[-1], 2),
                'mean': round(cci.mean(), 2),
                'std_dev': round(cci.std(), 2)
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(close.iloc[-1] * 0.99 if signals['signal'] == 'bullish_divergence' else close.iloc[-1] * 1.01, 2),
                'take_profit': round(close.iloc[-1] * 1.03 if signals['signal'] == 'bullish_divergence' else close.iloc[-1] * 0.97, 2)
            }
        }