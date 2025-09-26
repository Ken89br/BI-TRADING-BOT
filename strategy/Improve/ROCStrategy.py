import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class ROCStrategy:
    """
    Implementação profissional do ROC (Rate of Change) com:
    - Cálculo do ROC padrão e suavizado
    - Identificação de divergências (regular e oculta)
    - Zonas de overbought/oversold dinâmicas
    - Confirmação com volume e volatilidade
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_roc(close: pd.Series, 
                     period: int = 12,
                     smooth_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calcula o ROC e sua versão suavizada:
        ROC = (Preço Atual / Preço n períodos atrás - 1) * 100
        """
        roc = (close / close.shift(period) - 1) * 100
        roc_smoothed = roc.rolling(window=smooth_period).mean()
        
        return {
            'roc': roc,
            'roc_smoothed': roc_smoothed,
            'signal_line': roc_smoothed.rolling(window=smooth_period).mean()  # Linha de sinal
        }

    @staticmethod
    def identify_divergence(close: pd.Series,
                          roc: pd.Series,
                          lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e ROC:
        - Regular (reversão)
        - Oculta (continuação)
        """
        # Encontra máximos/mínimos locais
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        roc_peaks = roc.rolling(window=lookback, center=True).max() == roc
        roc_troughs = roc.rolling(window=lookback, center=True).min() == roc
        
        # Divergência regular
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(roc_peaks):
            last_price_peak = close[price_peaks].index[-1]
            last_roc_peak = roc[roc_peaks].index[-1]
            if (close[last_price_peak] > close[price_peaks].iloc[-2] and
                roc[last_roc_peak] < roc[roc_peaks].iloc[-2]):
                bearish_div = True
                
        if any(price_troughs) and any(roc_troughs):
            last_price_trough = close[price_troughs].index[-1]
            last_roc_trough = roc[roc_troughs].index[-1]
            if (close[last_price_trough] < close[price_troughs].iloc[-2] and
                roc[last_roc_trough] > roc[roc_troughs].iloc[-2]):
                bullish_div = True
        
        # Divergência oculta
        hidden_bearish = False
        hidden_bullish = False
        
        if any(price_peaks) and any(roc_peaks):
            if (close[price_peaks].iloc[-1] < close[price_peaks].iloc[-2] and
                roc[roc_peaks].iloc[-1] > roc[roc_peaks].iloc[-2]):
                hidden_bearish = True
                
        if any(price_troughs) and any(roc_troughs):
            if (close[price_troughs].iloc[-1] > close[price_troughs].iloc[-2] and
                roc[roc_troughs].iloc[-1] < roc[roc_troughs].iloc[-2]):
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
                       roc: pd.Series,
                       roc_smoothed: pd.Series,
                       signal_line: pd.Series,
                       volume: pd.Series = None,
                       atr: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento do ROC com sua linha de sinal
        - Divergências
        - Níveis extremos dinâmicos
        """
        current_roc = roc.iloc[-1]
        previous_roc = roc.iloc[-2]
        current_signal = signal_line.iloc[-1]
        previous_signal = signal_line.iloc[-2]
        
        # Sinais de cruzamento
        signal = None
        if current_roc > current_signal and previous_roc <= previous_signal:
            signal = 'bullish_crossover'
        elif current_roc < current_signal and previous_roc >= previous_signal:
            signal = 'bearish_crossover'
        
        # Divergências (sobrescrevem cruzamentos)
        divergence = ROCStrategy.identify_divergence(close, roc)
        if divergence['regular']['bearish']:
            signal = 'bearish_divergence'
        elif divergence['regular']['bullish']:
            signal = 'bullish_divergence'
        
        # Níveis dinâmicos baseados em ATR
        overbought = 10 + (atr.iloc[-1]/close.iloc[-1]*100 if atr is not None else 0)
        oversold = -10 - (atr.iloc[-1]/close.iloc[-1]*100 if atr is not None else 0)
        
        # Confirmação de volume
        confirmation = None
        if volume is not None and signal:
            if volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
        
        return {
            'signal': signal,
            'confirmation': confirmation,
            'current_roc': current_roc,
            'roc_smoothed': roc_smoothed.iloc[-1],
            'overbought_level': overbought,
            'oversold_level': oversold,
            'divergence': divergence
        }

    @staticmethod
    def full_analysis(close: pd.Series,
                     high: pd.Series = None,
                     low: pd.Series = None,
                     volume: pd.Series = None,
                     roc_period: int = 12,
                     smooth_period: int = 3,
                     atr_period: int = 14) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do ROC:
        - Valores do ROC e linha de sinal
        - Sinais de trading
        - Divergências
        - Gerenciamento de risco
        """
        # Calcula ROC
        roc_data = ROCStrategy.calculate_roc(close, roc_period, smooth_period)
        
        # Calcula ATR se fornecido high/low
        atr = None
        if high is not None and low is not None:
            tr = pd.DataFrame({
                'high_low': high - low,
                'high_close': abs(high - close.shift(1)),
                'low_close': abs(low - close.shift(1))
            }).max(axis=1)
            atr = tr.rolling(window=atr_period).mean()
        
        # Gera sinais
        signals = ROCStrategy.generate_signals(
            close,
            roc_data['roc'],
            roc_data['roc_smoothed'],
            roc_data['signal_line'],
            volume,
            atr
        )
        
        return {
            'roc': {
                'current': round(roc_data['roc'].iloc[-1], 2),
                'smoothed': round(roc_data['roc_smoothed'].iloc[-1], 2),
                'signal_line': round(roc_data['signal_line'].iloc[-1], 2)
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(close.iloc[-1] * 0.98 if signals['signal'] in ['bullish_crossover', 'bullish_divergence'] 
                          else close.iloc[-1] * 1.02, 2),
                'take_profit': round(close.iloc[-1] * 1.03 if signals['signal'] in ['bullish_crossover', 'bullish_divergence']
                           else close.iloc[-1] * 0.97, 2)
            }
        }