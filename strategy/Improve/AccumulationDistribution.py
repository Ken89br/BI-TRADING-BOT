import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class AccumulationDistribution:
    """
    Implementação profissional do Accumulation/Distribution (A/D) com:
    - Cálculo preciso do A/D Line
    - Identificação de divergências (regular e oculta)
    - Confirmação com volume e price action
    - Análise de fluxo de capital
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_ad_line(high: pd.Series,
                         low: pd.Series,
                         close: pd.Series,
                         volume: pd.Series) -> pd.Series:
        """
        Calcula a linha Accumulation/Distribution:
        A/D = ((Close - Low) - (High - Close)) / (High - Low) * Volume
        """
        money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, 0.0001)
        money_flow_volume = money_flow_multiplier * volume
        ad_line = money_flow_volume.cumsum()
        return ad_line

    @staticmethod
    def identify_divergence(close: pd.Series,
                           ad_line: pd.Series,
                           lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e A/D Line:
        - Regular (reversão)
        - Oculta (continuação)
        """
        # Encontra máximos/mínimos locais
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        ad_peaks = ad_line.rolling(window=lookback, center=True).max() == ad_line
        ad_troughs = ad_line.rolling(window=lookback, center=True).min() == ad_line
        
        # Divergência regular
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(ad_peaks):
            if (close[price_peaks].iloc[-1] > close[price_peaks].iloc[-2] and
                ad_line[ad_peaks].iloc[-1] < ad_line[ad_peaks].iloc[-2]):
                bearish_div = True
                
        if any(price_troughs) and any(ad_troughs):
            if (close[price_troughs].iloc[-1] < close[price_troughs].iloc[-2] and
                ad_line[ad_troughs].iloc[-1] > ad_line[ad_troughs].iloc[-2]):
                bullish_div = True
                
        # Divergência oculta
        hidden_bearish = False
        hidden_bullish = False
        
        if any(price_peaks) and any(ad_peaks):
            if (close[price_peaks].iloc[-1] < close[price_peaks].iloc[-2] and
                ad_line[ad_peaks].iloc[-1] > ad_line[ad_peaks].iloc[-2]):
                hidden_bearish = True
                
        if any(price_troughs) and any(ad_troughs):
            if (close[price_troughs].iloc[-1] > close[price_troughs].iloc[-2] and
                ad_line[ad_troughs].iloc[-1] < ad_line[ad_troughs].iloc[-2]):
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
                        ad_line: pd.Series,
                        volume: pd.Series,
                        sma_period: int = 20) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Divergências
        - Cruzamento da linha A/D com sua média
        - Confirmação de volume
        """
        signals = []
        ad_sma = ad_line.rolling(window=sma_period).mean()
        
        # 1. Divergências (prioridade máxima)
        divergence = AccumulationDistribution.identify_divergence(close, ad_line)
        if divergence['regular']['bearish']:
            signals.append({
                'type': 'bearish_divergence',
                'price': close.iloc[-1],
                'confidence': 'high'
            })
        elif divergence['regular']['bullish']:
            signals.append({
                'type': 'bullish_divergence',
                'price': close.iloc[-1],
                'confidence': 'high'
            })
            
        # 2. Cruzamento A/D Line x SMA (confirmado por volume)
        if (ad_line.iloc[-1] > ad_sma.iloc[-1] and 
            ad_line.iloc[-2] <= ad_sma.iloc[-2] and
            volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]):
            signals.append({
                'type': 'ad_cross_bullish',
                'price': close.iloc[-1],
                'confidence': 'medium'
            })
        elif (ad_line.iloc[-1] < ad_sma.iloc[-1] and 
              ad_line.iloc[-2] >= ad_sma.iloc[-2] and
              volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]):
            signals.append({
                'type': 'ad_cross_bearish',
                'price': close.iloc[-1],
                'confidence': 'medium'
            })
            
        # 3. Confirmação de tendência (A/D e preço na mesma direção)
        if (close.iloc[-1] > close.iloc[-2] and 
            ad_line.iloc[-1] > ad_line.iloc[-2] and
            volume.iloc[-1] > volume.iloc[-2]):
            signals.append({
                'type': 'uptrend_confirmation',
                'price': close.iloc[-1],
                'confidence': 'high'
            })
        elif (close.iloc[-1] < close.iloc[-2] and 
              ad_line.iloc[-1] < ad_line.iloc[-2] and
              volume.iloc[-1] > volume.iloc[-2]):
            signals.append({
                'type': 'downtrend_confirmation',
                'price': close.iloc[-1],
                'confidence': 'high'
            })
            
        return {
            'signals': signals,
            'current_ad': ad_line.iloc[-1],
            'ad_sma': ad_sma.iloc[-1],
            'divergence': divergence
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series,
                      sma_period: int = 20,
                      lookback: int = 14) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do A/D Line:
        - Valor atual e SMA
        - Sinais de trading
        - Divergências
        - Gerenciamento de risco
        """
        ad_line = AccumulationDistribution.calculate_ad_line(high, low, close, volume)
        signals = AccumulationDistribution.generate_signals(close, ad_line, volume, sma_period)
        
        # Gerenciamento de Risco
        risk_params = {}
        if signals['signals']:
            last_signal = signals['signals'][-1]
            atr = high.rolling(14).max() - low.rolling(14).min()
            
            if 'bullish' in last_signal['type']:
                risk_params = {
                    'stop_loss': close.iloc[-1] - atr.iloc[-1],
                    'take_profit': close.iloc[-1] + (2 * atr.iloc[-1]),
                    'risk_reward': 2.0
                }
            elif 'bearish' in last_signal['type']:
                risk_params = {
                    'stop_loss': close.iloc[-1] + atr.iloc[-1],
                    'take_profit': close.iloc[-1] - (2 * atr.iloc[-1]),
                    'risk_reward': 2.0
                }
        
        return {
            'components': {
                'ad_line': round(ad_line.iloc[-1], 2),
                'ad_sma': round(signals['ad_sma'], 2),
                'slope': 'up' if ad_line.iloc[-1] > ad_line.iloc[-2] else 'down'
            },
            'signals': signals['signals'],
            'divergence': signals['divergence'],
            'risk_management': risk_params
        }