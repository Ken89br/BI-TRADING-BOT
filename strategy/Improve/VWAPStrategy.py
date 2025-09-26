import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class VWAPStrategy:
    """
    Implementação profissional do VWAP (Volume-Weighted Average Price) com:
    - Cálculo intradiário do VWAP clássico
    - Bandas de desvio padrão (1x, 2x, 3x)
    - Identificação de zonas de valor extremas
    - Confirmação com perfis de volume
    - Sinais de trading institucionais
    """

    @staticmethod
    def calculate_vwap(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series,
                      lookback: int = 20) -> Dict[str, Union[pd.Series, Dict]]:
        """
        Calcula o VWAP e bandas de desvio padrão.
        Fórmula: VWAP = ∑(Preço típico * Volume) / ∑Volume
        Onde Preço típico = (High + Low + Close)/3
        """
        typical_price = (high + low + close) / 3
        cumulative_vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Calcula desvios padrão móveis
        std_dev = typical_price.rolling(window=lookback).std()
        
        # Bandas de desvio
        upper_band_1 = cumulative_vwap + std_dev
        upper_band_2 = cumulative_vwap + 2 * std_dev
        upper_band_3 = cumulative_vwap + 3 * std_dev
        lower_band_1 = cumulative_vwap - std_dev
        lower_band_2 = cumulative_vwap - 2 * std_dev
        lower_band_3 = cumulative_vwap - 3 * std_dev
        
        return {
            'vwap': cumulative_vwap,
            'upper_bands': {
                '1st': upper_band_1,
                '2nd': upper_band_2,
                '3rd': upper_band_3
            },
            'lower_bands': {
                '1st': lower_band_1,
                '2nd': lower_band_2,
                '3rd': lower_band_3
            },
            'typical_price': typical_price
        }

    @staticmethod
    def identify_extremes(close: pd.Series,
                         vwap: pd.Series,
                         upper_bands: Dict[str, pd.Series],
                         lower_bands: Dict[str, pd.Series]) -> Dict[str, Union[bool, float]]:
        """
        Identifica zonas extremas baseadas no VWAP:
        - Acima da 3ª banda superior = supercomprado extremo
        - Abaixo da 3ª banda inferior = supervendido extremo
        """
        current_price = close.iloc[-1]
        upper_3rd = upper_bands['3rd'].iloc[-1]
        lower_3rd = lower_bands['3rd'].iloc[-1]
        
        is_extreme_overbought = current_price >= upper_3rd
        is_extreme_oversold = current_price <= lower_3rd
        
        return {
            'extreme_overbought': is_extreme_overbought,
            'extreme_oversold': is_extreme_oversold,
            'distance_to_vwap_pct': ((current_price - vwap.iloc[-1]) / vwap.iloc[-1]) * 100
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                        vwap: pd.Series,
                        upper_bands: Dict[str, pd.Series],
                        lower_bands: Dict[str, pd.Series],
                        volume: pd.Series) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Reversão de zonas extremas
        - Crossovers do preço com o VWAP
        - Confirmação de volume
        """
        current_price = close.iloc[-1]
        previous_price = close.iloc[-2]
        current_vwap = vwap.iloc[-1]
        previous_vwap = vwap.iloc[-2]
        
        # Extremos
        extremes = VWAPStrategy.identify_extremes(close, vwap, upper_bands, lower_bands)
        
        # Sinal de crossover
        signal = None
        if current_price > current_vwap and previous_price <= previous_vwap:
            signal = 'bullish_crossover'
        elif current_price < current_vwap and previous_price >= previous_vwap:
            signal = 'bearish_crossover'
        
        # Sinal de reversão de zona extrema
        if extremes['extreme_overbought'] and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
            signal = 'extreme_reversal_short'
        elif extremes['extreme_oversold'] and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
            signal = 'extreme_reversal_long'
        
        return {
            'signal': signal,
            'current_vwap': current_vwap,
            'distance_to_vwap_pct': extremes['distance_to_vwap_pct'],
            'extremes': extremes
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     volume: pd.Series,
                     lookback: int = 20) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do VWAP:
        - Valor do VWAP e bandas
        - Sinais de trading
        - Zonas extremas
        - Gerenciamento de risco
        """
        vwap_data = VWAPStrategy.calculate_vwap(high, low, close, volume, lookback)
        signals = VWAPStrategy.generate_signals(
            close,
            vwap_data['vwap'],
            vwap_data['upper_bands'],
            vwap_data['lower_bands'],
            volume
        )
        
        return {
            'vwap': {
                'current': round(vwap_data['vwap'].iloc[-1], 5),
                'upper_bands': {
                    '1st': round(vwap_data['upper_bands']['1st'].iloc[-1], 5),
                    '2nd': round(vwap_data['upper_bands']['2nd'].iloc[-1], 5),
                    '3rd': round(vwap_data['upper_bands']['3rd'].iloc[-1], 5)
                },
                'lower_bands': {
                    '1st': round(vwap_data['lower_bands']['1st'].iloc[-1], 5),
                    '2nd': round(vwap_data['lower_bands']['2nd'].iloc[-1], 5),
                    '3rd': round(vwap_data['lower_bands']['3rd'].iloc[-1], 5)
                }
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(close.iloc[-1] * 0.99 if signals['signal'] == 'extreme_reversal_long' else close.iloc[-1] * 1.01, 2),
                'take_profit': round(close.iloc[-1] * 1.03 if signals['signal'] == 'extreme_reversal_long' else close.iloc[-1] * 0.97, 2)
            }
        }