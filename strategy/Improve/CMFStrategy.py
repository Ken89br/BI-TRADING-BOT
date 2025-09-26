import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class CMFStrategy:
    """
    Implementação profissional do Chaikin Money Flow com:
    - Cálculo preciso do CMF com periodos ajustáveis
    - Identificação de divergências preço/CMF
    - Análise de acumulação/distribuição
    - Confirmação com volume e ADL (Accumulation/Distribution Line)
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_cmf(high: pd.Series,
                    low: pd.Series,
                    close: pd.Series,
                    volume: pd.Series,
                    period: int = 20) -> Dict[str, pd.Series]:
        """
        Calcula o Chaikin Money Flow:
        CMF = Soma(MFV, n períodos) / Soma(Volume, n períodos)
        Onde MFV = [(Fechamento - Mínimo) - (Máximo - Fechamento)] / (Máximo - Mínimo) * Volume
        """
        # Evita divisão por zero
        price_range = high - low
        price_range[price_range == 0] = 0.0001  # Substitui zeros por valor mínimo
        
        # Money Flow Volume
        mfv = ((close - low) - (high - close)) / price_range * volume
        
        # Chaikin Money Flow
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        # Accumulation/Distribution Line
        adl = ((close - low) - (high - close)) / price_range * volume
        adl = adl.cumsum()
        
        return {
            'cmf': cmf,
            'mfv': mfv,
            'adl': adl,
            'price_range': price_range
        }

    @staticmethod
    def identify_divergence(close: pd.Series,
                          cmf: pd.Series,
                          lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e CMF:
        - Bullish: Preço faz fundo mais baixo, CMF faz fundo mais alto
        - Bearish: Preço faz topo mais alto, CMF faz topo mais baixo
        """
        # Encontra máximos/mínimos locais
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        cmf_peaks = cmf.rolling(window=lookback, center=True).max() == cmf
        cmf_troughs = cmf.rolling(window=lookback, center=True).min() == cmf
        
        # Verifica divergências
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(cmf_peaks):
            last_price_peak = close[price_peaks].index[-1]
            last_cmf_peak = cmf[cmf_peaks].index[-1]
            if (close[last_price_peak] > close[price_peaks].iloc[-2] and
                cmf[last_cmf_peak] < cmf[cmf_peaks].iloc[-2]):
                bearish_div = True
        
        if any(price_troughs) and any(cmf_troughs):
            last_price_trough = close[price_troughs].index[-1]
            last_cmf_trough = cmf[cmf_troughs].index[-1]
            if (close[last_price_trough] < close[price_troughs].iloc[-2] and
                cmf[last_cmf_trough] > cmf[cmf_troughs].iloc[-2]):
                bullish_div = True
        
        return {
            'bearish': bearish_div,
            'bullish': bullish_div
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                       cmf: pd.Series,
                       adl: pd.Series,
                       volume: pd.Series,
                       threshold: float = 0.2) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento do zero line
        - Divergências
        - Níveis extremos (+0.3/-0.3)
        - Confirmação com ADL e volume
        """
        current_cmf = cmf.iloc[-1]
        previous_cmf = cmf.iloc[-2]
        current_adl = adl.iloc[-1]
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # Sinal básico
        signal = None
        if current_cmf > threshold and previous_cmf <= threshold:
            signal = 'bullish_breakout'
        elif current_cmf < -threshold and previous_cmf >= -threshold:
            signal = 'bearish_breakout'
        elif current_cmf > 0 and previous_cmf <= 0:
            signal = 'bullish_crossover'
        elif current_cmf < 0 and previous_cmf >= 0:
            signal = 'bearish_crossover'
        
        # Divergências (sobrescrevem outros sinais)
        divergence = CMFStrategy.identify_divergence(close, cmf)
        if divergence['bullish']:
            signal = 'bullish_divergence'
        elif divergence['bearish']:
            signal = 'bearish_divergence'
        
        # Confirmação de força
        strength = 'weak'
        if signal:
            if current_volume > avg_volume * 1.5:
                strength = 'strong_volume'
            elif abs(current_adl) > abs(adl.rolling(20).mean().iloc[-1]) * 1.5:
                strength = 'strong_adl'
            elif abs(current_cmf) > 0.4:
                strength = 'strong_cmf'
        
        return {
            'signal': signal,
            'strength': strength,
            'current_cmf': current_cmf,
            'current_adl': current_adl,
            'volume_ratio': current_volume / avg_volume
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     volume: pd.Series,
                     period: int = 20,
                     threshold: float = 0.2) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do CMF:
        - Valores do CMF e ADL
        - Sinais de trading
        - Divergências
        - Gerenciamento de risco
        """
        cmf_data = CMFStrategy.calculate_cmf(high, low, close, volume, period)
        signals = CMFStrategy.generate_signals(
            close, cmf_data['cmf'], cmf_data['adl'], volume, threshold
        )
        
        return {
            'cmf': {
                'current': round(cmf_data['cmf'].iloc[-1], 3),
                'mean': round(cmf_data['cmf'].mean(), 3),
                'std_dev': round(cmf_data['cmf'].std(), 3)
            },
            'adl': round(cmf_data['adl'].iloc[-1], 2),
            'signals': signals,
            'risk_management': {
                'stop_loss': round(
                    close.iloc[-1] * 0.98 if signals['signal'] in ['bullish_breakout', 'bullish_crossover', 'bullish_divergence']
                    else close.iloc[-1] * 1.02, 2
                ),
                'take_profit': round(
                    close.iloc[-1] * 1.03 if signals['signal'] in ['bullish_breakout', 'bullish_crossover', 'bullish_divergence']
                    else close.iloc[-1] * 0.97, 2
                ),
                'risk_reward_ratio': '1:2'  # Padrão institucional
            }
        }