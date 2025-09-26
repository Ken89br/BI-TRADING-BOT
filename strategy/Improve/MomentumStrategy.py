import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class MomentumStrategy:
    """
    Implementação profissional da estratégia Momentum com:
    - Cálculo do Momentum clássico e normalizado
    - Bandas dinâmicas de overbought/oversold
    - Identificação de divergências
    - Confirmação com volume e volatilidade
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_momentum(close: pd.Series,
                         period: int = 10,
                         normalize: bool = True) -> Dict[str, pd.Series]:
        """
        Calcula o Momentum:
        - Clássico: M = Preço(t) / Preço(t-n) * 100
        - Normalizado: (M - Média(M)) / Desvio Padrão(M)
        """
        # Momentum bruto
        raw_momentum = (close / close.shift(period)) * 100
        
        if normalize:
            # Momentum normalizado (Z-Score)
            momentum_mean = raw_momentum.rolling(window=period).mean()
            momentum_std = raw_momentum.rolling(window=period).std()
            momentum = (raw_momentum - momentum_mean) / momentum_std
        else:
            momentum = raw_momentum
        
        # Linha de sinal (média móvel do momentum)
        signal_line = momentum.rolling(window=3).mean()
        
        return {
            'momentum': momentum,
            'signal_line': signal_line,
            'raw_momentum': raw_momentum
        }

    @staticmethod
    def identify_divergence(close: pd.Series,
                          momentum: pd.Series,
                          lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e momentum:
        - Regular (reversão)
        - Oculta (continuação)
        """
        # Encontra máximos/mínimos locais
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        mom_peaks = momentum.rolling(window=lookback, center=True).max() == momentum
        mom_troughs = momentum.rolling(window=lookback, center=True).min() == momentum
        
        # Divergência regular
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(mom_peaks):
            last_price_peak = close[price_peaks].index[-1]
            last_mom_peak = momentum[mom_peaks].index[-1]
            if (close[last_price_peak] > close[price_peaks].iloc[-2] and
                momentum[last_mom_peak] < momentum[mom_peaks].iloc[-2]):
                bearish_div = True
                
        if any(price_troughs) and any(mom_troughs):
            last_price_trough = close[price_troughs].index[-1]
            last_mom_trough = momentum[mom_troughs].index[-1]
            if (close[last_price_trough] < close[price_troughs].iloc[-2] and
                momentum[last_mom_trough] > momentum[mom_troughs].iloc[-2]):
                bullish_div = True
        
        # Divergência oculta
        hidden_bearish = False
        hidden_bullish = False
        
        if any(price_peaks) and any(mom_peaks):
            if (close[price_peaks].iloc[-1] < close[price_peaks].iloc[-2] and
                momentum[mom_peaks].iloc[-1] > momentum[mom_peaks].iloc[-2]):
                hidden_bearish = True
                
        if any(price_troughs) and any(mom_troughs):
            if (close[price_troughs].iloc[-1] > close[price_troughs].iloc[-2] and
                momentum[mom_troughs].iloc[-1] < momentum[mom_troughs].iloc[-2]):
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
                       momentum: pd.Series,
                       signal_line: pd.Series,
                       volume: pd.Series,
                       volatility: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento do momentum com a linha de sinal
        - Divergências
        - Níveis extremos dinâmicos
        - Confirmação de volume
        """
        current_mom = momentum.iloc[-1]
        current_signal = signal_line.iloc[-1]
        previous_mom = momentum.iloc[-2]
        previous_signal = signal_line.iloc[-2]
        
        # Níveis dinâmicos de overbought/oversold
        if volatility is not None:
            overbought = 2.0 + (volatility.iloc[-1] * 0.5)
            oversold = -2.0 - (volatility.iloc[-1] * 0.5)
        else:
            overbought = 2.0
            oversold = -2.0
        
        # Sinal básico
        signal = None
        if current_mom > current_signal and previous_mom <= previous_signal:
            signal = 'bullish_crossover'
        elif current_mom < current_signal and previous_mom >= previous_signal:
            signal = 'bearish_crossover'
        
        # Níveis extremos
        if current_mom > overbought:
            signal = 'overbought'
        elif current_mom < oversold:
            signal = 'oversold'
        
        # Divergências (sobrescrevem outros sinais)
        divergence = MomentumStrategy.identify_divergence(close, momentum)
        if divergence['regular']['bearish']:
            signal = 'bearish_divergence'
        elif divergence['regular']['bullish']:
            signal = 'bullish_divergence'
        
        # Força do sinal
        strength = 'weak'
        if signal:
            if volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                strength = 'strong_volume'
            elif abs(current_mom) > 3.0:
                strength = 'strong_momentum'
            elif divergence['regular']['bearish'] or divergence['regular']['bullish']:
                strength = 'strong_divergence'
        
        return {
            'signal': signal,
            'strength': strength,
            'current_momentum': current_mom,
            'overbought_level': overbought,
            'oversold_level': oversold,
            'divergence': divergence
        }

    @staticmethod
    def full_analysis(close: pd.Series,
                     high: pd.Series = None,
                     low: pd.Series = None,
                     volume: pd.Series = None,
                     momentum_period: int = 10,
                     normalize: bool = True,
                     atr_period: int = 14) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do Momentum:
        - Valores do momentum e linha de sinal
        - Sinais de trading
        - Divergências
        - Gerenciamento de risco
        """
        # Calcula volatilidade (ATR) se high/low forem fornecidos
        volatility = None
        if high is not None and low is not None:
            tr = pd.DataFrame({
                'high_low': high - low,
                'high_close': abs(high - close.shift(1)),
                'low_close': abs(low - close.shift(1))
            }).max(axis=1)
            volatility = tr.rolling(window=atr_period).mean()
        
        # Calcula momentum
        momentum_data = MomentumStrategy.calculate_momentum(
            close, momentum_period, normalize
        )
        
        # Gera sinais
        signals = MomentumStrategy.generate_signals(
            close,
            momentum_data['momentum'],
            momentum_data['signal_line'],
            volume,
            volatility
        )
        
        return {
            'momentum': {
                'current': round(momentum_data['momentum'].iloc[-1], 3),
                'signal_line': round(momentum_data['signal_line'].iloc[-1], 3),
                'normalized': normalize
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(
                    close.iloc[-1] * 0.98 if signals['signal'] in ['bullish_crossover', 'bullish_divergence']
                    else close.iloc[-1] * 1.02, 2
                ),
                'take_profit': round(
                    close.iloc[-1] * 1.03 if signals['signal'] in ['bullish_crossover', 'bullish_divergence']
                    else close.iloc[-1] * 0.97, 2
                ),
                'risk_reward_ratio': '1:2'  # Padrão institucional
            }
        }
