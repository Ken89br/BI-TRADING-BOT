import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class StochasticStrategy:
    """
    Implementação profissional do Stochastic Oscillator com:
    - Cálculo do Stochastic Fast, Slow e Full
    - Identificação de divergências (regular e oculta)
    - Sinais de overbought/oversold dinâmicos
    - Confirmação com volume e tendência
    - Gerenciamento de risco automático
    """

    @staticmethod
    def calculate_stochastic(high: pd.Series,
                            low: pd.Series,
                            close: pd.Series,
                            k_period: int = 14,
                            d_period: int = 3,
                            smooth: int = 3) -> Dict[str, pd.Series]:
        """
        Calcula o Stochastic Oscillator (Fast, Slow e Full):
        - %K Fast = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
        - %D Fast = SMA(%K, d_period)
        - %K Slow = SMA(%K Fast, smooth)
        - %D Slow = SMA(%K Slow, d_period)
        """
        # Cálculo do %K Fast
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Cálculo do %D Fast
        d_fast = k_fast.rolling(window=d_period).mean()
        
        # Cálculo do %K Slow e %D Slow
        k_slow = k_fast.rolling(window=smooth).mean()
        d_slow = k_slow.rolling(window=d_period).mean()
        
        return {
            'k_fast': k_fast,
            'd_fast': d_fast,
            'k_slow': k_slow,
            'd_slow': d_slow
        }

    @staticmethod
    def identify_divergence(close: pd.Series,
                           k_line: pd.Series,
                           d_line: pd.Series,
                           lookback: int = 14) -> Dict[str, Union[bool, str]]:
        """
        Identifica divergências entre preço e Stochastic:
        - Regular (reversão): Preço faz novo máximo, mas Stochastic não.
        - Oculta (continuação): Preço faz correção, mas Stochastic mantém tendência.
        """
        # Encontra máximos/mínimos locais
        price_peaks = close.rolling(window=lookback, center=True).max() == close
        price_troughs = close.rolling(window=lookback, center=True).min() == close
        stochastic_peaks = k_line.rolling(window=lookback, center=True).max() == k_line
        stochastic_troughs = k_line.rolling(window=lookback, center=True).min() == k_line
        
        # Divergência regular (reversão)
        bearish_div = False
        bullish_div = False
        
        if any(price_peaks) and any(stochastic_peaks):
            if (close[price_peaks].iloc[-1] > close[price_peaks].iloc[-2] and
                k_line[stochastic_peaks].iloc[-1] < k_line[stochastic_peaks].iloc[-2]):
                bearish_div = True
        
        if any(price_troughs) and any(stochastic_troughs):
            if (close[price_troughs].iloc[-1] < close[price_troughs].iloc[-2] and
                k_line[stochastic_troughs].iloc[-1] > k_line[stochastic_troughs].iloc[-2]):
                bullish_div = True
        
        # Divergência oculta (continuação)
        hidden_bearish_div = False
        hidden_bullish_div = False
        
        if any(price_peaks) and any(stochastic_peaks):
            if (close[price_peaks].iloc[-1] < close[price_peaks].iloc[-2] and
                k_line[stochastic_peaks].iloc[-1] > k_line[stochastic_peaks].iloc[-2]):
                hidden_bearish_div = True
        
        if any(price_troughs) and any(stochastic_troughs):
            if (close[price_troughs].iloc[-1] > close[price_troughs].iloc[-2] and
                k_line[stochastic_troughs].iloc[-1] < k_line[stochastic_troughs].iloc[-2]):
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
                        k_line: pd.Series,
                        d_line: pd.Series,
                        volume: pd.Series = None,
                        trend_filter: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Cruzamento %K x %D
        - Overbought (>80) e Oversold (<20)
        - Divergências
        - Confirmação de volume e tendência
        """
        current_k = k_line.iloc[-1]
        current_d = d_line.iloc[-1]
        previous_k = k_line.iloc[-2]
        previous_d = d_line.iloc[-2]
        
        # Sinal de cruzamento
        signal = None
        if previous_k < previous_d and current_k > current_d:
            signal = 'buy' if current_k < 50 else None  # Filtro: só compra abaixo da média
        elif previous_k > previous_d and current_k < current_d:
            signal = 'sell' if current_k > 50 else None  # Filtro: só venda acima da média
        
        # Overbought/Oversold dinâmico (ajustado pela volatilidade)
        overbought = 80 + (5 if trend_filter is not None and trend_filter.iloc[-1] > 0 else 0)
        oversold = 20 - (5 if trend_filter is not None and trend_filter.iloc[-1] < 0 else 0)
        
        if current_k > overbought:
            signal = 'overbought'
        elif current_k < oversold:
            signal = 'oversold'
        
        # Divergências (prioritárias)
        divergence = StochasticStrategy.identify_divergence(close, k_line, d_line)
        if divergence['regular']['bearish']:
            signal = 'bearish_divergence'
        elif divergence['regular']['bullish']:
            signal = 'bullish_divergence'
        
        # Confirmação de volume e tendência
        confirmation = None
        if volume is not None and signal in ('buy', 'sell'):
            if volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                confirmation = 'volume_confirmed'
        
        if trend_filter is not None:
            if signal == 'buy' and trend_filter.iloc[-1] < 0:
                signal = None  # Ignora compras em tendência de baixa
            elif signal == 'sell' and trend_filter.iloc[-1] > 0:
                signal = None  # Ignora vendas em tendência de alta
        
        return {
            'signal': signal,
            'confirmation': confirmation,
            'current_k': current_k,
            'current_d': current_d,
            'overbought_level': overbought,
            'oversold_level': oversold,
            'divergence': divergence
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series = None,
                      k_period: int = 14,
                      d_period: int = 3,
                      smooth: int = 3) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do Stochastic:
        - Valores atuais de %K e %D
        - Sinais de trading
        - Divergências
        - Gerenciamento de risco
        """
        stochastic = StochasticStrategy.calculate_stochastic(
            high, low, close, k_period, d_period, smooth
        )
        
        # Filtro de tendência (opcional: usar ADX ou média móvel)
        trend_filter = close.rolling(window=50).mean()  # Exemplo simples
        
        signals = StochasticStrategy.generate_signals(
            close,
            stochastic['k_slow'],
            stochastic['d_slow'],
            volume,
            trend_filter
        )
        
        return {
            'stochastic': {
                'k_line': round(stochastic['k_slow'].iloc[-1], 2),
                'd_line': round(stochastic['d_slow'].iloc[-1], 2),
                'trend': 'up' if stochastic['k_slow'].iloc[-1] > stochastic['d_slow'].iloc[-1] else 'down'
            },
            'signals': signals,
            'risk_management': {
                'stop_loss': round(close.iloc[-1] * 0.98 if signals['signal'] == 'buy' else close.iloc[-1] * 1.02, 2),
                'take_profit': round(close.iloc[-1] * 1.04 if signals['signal'] == 'buy' else close.iloc[-1] * 0.96, 2)
            }
        }