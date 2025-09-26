import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class EnvelopesStrategy:
    """
    Implementação profissional da estratégia Envelopes com:
    - Cálculo de bandas dinâmicas baseadas em média móvel
    - Ajuste automático de largura baseado na volatilidade (ATR)
    - Identificação de condições de sobrecompra/sobrevenda
    - Confirmação de volume e momentum
    - Gerenciamento de risco integrado
    """

    @staticmethod
    def calculate_envelopes(close: pd.Series,
                          ma_type: str = 'sma',
                          ma_period: int = 20,
                          atr_period: int = 14,
                          width_multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calcula as bandas de Envelope:
        - Banda Superior: Média Móvel + (ATR * Multiplicador)
        - Banda Inferior: Média Móvel - (ATR * Multiplicador)
        
        Tipos de média móvel suportados: 'sma', 'ema', 'wma'
        """
        # Calcula a média móvel
        if ma_type == 'sma':
            ma = close.rolling(window=ma_period).mean()
        elif ma_type == 'ema':
            ma = close.ewm(span=ma_period, adjust=False).mean()
        elif ma_type == 'wma':
            weights = np.arange(1, ma_period + 1)
            wma = close.rolling(window=ma_period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True)
            ma = wma
        
        # Calcula o ATR para ajuste dinâmico da largura
        high = close  # Para simplificar, usamos close como high/low
        low = close
        tr = pd.DataFrame({
            'high_low': high - low,
            'high_close': abs(high - close.shift(1)),
            'low_close': abs(low - close.shift(1))
        }).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        # Calcula as bandas
        upper_band = ma + (atr * width_multiplier)
        lower_band = ma - (atr * width_multiplier)
        
        return {
            'ma': ma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr,
            'width': upper_band - lower_band
        }

    @staticmethod
    def identify_conditions(close: pd.Series,
                          upper_band: pd.Series,
                          lower_band: pd.Series,
                          rsi: pd.Series = None) -> Dict[str, Union[bool, float]]:
        """
        Identifica condições de mercado:
        - Sobrecompra: Preço toca banda superior + confirmação RSI
        - Sobrevenda: Preço toca banda inferior + confirmação RSI
        - Volatilidade anormal: Alargamento repentino das bandas
        """
        current_price = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        overbought = current_price >= current_upper
        oversold = current_price <= current_lower
        
        # Confirmação com RSI se fornecido
        rsi_confirmed = None
        if rsi is not None:
            rsi_value = rsi.iloc[-1]
            rsi_overbought = rsi_value > 70 if overbought else None
            rsi_oversold = rsi_value < 30 if oversold else None
            rsi_confirmed = {
                'overbought': rsi_overbought,
                'oversold': rsi_oversold
            }
        
        # Detecção de volatilidade anormal
        width = upper_band - lower_band
        width_zscore = (width.iloc[-1] - width.mean()) / width.std()
        high_volatility = abs(width_zscore) > 2
        
        return {
            'overbought': overbought,
            'oversold': oversold,
            'rsi_confirmation': rsi_confirmed,
            'high_volatility': high_volatility,
            'distance_to_upper_pct': ((current_upper - current_price) / current_price) * 100,
            'distance_to_lower_pct': ((current_price - current_lower) / current_price) * 100
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                       ma: pd.Series,
                       upper_band: pd.Series,
                       lower_band: pd.Series,
                       volume: pd.Series = None,
                       conditions: Dict = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais de trading baseados em:
        - Reversão nas bandas extremas
        - Cruzamento com a média central
        - Confirmação de volume
        """
        current_price = close.iloc[-1]
        current_ma = ma.iloc[-1]
        previous_price = close.iloc[-2]
        previous_ma = ma.iloc[-2]
        
        # Sinal de reversão
        signal = None
        if conditions['overbought']:
            if current_price < previous_price and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                signal = 'bearish_reversal'
        elif conditions['oversold']:
            if current_price > previous_price and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                signal = 'bullish_reversal'
        
        # Sinal de cruzamento com a média
        if signal is None:
            if current_price > current_ma and previous_price <= previous_ma:
                signal = 'bullish_crossover'
            elif current_price < current_ma and previous_price >= previous_ma:
                signal = 'bearish_crossover'
        
        # Força do sinal
        strength = 'weak'
        if signal:
            if conditions['high_volatility']:
                strength = 'strong_volatility'
            elif volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                strength = 'strong_volume'
            elif conditions['rsi_confirmation'] is not None:
                if (signal.startswith('bullish') and conditions['rsi_confirmation']['oversold']) or \
                   (signal.startswith('bearish') and conditions['rsi_confirmation']['overbought']):
                    strength = 'strong_rsi'
        
        return {
            'signal': signal,
            'strength': strength,
            'current_price': current_price,
            'ma_value': current_ma,
            'upper_band_value': upper_band.iloc[-1],
            'lower_band_value': lower_band.iloc[-1]
        }

    @staticmethod
    def full_analysis(close: pd.Series,
                     high: pd.Series = None,
                     low: pd.Series = None,
                     volume: pd.Series = None,
                     rsi: pd.Series = None,
                     ma_type: str = 'sma',
                     ma_period: int = 20,
                     atr_period: int = 14,
                     width_multiplier: float = 2.0) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa da estratégia Envelopes:
        - Cálculo das bandas
        - Identificação de condições
        - Geração de sinais
        - Gerenciamento de risco
        """
        envelopes = EnvelopesStrategy.calculate_envelopes(
            close, ma_type, ma_period, atr_period, width_multiplier
        )
        conditions = EnvelopesStrategy.identify_conditions(
            close, envelopes['upper_band'], envelopes['lower_band'], rsi
        )
        signals = EnvelopesStrategy.generate_signals(
            close, envelopes['ma'], envelopes['upper_band'], 
            envelopes['lower_band'], volume, conditions
        )
        
        return {
            'envelopes': {
                'ma': round(envelopes['ma'].iloc[-1], 5),
                'upper_band': round(envelopes['upper_band'].iloc[-1], 5),
                'lower_band': round(envelopes['lower_band'].iloc[-1], 5),
                'width': round(envelopes['width'].iloc[-1], 5),
                'atr': round(envelopes['atr'].iloc[-1], 5)
            },
            'conditions': conditions,
            'signals': signals,
            'risk_management': {
                'stop_loss': round(
                    envelopes['lower_band'].iloc[-1] * 0.99 if signals['signal'] == 'bullish_reversal' 
                    else envelopes['upper_band'].iloc[-1] * 1.01, 2
                ),
                'take_profit': round(
                    envelopes['ma'].iloc[-1] if signals['signal'] == 'bullish_reversal'
                    else envelopes['ma'].iloc[-1], 2
                ),
                'risk_reward_ratio': round(
                    (envelopes['ma'].iloc[-1] - envelopes['lower_band'].iloc[-1]) / 
                    (envelopes['lower_band'].iloc[-1] * 0.01), 2
                ) if signals['signal'] == 'bullish_reversal' else round(
                    (envelopes['upper_band'].iloc[-1] - envelopes['ma'].iloc[-1]) / 
                    (envelopes['upper_band'].iloc[-1] * 0.01), 2
                )
            }
        }