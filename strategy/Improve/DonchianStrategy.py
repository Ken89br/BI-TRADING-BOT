import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class DonchianStrategy:
    """
    Implementação profissional dos Donchian Channels com:
    - Cálculo dos canais baseado em máximos/mínimos móveis
    - Breakouts com confirmação de volume e fechamento
    - Identificação de pullbacks dentro da tendência
    - Gerenciamento de risco baseado na largura do canal
    - Filtro de tendência ADX opcional
    """

    @staticmethod
    def calculate_channels(high: pd.Series,
                          low: pd.Series,
                          period: int = 20) -> Dict[str, pd.Series]:
        """
        Calcula os Donchian Channels:
        - Upper Band: Máximo dos últimos 'period' períodos
        - Lower Band: Mínimo dos últimos 'period' períodos
        - Middle Band: Ponto médio entre as bandas
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return {
            'upper': upper,
            'lower': lower,
            'middle': middle,
            'width': upper - lower  # Largura do canal (volatilidade)
        }

    @staticmethod
    def identify_breakout(close: pd.Series,
                         upper: pd.Series,
                         lower: pd.Series,
                         volume: pd.Series = None,
                         min_volume_multiplier: float = 1.5) -> Dict[str, Union[bool, str]]:
        """
        Identifica breakouts válidos:
        - Fechamento acima/abaixo do canal
        - Confirmação de volume (opcional)
        - Filtro de fechamento (evita falsos breakouts intradiários)
        """
        current_close = close.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        breakout = None
        confirmed = False
        
        # Breakout de alta
        if current_close > current_upper:
            breakout = 'bullish'
            # Confirmação: volume acima da média e fechamento no top 25% do candle
            if volume is not None:
                volume_ok = volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * min_volume_multiplier
                close_position_ok = (current_close - close.iloc[-2]) / (high.iloc[-1] - low.iloc[-1]) > 0.5
                confirmed = volume_ok and close_position_ok
            else:
                confirmed = True
        
        # Breakout de baixa
        elif current_close < current_lower:
            breakout = 'bearish'
            if volume is not None:
                volume_ok = volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * min_volume_multiplier
                close_position_ok = (close.iloc[-2] - current_close) / (high.iloc[-1] - low.iloc[-1]) > 0.5
                confirmed = volume_ok and close_position_ok
            else:
                confirmed = True
        
        return {
            'breakout': breakout,
            'confirmed': confirmed,
            'price': current_close,
            'upper_band': current_upper,
            'lower_band': current_lower
        }

    @staticmethod
    def generate_signals(high: pd.Series,
                        low: pd.Series,
                        close: pd.Series,
                        upper: pd.Series,
                        lower: pd.Series,
                        middle: pd.Series,
                        volume: pd.Series = None,
                        adx: pd.Series = None,
                        min_adx: float = 25) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Breakouts confirmados
        - Pullbacks para a banda média
        - Reversões nos extremos do canal
        """
        signals = []
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        
        # 1. Breakout Analysis
        breakout = DonchianStrategy.identify_breakout(close, upper, lower, volume)
        if breakout['breakout'] and breakout['confirmed']:
            if adx is not None and adx.iloc[-1] >= min_adx:
                signals.append({
                    'type': f"breakout_{breakout['breakout']}",
                    'price': breakout['price'],
                    'confidence': 'high' if volume is not None else 'medium'
                })
        
        # 2. Pullback to Middle Band
        if (middle.iloc[-1] < close.iloc[-1] < upper.iloc[-1] and 
            close.iloc[-2] <= middle.iloc[-2] and
            (adx is None or adx.iloc[-1] >= min_adx)):
            signals.append({
                'type': 'pullback_bullish',
                'price': current_close,
                'confidence': 'medium'
            })
        elif (lower.iloc[-1] < close.iloc[-1] < middle.iloc[-1] and 
              close.iloc[-2] >= middle.iloc[-2] and
              (adx is None or adx.iloc[-1] >= min_adx)):
            signals.append({
                'type': 'pullback_bearish',
                'price': current_close,
                'confidence': 'medium'
            })
        
        # 3. Extreme Reversals (Pin Bars at Bands)
        upper_rejection = (current_high >= upper.iloc[-1] and 
                          (current_high - current_close) > 0.6 * (current_high - current_low))
        lower_rejection = (current_low <= lower.iloc[-1] and 
                          (current_close - current_low) > 0.6 * (current_high - current_low))
        
        if upper_rejection:
            signals.append({
                'type': 'reversal_bearish',
                'price': current_close,
                'confidence': 'high' if volume.iloc[-1] > volume.rolling(5).mean().iloc[-1] else 'medium'
            })
        elif lower_rejection:
            signals.append({
                'type': 'reversal_bullish',
                'price': current_close,
                'confidence': 'high' if volume.iloc[-1] > volume.rolling(5).mean().iloc[-1] else 'medium'
            })
        
        return {
            'signals': signals,
            'current_upper': upper.iloc[-1],
            'current_lower': lower.iloc[-1],
            'current_middle': middle.iloc[-1],
            'channel_width': upper.iloc[-1] - lower.iloc[-1]
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series = None,
                      period: int = 20,
                      adx_period: int = 14,
                      min_adx: float = 25) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa dos Donchian Channels:
        - Componentes dos canais
        - Sinais de trading
        - Gerenciamento de risco
        - Filtro de tendência (ADX opcional)
        """
        channels = DonchianStrategy.calculate_channels(high, low, period)
        
        # Calcula ADX se necessário
        adx = None
        if adx_period:
            from ta.trend import ADXIndicator
            adx = ADXIndicator(high, low, close, window=adx_period).adx()
        
        signals = DonchianStrategy.generate_signals(
            high, low, close,
            channels['upper'],
            channels['lower'],
            channels['middle'],
            volume,
            adx,
            min_adx
        )
        
        # Gerenciamento de Risco
        risk_params = {}
        if signals['signals']:
            last_signal = signals['signals'][-1]
            channel_width = signals['channel_width']
            
            if 'breakout_bullish' in last_signal['type']:
                risk_params = {
                    'stop_loss': channels['lower'].iloc[-1],
                    'take_profit': close.iloc[-1] + channel_width,
                    'risk_reward': round((close.iloc[-1] + channel_width - close.iloc[-1]) / 
                                    (close.iloc[-1] - channels['lower'].iloc[-1]), 2)
                }
            elif 'breakout_bearish' in last_signal['type']:
                risk_params = {
                    'stop_loss': channels['upper'].iloc[-1],
                    'take_profit': close.iloc[-1] - channel_width,
                    'risk_reward': round((channels['upper'].iloc[-1] - close.iloc[-1]) / 
                                    (close.iloc[-1] - (close.iloc[-1] - channel_width)), 2)
                }
        
        return {
            'components': {
                'upper_band': round(channels['upper'].iloc[-1], 5),
                'lower_band': round(channels['lower'].iloc[-1], 5),
                'middle_band': round(channels['middle'].iloc[-1], 5),
                'width': round(signals['channel_width'], 5),
                'adx': round(adx.iloc[-1], 2) if adx is not None else None
            },
            'signals': signals['signals'],
            'risk_management': risk_params
        }