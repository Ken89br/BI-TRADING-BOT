import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class BollingerBandsStrategy:
    """
    Implementação profissional das Bollinger Bands com:
    - Cálculo das bandas com desvio padrão dinâmico
    - Detecção de squeeze e condições extremas
    - Confirmação de volume e momentum
    - Gerenciamento de risco integrado
    - Ajuste automático de parâmetros
    """

    @staticmethod
    def calculate_bollinger_bands(close: pd.Series,
                                period: int = 20,
                                std_dev: float = 2.0,
                                ma_type: str = 'sma') -> Dict[str, pd.Series]:
        """
        Calcula as Bollinger Bands:
        - Banda Média: Média Móvel (SMA/EMA)
        - Banda Superior: Média + (Desvio Padrão * Multiplicador)
        - Banda Inferior: Média - (Desvio Padrão * Multiplicador)
        """
        if ma_type == 'sma':
            middle_band = close.rolling(window=period).mean()
        elif ma_type == 'ema':
            middle_band = close.ewm(span=period, adjust=False).mean()
        
        std = close.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        bandwidth = (upper_band - lower_band) / middle_band * 100
        
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'bandwidth': bandwidth,
            'std_dev': std
        }

    @staticmethod
    def identify_conditions(close: pd.Series,
                          upper_band: pd.Series,
                          middle_band: pd.Series,
                          lower_band: pd.Series,
                          bandwidth: pd.Series) -> Dict[str, Union[bool, float]]:
        """
        Identifica condições de mercado:
        - Preço acima/abaixo das bandas
        - Squeeze (bandwidth estreito)
        - Afastamento da banda média
        """
        current_price = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = middle_band.iloc[-1]
        
        # Condições básicas
        above_upper = current_price >= current_upper
        below_lower = current_price <= current_lower
        near_middle = abs(current_price - current_middle) < (current_upper - current_middle) * 0.3
        
        # Squeeze (percentil 20 da largura histórica)
        bandwidth_pct = bandwidth.rank(pct=True).iloc[-1] * 100
        squeeze = bandwidth_pct < 20
        
        return {
            'above_upper': above_upper,
            'below_lower': below_lower,
            'near_middle': near_middle,
            'squeeze': squeeze,
            'bandwidth_pct': bandwidth_pct,
            'distance_to_middle': ((current_price - current_middle) / current_middle) * 100
        }

    @staticmethod
    def generate_signals(close: pd.Series,
                       upper_band: pd.Series,
                       middle_band: pd.Series,
                       lower_band: pd.Series,
                       conditions: Dict,
                       volume: pd.Series = None,
                       rsi: pd.Series = None) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Reversão nas bandas extremas
        - Breakouts após squeeze
        - Confirmação com RSI e volume
        """
        current_price = close.iloc[-1]
        previous_price = close.iloc[-2]
        current_middle = middle_band.iloc[-1]
        
        # Sinal básico
        signal = None
        if conditions['above_upper']:
            if rsi is not None and rsi.iloc[-1] > 70:
                signal = 'bearish_reversal'
        elif conditions['below_lower']:
            if rsi is not None and rsi.iloc[-1] < 30:
                signal = 'bullish_reversal'
        
        # Breakout após squeeze
        if conditions['squeeze'] and not signal:
            if current_price > previous_price and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.2:
                signal = 'bullish_breakout'
            elif current_price < previous_price and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.2:
                signal = 'bearish_breakout'
        
        # Força do sinal
        strength = 'weak'
        if signal:
            if volume is not None and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5:
                strength = 'strong_volume'
            elif conditions['squeeze'] and signal.endswith('breakout'):
                strength = 'strong_squeeze'
            elif (signal == 'bullish_reversal' and rsi.iloc[-1] < 25) or (signal == 'bearish_reversal' and rsi.iloc[-1] > 75):
                strength = 'strong_rsi'
        
        return {
            'signal': signal,
            'strength': strength,
            'current_price': current_price,
            'middle_band': current_middle,
            'bandwidth_pct': conditions['bandwidth_pct']
        }

    @staticmethod
    def risk_management(close: pd.Series,
                      middle_band: pd.Series,
                      signal: str) -> Dict[str, float]:
        """
        Calcula níveis de stop loss e take profit:
        - Stop na banda oposta ou média móvel
        - Take profit com relação risco/recompensa
        """
        current_price = close.iloc[-1]
        current_middle = middle_band.iloc[-1]
        
        if signal == 'bullish_reversal' or signal == 'bullish_breakout':
            stop_loss = min(current_middle, close.iloc[-3:].min())
            take_profit = current_price + (current_price - stop_loss) * 1.5
        elif signal == 'bearish_reversal' or signal == 'bearish_breakout':
            stop_loss = max(current_middle, close.iloc[-3:].max())
            take_profit = current_price - (stop_loss - current_price) * 1.5
        else:
            stop_loss = take_profit = None
        
        return {
            'stop_loss': round(stop_loss, 2) if stop_loss else None,
            'take_profit': round(take_profit, 2) if take_profit else None,
            'risk_reward_ratio': 1.5
        }

    @staticmethod
    def full_analysis(close: pd.Series,
                    high: pd.Series = None,
                    low: pd.Series = None,
                    volume: pd.Series = None,
                    rsi: pd.Series = None,
                    period: int = 20,
                    std_dev: float = 2.0,
                    ma_type: str = 'sma') -> Dict[str, Union[Dict, str]]:
        """
        Análise completa das Bollinger Bands:
        - Cálculo das bandas
        - Identificação de condições
        - Geração de sinais
        - Gestão de risco
        """
        bands = BollingerBandsStrategy.calculate_bollinger_bands(close, period, std_dev, ma_type)
        conditions = BollingerBandsStrategy.identify_conditions(close, bands['upper_band'], bands['middle_band'], bands['lower_band'], bands['bandwidth'])
        signals = BollingerBandsStrategy.generate_signals(close, bands['upper_band'], bands['middle_band'], bands['lower_band'], conditions, volume, rsi)
        risk_mgmt = BollingerBandsStrategy.risk_management(close, bands['middle_band'], signals['signal'])
        
        return {
            'bands': {
                'upper': round(bands['upper_band'].iloc[-1], 5),
                'middle': round(bands['middle_band'].iloc[-1], 5),
                'lower': round(bands['lower_band'].iloc[-1], 5),
                'bandwidth': round(bands['bandwidth'].iloc[-1], 2),
                'std_dev': round(bands['std_dev'].iloc[-1], 5)
            },
            'conditions': conditions,
            'signals': signals,
            'risk_management': risk_mgmt
        }

    @staticmethod
    def optimize_parameters(close: pd.Series,
                          min_period: int = 10,
                          max_period: int = 50,
                          min_std: float = 1.5,
                          max_std: float = 3.0) -> Dict[str, float]:
        """
        Otimiza os parâmetros (período e desvio padrão) para:
        - Maximizar o número de toques nas bandas
        - Minimizar falsos breakouts
        """
        def objective(params):
            period, std = params
            period = int(round(period))
            bands = BollingerBandsStrategy.calculate_bollinger_bands(close, period, std)
            
            # Conta toques válidos (preço toca banda e reverte)
            touches = 0
            for i in range(1, len(close)):
                if (close.iloc[i] >= bands['upper_band'].iloc[i] and close.iloc[i-1] < bands['upper_band'].iloc[i-1]) or \
                   (close.iloc[i] <= bands['lower_band'].iloc[i] and close.iloc[i-1] > bands['lower_band'].iloc[i-1]):
                    touches += 1
            
            # Penaliza parâmetros extremos
            penalty = abs(period - 20) * 0.1 + abs(std - 2.0) * 0.5
            return -(touches - penalty)  # Negativo para minimização
        
        from scipy.optimize import differential_evolution
        bounds = [(min_period, max_period), (min_std, max_std)]
        result = differential_evolution(objective, bounds)
        
        return {
            'optimal_period': int(round(result.x[0])),
            'optimal_std_dev': round(result.x[1], 2)
        }
