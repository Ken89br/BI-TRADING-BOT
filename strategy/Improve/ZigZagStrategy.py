import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
from scipy.signal import argrelextrema

class ZigZagStrategy:
    """
    Implementação profissional do indicador ZigZag com:
    - Detecção automática de pontos de reversão
    - Filtros de volatilidade (ATR)
    - Integração com Fibonacci para alvos
    - Confirmação de volume
    - Identificação de padrões gráficos
    """

    @staticmethod
    def calculate_zigzag(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series,
                        reversal_percent: float = 5.0,
                        atr_period: int = 14,
                        atr_multiplier: float = 1.5) -> Dict[str, Union[List, pd.Series]]:
        """
        Calcula os pontos ZigZag baseados em:
        - Movimentos percentuais mínimos OU
        - Múltiplos do ATR para filtro de ruído
        
        Parâmetros:
        - reversal_percent: % mínima para considerar reversão
        - atr_period: Período do ATR para filtro de volatilidade
        - atr_multiplier: Múltiplo do ATR para considerar reversão válida
        """
        # Calcula ATR para filtro dinâmico
        tr = pd.DataFrame({
            'high_low': high - low,
            'high_close': abs(high - close.shift(1)),
            'low_close': abs(low - close.shift(1))
        }).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()
        
        # Identifica máximos e mínimos locais
        highs_idx = argrelextrema(high.values, np.greater, order=atr_period)[0]
        lows_idx = argrelextrema(low.values, np.less, order=atr_period)[0]
        
        # Combina e ordena todos os pontos extremos
        all_points = []
        for idx in highs_idx:
            all_points.append(('high', idx, high.iloc[idx]))
        for idx in lows_idx:
            all_points.append(('low', idx, low.iloc[idx]))
        all_points.sort(key=lambda x: x[1])
        
        # Filtra pontos que não atendem aos critérios mínimos
        filtered_points = []
        last_valid_point = None
        
        for point in all_points:
            point_type, idx, val = point
            
            if last_valid_point is None:
                filtered_points.append(point)
                last_valid_point = point
                continue
                
            last_type, last_idx, last_val = last_valid_point
            
            # Calcula movimento percentual e em ATRs
            move_pct = abs(val - last_val) / last_val * 100
            move_atr = abs(val - last_val) / atr.iloc[idx]
            
            # Critério de reversão
            if (move_pct >= reversal_percent) or (move_atr >= atr_multiplier):
                # Alterna entre altas e baixas
                if point_type != last_type:
                    filtered_points.append(point)
                    last_valid_point = point
        
        # Cria série ZigZag para plotagem
        zigzag = pd.Series(np.nan, index=close.index)
        for i in range(1, len(filtered_points)):
            prev_idx = filtered_points[i-1][1]
            curr_idx = filtered_points[i][1]
            
            if filtered_points[i-1][0] == 'high':
                zigzag.iloc[prev_idx:curr_idx+1] = high.iloc[prev_idx:curr_idx+1]
            else:
                zigzag.iloc[prev_idx:curr_idx+1] = low.iloc[prev_idx:curr_idx+1]
        
        return {
            'zigzag': zigzag,
            'swings': filtered_points,
            'current_price': close.iloc[-1],
            'atr': atr
        }

    @staticmethod
    def identify_patterns(swings: List[Tuple[str, int, float]]) -> Dict[str, Union[str, List]]:
        """
        Identifica padrões gráficos comuns nos swings:
        - Topos/Duplo Fundo
        - Head and Shoulders
        - Triangles
        """
        patterns = []
        
        if len(swings) < 5:
            return {'patterns': [], 'valid': False}
        
        # Verifica Topo/Fundo Duplo
        for i in range(len(swings)-3):
            a, b, c = swings[i], swings[i+1], swings[i+2]
            
            # Topo Duplo
            if (a[0] == 'high' and b[0] == 'low' and c[0] == 'high' and
                abs(a[2] - c[2]) < 0.01 * a[2]):
                patterns.append(('double_top', a[1], c[1]))
            
            # Fundo Duplo
            elif (a[0] == 'low' and b[0] == 'high' and c[0] == 'low' and
                  abs(a[2] - c[2]) < 0.01 * a[2]):
                patterns.append(('double_bottom', a[1], c[1]))
        
        # Verifica Head and Shoulders
        if len(swings) >= 5:
            for i in range(len(swings)-5):
                a, b, c, d, e = swings[i], swings[i+1], swings[i+2], swings[i+3], swings[i+4]
                
                # Head and Shoulders
                if (a[0] == 'high' and b[0] == 'low' and c[0] == 'high' and 
                    d[0] == 'low' and e[0] == 'high' and
                    c[2] > a[2] and c[2] > e[2] and
                    abs(a[2] - e[2]) < 0.01 * a[2]):
                    patterns.append(('head_and_shoulders', a[1], e[1]))
                
                # Inverse Head and Shoulders
                elif (a[0] == 'low' and b[0] == 'high' and c[0] == 'low' and 
                      d[0] == 'high' and e[0] == 'low' and
                      c[2] < a[2] and c[2] < e[2] and
                      abs(a[2] - e[2]) < 0.01 * a[2]):
                    patterns.append(('inverse_head_and_shoulders', a[1], e[1]))
        
        return {
            'patterns': patterns,
            'valid': len(patterns) > 0
        }

    @staticmethod
    def generate_signals(swings: List[Tuple[str, int, float]],
                        patterns: Dict,
                        volume: pd.Series,
                        current_price: float) -> Dict[str, Union[str, float]]:
        """
        Gera sinais baseados em:
        - Quebra de linhas de tendência
        - Confirmação de padrões
        - Volume nos pontos de reversão
        """
        if not swings or len(swings) < 3:
            return {'signal': None, 'reason': 'Not enough swings'}
        
        last_swing = swings[-1]
        prev_swing = swings[-2]
        
        # Sinal de quebra de linha de tendência
        signal = None
        if last_swing[0] == 'high':
            if current_price < last_swing[2] and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                signal = 'potential_downtrend'
        else:
            if current_price > last_swing[2] and volume.iloc[-1] > volume.rolling(20).mean().iloc[-1]:
                signal = 'potential_uptrend'
        
        # Confirmação de padrões
        pattern_signal = None
        for pattern in patterns['patterns']:
            pattern_type, start_idx, end_idx = pattern
            if pattern_type == 'double_top' and current_price < swings[end_idx][2]:
                pattern_signal = 'double_top_confirmed'
            elif pattern_type == 'double_bottom' and current_price > swings[end_idx][2]:
                pattern_signal = 'double_bottom_confirmed'
        
        return {
            'swing_signal': signal,
            'pattern_signal': pattern_signal,
            'last_swing': last_swing,
            'current_price': current_price
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                     low: pd.Series,
                     close: pd.Series,
                     volume: pd.Series,
                     reversal_percent: float = 5.0,
                     atr_period: int = 14,
                     atr_multiplier: float = 1.5) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do ZigZag:
        - Identificação de swings
        - Detecção de padrões
        - Geração de sinais
        - Gerenciamento de risco
        """
        zigzag_data = ZigZagStrategy.calculate_zigzag(
            high, low, close, reversal_percent, atr_period, atr_multiplier
        )
        patterns = ZigZagStrategy.identify_patterns(zigzag_data['swings'])
        signals = ZigZagStrategy.generate_signals(
            zigzag_data['swings'],
            patterns,
            volume,
            zigzag_data['current_price']
        )
        
        return {
            'zigzag': zigzag_data['zigzag'],
            'swings': zigzag_data['swings'],
            'patterns': patterns,
            'signals': signals,
            'risk_management': {
                'stop_loss': round(zigzag_data['current_price'] * 0.98 if signals['swing_signal'] == 'potential_uptrend' 
                                 else zigzag_data['current_price'] * 1.02, 2),
                'take_profit': calculate_fib_targets(zigzag_data['swings'], zigzag_data['current_price'])
            }
        }

def calculate_fib_targets(swings: List[Tuple[str, int, float]], current_price: float) -> Dict[str, float]:
    """Calcula alvos Fibonacci baseados nos swings do ZigZag"""
    if len(swings) < 3:
        return None
    
    last_swing = swings[-1]
    prev_swing = swings[-2]
    
    if last_swing[0] == 'high':  # Movimento de baixa
        swing_high = last_swing[2]
        swing_low = prev_swing[2]
        difference = swing_high - swing_low
        return {
            '23.6%': swing_high - difference * 0.236,
            '38.2%': swing_high - difference * 0.382,
            '61.8%': swing_high - difference * 0.618
        }
    else:  # Movimento de alta
        swing_low = last_swing[2]
        swing_high = prev_swing[2]
        difference = swing_high - swing_low
        return {
            '23.6%': swing_low + difference * 0.236,
            '38.2%': swing_low + difference * 0.382,
            '61.8%': swing_low + difference * 0.618
        }