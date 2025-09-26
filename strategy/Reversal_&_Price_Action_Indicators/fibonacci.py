import pandas as pd
import numpy as np
from typing import Dict, List, Union

class FibonacciStrategy:
    """
    Implementação profissional de Fibonacci para trading com:
    - Retracements clássicos
    - Extensions
    - Projections
    - Expansões
    - Análise de clusters
    - Zonas de preço com confirmação
    """
    
    @staticmethod
    def calculate_retracements(high: float, low: float) -> Dict[str, float]:
        """
        Calcula os níveis clássicos de retracement de Fibonacci.
        Usado para identificar potenciais zonas de suporte/resistência.
        """
        difference = high - low
        return {
            '0%': high,
            '23.6%': high - difference * 0.236,
            '38.2%': high - difference * 0.382,
            '50%': high - difference * 0.5,
            '61.8%': high - difference * 0.618,
            '78.6%': high - difference * 0.786,
            '100%': low,
            '161.8%': low - difference * 0.618,
            '261.8%': low - difference * 1.618,
            '423.6%': low - difference * 2.618
        }

    @staticmethod
    def calculate_extensions(swing_high: float, 
                           swing_low: float, 
                           pullback_high: float) -> Dict[str, float]:
        """
        Calcula as extensões de Fibonacci para alvos de take-profit.
        Swing High → Swing Low → Pullback High
        """
        difference = swing_high - swing_low
        pullback_diff = swing_high - pullback_high
        return {
            '61.8%': pullback_high - difference * 0.618,
            '100%': pullback_high - difference * 1.0,
            '161.8%': pullback_high - difference * 1.618,
            '261.8%': pullback_high - difference * 2.618,
            '423.6%': pullback_high - difference * 4.236
        }

    @staticmethod
    def calculate_projections(swing_A_high: float,
                            swing_A_low: float,
                            swing_B_high: float,
                            swing_B_low: float) -> Dict[str, float]:
        """
        Projeções de Fibonacci (AB=CD pattern).
        """
        AB = swing_A_high - swing_A_low
        BC = swing_B_high - swing_B_low
        return {
            '127.2%': swing_B_low - AB * 1.272,
            '161.8%': swing_B_low - AB * 1.618,
            '200%': swing_B_low - AB * 2.0,
            '261.8%': swing_B_low - AB * 2.618
        }

    @staticmethod
    def find_clusters(prices: pd.Series, 
                     lookback: int = 100,
                     threshold: float = 0.02) -> Dict[str, Union[List, Dict]]:
        """
        Identifica clusters de Fibonacci (zonas onde múltiplos níveis convergem).
        """
        highs = prices.rolling(window=lookback).max()
        lows = prices.rolling(window=lookback).min()
        
        clusters = []
        for i in range(len(prices)):
            current_high = highs.iloc[i]
            current_low = lows.iloc[i]
            levels = FibonacciStrategy.calculate_retracements(current_high, current_low)
            
            for level_name, level_value in levels.items():
                # Verifica se o preço está próximo a este nível
                if abs(prices.iloc[i] - level_value) < threshold * prices.iloc[i]:
                    clusters.append({
                        'price': prices.iloc[i],
                        'level': level_name,
                        'timestamp': prices.index[i],
                        'high_ref': current_high,
                        'low_ref': current_low
                    })
        
        # Agrupa clusters próximos
        merged_clusters = []
        if clusters:
            current_cluster = {
                'min_price': clusters[0]['price'],
                'max_price': clusters[0]['price'],
                'levels': [clusters[0]['level']],
                'timestamps': [clusters[0]['timestamp']]
            }
            
            for point in clusters[1:]:
                if point['price'] <= current_cluster['max_price'] * (1 + threshold):
                    current_cluster['min_price'] = min(current_cluster['min_price'], point['price'])
                    current_cluster['max_price'] = max(current_cluster['max_price'], point['price'])
                    current_cluster['levels'].append(point['level'])
                    current_cluster['timestamps'].append(point['timestamp'])
                else:
                    merged_clusters.append(current_cluster)
                    current_cluster = {
                        'min_price': point['price'],
                        'max_price': point['price'],
                        'levels': [point['level']],
                        'timestamps': [point['timestamp']]
                    }
            
            merged_clusters.append(current_cluster)
        
        return {
            'clusters': merged_clusters,
            'most_significant': sorted(merged_clusters, 
                                     key=lambda x: len(x['levels']), 
                                     reverse=True)[:3] if merged_clusters else []
        }

    @staticmethod
    def full_analysis(swing_high: float,
                     swing_low: float,
                     current_price: float,
                     pullback_high: float = None,
                     swing_A_high: float = None,
                     swing_A_low: float = None,
                     swing_B_high: float = None,
                     swing_B_low: float = None) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa com todos os métodos Fibonacci + interpretação profissional.
        """
        retracements = FibonacciStrategy.calculate_retracements(swing_high, swing_low)
        
        analysis = {
            'retracements': retracements,
            'current_position': None,
            'nearest_level': None,
            'potential_support': [],
            'potential_resistance': []
        }
        
        # Determina posição atual em relação aos níveis
        for level, price in sorted(retracements.items(), key=lambda x: x[1]):
            if current_price >= price:
                analysis['current_position'] = f"above_{level}"
                analysis['nearest_level'] = (level, price)
                break
        
        # Identifica potenciais zonas de suporte/resistência
        for level, price in retracements.items():
            if current_price > price and (current_price - price) < 0.03 * current_price:
                analysis['potential_resistance'].append((level, price))
            elif current_price < price and (price - current_price) < 0.03 * current_price:
                analysis['potential_support'].append((level, price))
        
        # Adiciona extensões se houver pullback
        if pullback_high:
            extensions = FibonacciStrategy.calculate_extensions(swing_high, swing_low, pullback_high)
            analysis['extensions'] = extensions
            analysis['potential_targets'] = [
                (level, price) for level, price in extensions.items() 
                if price < current_price  # Para tendência de baixa
            ]
        
        # Adiciona projeções se houver swings AB
        if all([swing_A_high, swing_A_low, swing_B_high, swing_B_low]):
            projections = FibonacciStrategy.calculate_projections(swing_A_high, swing_A_low, swing_B_high, swing_B_low)
            analysis['projections'] = projections
            analysis['pattern_AB=CD'] = {
                'completed': current_price <= projections['161.8%'],
                'potential_reversal_zone': projections['161.8%']
            }
        
        # Interpretação profissional
        analysis['interpretation'] = FibonacciStrategy.interpret_analysis(analysis)
        
        return analysis
    
    @staticmethod
    def interpret_analysis(analysis: Dict) -> str:
        """Gera uma interpretação profissional dos dados Fibonacci"""
        current_pos = analysis['current_position']
        nearest_level = analysis['nearest_level']
        
        if not nearest_level:
            return "Mercado fora dos níveis Fibonacci significativos"
        
        level_name, level_price = nearest_level
        
        # Tendência de alta
        if 'above' in current_pos:
            if level_name in ['23.6%', '38.2%']:
                return f"Tendência forte - Correção rasa ({level_name}) - Buscar oportunidades de compra"
            elif level_name == '50%':
                return f"Correção moderada - Zona de compra com stop abaixo de 61.8%"
            elif level_name == '61.8%':
                return f"Correção profunda - Confirmar reversão com preço acima de 50%"
            elif level_name == '78.6%':
                return f"Zona de perigo - Ruptura pode invalidar tendência"
        
        # Tendência de baixa
        else:
            if level_name in ['23.6%', '38.2%']:
                return f"Retração fraca - Potencial continuação de baixa"
            elif level_name == '50%':
                return f"Zona de venda com stop acima de 38.2%"
            elif level_name == '61.8%':
                return f"Retração forte - Confirmar continuação com preço abaixo de 50%"
        
        return "Aguardar confirmação de preço próximo aos níveis chave"
