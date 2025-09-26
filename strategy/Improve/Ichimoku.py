import pandas as pd
import numpy as np
from typing import Dict, Union

class IchimokuStrategy:
    """
    Implementação profissional do Ichimoku Kinko Hyo (Cloud) com:
    - Todos os 5 componentes originais
    - Sinais de trading baseados nas regras tradicionais
    - Análise da nuvem (Kumo)
    - Cálculo de suportes/resistências
    - Geração de sinais conforme usado pelos traders
    """
    
    @staticmethod
    def calculate_components(high: pd.Series, 
                           low: pd.Series, 
                           close: pd.Series,
                           conversion_period: int = 9,
                           base_period: int = 26,
                           leading_span_b_period: int = 52,
                           displacement: int = 26) -> Dict[str, pd.Series]:
        """
        Calcula os 5 componentes do Ichimoku:
        1. Tenkan-sen (Conversion Line)
        2. Kijun-sen (Base Line)
        3. Senkou Span A (Leading Span A)
        4. Senkou Span B (Leading Span B)
        5. Chikou Span (Lagging Span)
        
        Parâmetros seguem o padrão original (9-26-52).
        """
        # 1. Tenkan-sen (Conversion Line)
        conversion = (high.rolling(window=conversion_period).max() + 
                     low.rolling(window=conversion_period).min()) / 2
        
        # 2. Kijun-sen (Base Line)
        base = (high.rolling(window=base_period).max() + 
               low.rolling(window=base_period).min()) / 2
        
        # 3. Senkou Span A (Leading Span A)
        leading_span_a = ((conversion + base) / 2).shift(displacement)
        
        # 4. Senkou Span B (Leading Span B)
        leading_span_b = ((high.rolling(window=leading_span_b_period).max() + 
                          low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)
        
        # 5. Chikou Span (Lagging Span)
        chikou = close.shift(-displacement)
        
        return {
            'conversion': conversion,
            'base': base,
            'leading_span_a': leading_span_a,
            'leading_span_b': leading_span_b,
            'chikou': chikou
        }
    
    @staticmethod
    def generate_signals(high: pd.Series, 
                        low: pd.Series, 
                        close: pd.Series,
                        conversion: pd.Series,
                        base: pd.Series,
                        leading_span_a: pd.Series,
                        leading_span_b: pd.Series,
                        chikou: pd.Series) -> Dict[str, Union[bool, str, float]]:
        """
        Gera sinais de trading baseados nas regras tradicionais do Ichimoku:
        - Sinal TK Cross (Conversão vs Base)
        - Sinal Kumo Breakout
        - Sinal Chikou Confirmation
        - Sinal Kumo Twist
        """
        current_price = close.iloc[-1]
        cloud_top = np.maximum(leading_span_a.iloc[-1], leading_span_b.iloc[-1])
        cloud_bottom = np.minimum(leading_span_a.iloc[-1], leading_span_b.iloc[-1])
        
        # 1. Sinal TK Cross (Conversão cruzando a Base)
        tk_cross = None
        if conversion.iloc[-1] > base.iloc[-1] and conversion.iloc[-2] <= base.iloc[-2]:
            tk_cross = "golden_cross"
        elif conversion.iloc[-1] < base.iloc[-1] and conversion.iloc[-2] >= base.iloc[-2]:
            tk_cross = "dead_cross"
        
        # 2. Sinal Kumo Breakout
        kumo_breakout = None
        if current_price > cloud_top and close.iloc[-2] <= cloud_top:
            kumo_breakout = "bullish_breakout"
        elif current_price < cloud_bottom and close.iloc[-2] >= cloud_bottom:
            kumo_breakout = "bearish_breakout"
        
        # 3. Confirmação do Chikou Span
        chikou_confirmation = None
        if chikou.iloc[-displacement] > close.iloc[-displacement]:
            chikou_confirmation = "bullish_confirmed"
        elif chikou.iloc[-displacement] < close.iloc[-displacement]:
            chikou_confirmation = "bearish_confirmed"
        
        # 4. Kumo Twist (Mudança na cor da nuvem futura)
        kumo_twist = None
        if (leading_span_a.iloc[-displacement] > leading_span_b.iloc[-displacement] and
            leading_span_a.iloc[-displacement-1] <= leading_span_b.iloc[-displacement-1]):
            kumo_twist = "bullish_twist"
        elif (leading_span_a.iloc[-displacement] < leading_span_b.iloc[-displacement] and
              leading_span_a.iloc[-displacement-1] >= leading_span_b.iloc[-displacement-1]):
            kumo_twist = "bearish_twist"
        
        # Sinal Composto (Regra profissional)
        strong_buy = all([
            tk_cross == "golden_cross",
            current_price > cloud_top,
            chikou_confirmation == "bullish_confirmed",
            kumo_twist == "bullish_twist"
        ])
        
        strong_sell = all([
            tk_cross == "dead_cross",
            current_price < cloud_bottom,
            chikou_confirmation == "bearish_confirmed",
            kumo_twist == "bearish_twist"
        ])
        
        return {
            'signals': {
                'tk_cross': tk_cross,
                'kumo_breakout': kumo_breakout,
                'chikou_confirmation': chikou_confirmation,
                'kumo_twist': kumo_twist,
                'composite_signal': "strong_buy" if strong_buy else 
                                  "strong_sell" if strong_sell else "neutral"
            },
            'cloud_metrics': {
                'cloud_top': round(cloud_top, 5),
                'cloud_bottom': round(cloud_bottom, 5),
                'cloud_color': "green" if leading_span_a.iloc[-1] > leading_span_b.iloc[-1] else "red",
                'cloud_thickness': round(abs(leading_span_a.iloc[-1] - leading_span_b.iloc[-1]), 5)
            },
            'price_position': {
                'relative_to_cloud': "above" if current_price > cloud_top else 
                                   "below" if current_price < cloud_bottom else "inside",
                'distance_to_cloud': round(abs(current_price - (cloud_top if current_price > cloud_top else cloud_bottom)) / current_price * 100, 2)
            }
        }
    
    @staticmethod
    def full_analysis(high: pd.Series, 
                     low: pd.Series, 
                     close: pd.Series,
                     conversion_period: int = 9,
                     base_period: int = 26,
                     leading_span_b_period: int = 52,
                     displacement: int = 26) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do Ichimoku em um único método, retornando:
        - Componentes calculados
        - Sinais de trading
        - Métricas da nuvem
        - Posição do preço
        """
        # Calcula todos os componentes
        components = IchimokuPure.calculate_components(
            high, low, close,
            conversion_period,
            base_period,
            leading_span_b_period,
            displacement
        )
        
        # Gera os sinais
        signals = IchimokuPure.generate_signals(
            high, low, close,
            components['conversion'],
            components['base'],
            components['leading_span_a'],
            components['leading_span_b'],
            components['chikou']
        )
        
        # Formatação profissional dos resultados
        return {
            'components': {
                'conversion_line': round(components['conversion'].iloc[-1], 5),
                'base_line': round(components['base'].iloc[-1], 5),
                'leading_span_a': round(components['leading_span_a'].iloc[-1], 5),
                'leading_span_b': round(components['leading_span_b'].iloc[-1], 5),
                'chikou_span': round(components['chikou'].iloc[-displacement], 5)
            },
            'signals': signals['signals'],
            'cloud_analysis': signals['cloud_metrics'],
            'price_analysis': signals['price_position'],
            'interpretation': IchimokuPure.interpret_signals(signals['signals'])
        }
    
    @staticmethod
    def interpret_signals(signals: Dict) -> str:
        """Interpretação profissional dos sinais para tomada de decisão"""
        composite = signals['composite_signal']
        
        if composite == "strong_buy":
            return "Tendência de alta forte - Entrada recomendada com stop abaixo da Base Line"
        elif composite == "strong_sell":
            return "Tendência de baixa forte - Entrada recomendada com stop acima da Base Line"
        else:
            if signals['tk_cross'] == "golden_cross" and signals['price_position']['relative_to_cloud'] == "above":
                return "Sinal de compra moderado - Confirmar com volume"
            elif signals['tk_cross'] == "dead_cross" and signals['price_position']['relative_to_cloud'] == "below":
                return "Sinal de venda moderado - Confirmar com volume"
            else:
                return "Mercado em consolidação - Aguardar confirmação"
