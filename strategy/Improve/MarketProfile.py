import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class MarketProfilePro:
    """
    Implementação profissional do Market Profile com:
    - Cálculo do POC (Point of Control), VAL (Value Area Low), VAH (Value Area High)
    - Perfil de volume por preço
    - Análise de desenvolvimento de mercado (iniciativa vs. aceitação)
    - Identificação de estruturas de mercado (balanceamento, tendência)
    - Integração com volume para confirmação
    """

    @staticmethod
    def calculate_market_profile(high: pd.Series,
                                low: pd.Series,
                                close: pd.Series,
                                volume: pd.Series,
                                tick_size: float = 0.25,
                                value_area_percentage: float = 0.70) -> Dict[str, Union[Dict, pd.Series]]:
        """
        Calcula o Market Profile completo para um DataFrame de candles.
        
        Parâmetros:
        - high: Série de máximos
        - low: Série de mínimos
        - close: Série de fechamentos
        - volume: Série de volumes
        - tick_size: Tamanho do tick para agrupar preços (ex: 0.25 para ouro)
        - value_area_percentage: % da área de valor (padrão 70%)
        
        Retorna:
        - price_volume: Dicionário {preço: volume_total}
        - poc: Point of Control (preço com maior volume)
        - value_area: Dicionário com VAH, VAL e largura
        - market_structure: Tipo de estrutura identificada
        """
        # 1. Agrupa volume por nível de preço (com tick_size)
        price_bins = np.arange(low.min(), high.max() + tick_size, tick_size)
        volume_per_price = pd.cut(close, bins=price_bins, labels=price_bins[:-1]).to_frame()
        volume_per_price['volume'] = volume
        price_volume = volume_per_price.groupby(volume_per_price.columns[0])['volume'].sum().to_dict()
        
        # 2. Calcula POC (Point of Control)
        poc_price = max(price_volume, key=price_volume.get)
        total_volume = sum(price_volume.values())
        
        # 3. Calcula Área de Valor (Value Area)
        sorted_prices = sorted(price_volume.items(), key=lambda x: x[1], reverse=True)
        cumulative_volume = 0
        value_area_prices = []
        
        for price, vol in sorted_prices:
            if cumulative_volume <= total_volume * value_area_percentage:
                value_area_prices.append(price)
                cumulative_volume += vol
            else:
                break
        
        vah = max(value_area_prices)
        val = min(value_area_prices)
        
        # 4. Identifica Estrutura de Mercado
        market_structure = MarketProfilePro._identify_market_structure(
            poc_price, vah, val, close.iloc[-1], price_volume
        )
        
        return {
            'price_volume': price_volume,
            'poc': poc_price,
            'value_area': {
                'vah': vah,
                'val': val,
                'width': vah - val
            },
            'market_structure': market_structure,
            'profile_range': {
                'high': max(price_volume.keys()),
                'low': min(price_volume.keys())
            }
        }

    @staticmethod
    def _identify_market_structure(poc: float,
                                  vah: float,
                                  val: float,
                                  current_price: float,
                                  price_volume: Dict[float, float]) -> str:
        """
        Classifica a estrutura do mercado com base no perfil:
        - Balanceamento (Balanço)
        - Tendência (Iniciativa)
        - Falha de Aceitação (Rejeição)
        """
        range_width = vah - val
        avg_volume = np.mean(list(price_volume.values()))
        
        # Critérios para balanceamento
        if range_width <= 0.5 * (max(price_volume.keys()) - min(price_volume.keys())):
            if current_price >= val and current_price <= vah:
                return "balanceamento"
        
        # Critérios para tendência
        if current_price > vah and price_volume.get(vah, 0) < 0.3 * avg_volume:
            return "tendência_de_alta"
        elif current_price < val and price_volume.get(val, 0) < 0.3 * avg_volume:
            return "tendência_de_baixa"
        
        # Falha de aceitação
        if current_price > vah and price_volume.get(current_price, 0) < 0.2 * avg_volume:
            return "falha_de_aceitação_alta"
        elif current_price < val and price_volume.get(current_price, 0) < 0.2 * avg_volume:
            return "falha_de_aceitação_baixa"
        
        return "indefinido"

    @staticmethod
    def generate_signals(high: pd.Series,
                        low: pd.Series,
                        close: pd.Series,
                        volume: pd.Series,
                        tick_size: float = 0.25) -> Dict[str, Union[str, float]]:
        """
        Gera sinais de trading baseados no Market Profile:
        - Breakout de Área de Valor
        - Teste do POC com volume
        - Rejeição de extremos (VAH/VAL)
        """
        profile = MarketProfilePro.calculate_market_profile(high, low, close, volume, tick_size)
        poc = profile['poc']
        vah = profile['value_area']['vah']
        val = profile['value_area']['val']
        current_price = close.iloc[-1]
        
        # Sinal de Breakout
        if current_price > vah and volume.iloc[-1] > 1.5 * volume.mean():
            signal = "breakout_alta"
        elif current_price < val and volume.iloc[-1] > 1.5 * volume.mean():
            signal = "breakout_baixa"
        # Sinal de Rejeição
        elif (current_price >= vah and 
              volume.iloc[-1] < 0.5 * volume.mean() and
              high.iloc[-1] - close.iloc[-1] > 0.3 * (high.iloc[-1] - low.iloc[-1])):
            signal = "rejeição_alta"
        elif (current_price <= val and 
              volume.iloc[-1] < 0.5 * volume.mean() and
              close.iloc[-1] - low.iloc[-1] > 0.3 * (high.iloc[-1] - low.iloc[-1])):
            signal = "rejeição_baixa"
        else:
            signal = None
        
        return {
            'signal': signal,
            'current_price': current_price,
            'poc': poc,
            'vah': vah,
            'val': val,
            'structure': profile['market_structure']
        }

    @staticmethod
    def full_analysis(high: pd.Series,
                      low: pd.Series,
                      close: pd.Series,
                      volume: pd.Series,
                      tick_size: float = 0.25) -> Dict[str, Union[Dict, str]]:
        """
        Análise completa do Market Profile (estilo institucional).
        Retorna:
        - Perfil de preço-volume
        - Área de valor e POC
        - Sinais de trading
        - Estrutura de mercado
        """
        profile = MarketProfilePro.calculate_market_profile(high, low, close, volume, tick_size)
        signals = MarketProfilePro.generate_signals(high, low, close, volume, tick_size)
        
        return {
            'profile': {
                'poc': profile['poc'],
                'value_area': profile['value_area'],
                'price_range': profile['profile_range'],
                'volume_distribution': profile['price_volume']
            },
            'signals': signals,
            'market_structure': profile['market_structure']
        }