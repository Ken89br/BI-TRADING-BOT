import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from data.data_client import FallbackDataClient
from data_processor import MarketAwareDataProcessor

class TopDownDataManager:
    """
    Gerenciador de dados para análise Top-Down (Macro → Micro)
    """
    
    # CONFIGURAÇÃO TOP-DOWN HIERÁRQUICA
    TOP_DOWN_STRATEGIES = {
        'position_trading': [
            {
                'main_tf': '1W', 
                'confirmation_tf': '1D', 
                'entry_tf': '6H',
                'timeframes': ['1W', '1D', '6H'],  # Todos necessários
                'limits': {'1W': 52, '1D': 30, '6H': 40}  # 1 ano, 1 mês, 10 dias
            }
        ],
        
        'swing_trading': [
            {
                'main_tf': '1D', 
                'confirmation_tf': '4H', 
                'entry_tf': '1H',
                'timeframes': ['1D', '4H', '1H'],
                'limits': {'1D': 30, '4H': 42, '1H': 24}  # 1 mês, 1 semana, 1 dia
            }
        ],
        
        'day_trading': [
            {
                'main_tf': '6H', 
                'confirmation_tf': '2H', 
                'entry_tf': '30min',
                'timeframes': ['6H', '2H', '30min'],
                'limits': {'6H': 20, '2H': 30, '30min': 48}  # 5 dias, 2.5 dias, 1 dia
            },
            {
                'main_tf': '4H', 
                'confirmation_tf': '1H', 
                'entry_tf': '15min',
                'timeframes': ['4H', '1H', '15min'],
                'limits': {'4H': 30, '1H': 24, '15min': 96}  # 5 dias, 1 dia, 1 dia
            }
        ],
        
        'scalping': [
            {
                'main_tf': '1H', 
                'confirmation_tf': '30min', 
                'entry_tf': '5min',
                'timeframes': ['1H', '30min', '5min'],
                'limits': {'1H': 24, '30min': 48, '5min': 144}  # 1 dia, 1 dia, 12 horas
            },
            {
                'main_tf': '30min', 
                'confirmation_tf': '15min', 
                'entry_tf': '3min',
                'timeframes': ['30min', '15min', '3min'],
                'limits': {'30min': 48, '15min': 96, '3min': 160}  # 1 dia, 1 dia, 8 horas
            }
        ],
        
        'ultra_scalping': [
            {
                'main_tf': '15min', 
                'confirmation_tf': '5min', 
                'entry_tf': '1min',
                'timeframes': ['15min', '5min', '1min'],
                'limits': {'15min': 96, '5min': 144, '1min': 240}  # 1 dia, 12 horas, 4 horas
            },
            {
                'main_tf': '5min', 
                'confirmation_tf': '3min', 
                'entry_tf': '1min',
                'timeframes': ['5min', '3min', '1min'],
                'limits': {'5min': 144, '3min': 240, '1min': 360}  # 12 horas, 12 horas, 6 horas
            }
        ]
    }
    
    # MAPEAMENTO CORRIGIDO - apenas timeframes principais válidos
    TIMEFRAME_TO_STRATEGY = {
        '1W': 'position_trading',
        '1D': 'swing_trading', 
        '6H': 'day_trading',
        '4H': 'day_trading',
        '1H': 'scalping',
        '30min': 'scalping',
        '15min': 'ultra_scalping',
        '5min': 'ultra_scalping',
    }

    def __init__(self):
        self.data_client = FallbackDataClient()
        self.processor = MarketAwareDataProcessor()
        self._cache = {}
        self._cache_ttl = {
            'position_trading': 3600,    # 1 hora
            'swing_trading': 1800,       # 30 minutos
            'day_trading': 600,          # 10 minutos
            'scalping': 300,             # 5 minutos
            'ultra_scalping': 60         # 1 minuto
        }

    def _detect_strategy_from_timeframe(self, timeframe: str) -> str:
        """Detecta a estratégia automaticamente baseado no timeframe principal"""
        return self.TIMEFRAME_TO_STRATEGY.get(timeframe, 'day_trading')

    def get_top_down_data(self, symbol: str, main_tf: str, 
                         strategy_type: str = None) -> Dict[str, pd.DataFrame]:
        """
        Obtém dados hierárquicos para análise Top-Down
        
        Args:
            symbol: Par a ser analisado
            main_tf: Timeframe principal (ex: '1H', '4H', '1D')
            strategy_type: Tipo de estratégia (opcional, detecta automaticamente)
        
        Returns:
            Dict com 3 timeframes: main_tf, confirmation_tf, entry_tf
            Ex: {'15min': DataFrame, '5min': DataFrame, '1min': DataFrame}
        """
        if not strategy_type:
            strategy_type = self._detect_strategy_from_timeframe(main_tf)
        
        # VALIDAÇÃO: Verificar se o timeframe é principal válido
        if main_tf not in self.TIMEFRAME_TO_STRATEGY:
            raise ValueError(f"Timeframe {main_tf} não é um timeframe principal válido. "
                           f"Use: {list(self.TIMEFRAME_TO_STRATEGY.keys())}")
        
        # Encontrar a configuração correta
        strategy_configs = self.TOP_DOWN_STRATEGIES.get(strategy_type, [])
        selected_config = None
        
        for config in strategy_configs:
            if config['main_tf'] == main_tf:
                selected_config = config
                break
        
        if not selected_config:
            # Fallback: usa a primeira configuração da estratégia
            selected_config = strategy_configs[0] if strategy_configs else None
        
        if not selected_config:
            raise ValueError(f"Configuração não encontrada para {main_tf} em {strategy_type}")
        
        return self._fetch_hierarchical_data(symbol, selected_config, strategy_type)

    def _fetch_hierarchical_data(self, symbol: str, config: Dict, strategy_type: str) -> Dict[str, pd.DataFrame]:
        """
        Busca dados hierárquicos (Macro → Micro)
        RETORNA SEMPRE 3 TIMEFRAMES: main, confirmation, entry
        """
        cache_key = f"topdown_{symbol}_{config['main_tf']}_{strategy_type}"
        
        if self._is_cache_valid(cache_key, strategy_type):
            return self._cache[cache_key]
        
        results = {}
        timeframes = config['timeframes']
        limits = config['limits']
        
        print(f"🎯 Buscando dados Top-Down: {symbol} | "
              f"Main: {config['main_tf']} | Confirmation: {config['confirmation_tf']} | "
              f"Entry: {config['entry_tf']}")
        
        try:
            with ThreadPoolExecutor(max_workers=len(timeframes)) as executor:
                futures = {}
                
                for tf in timeframes:
                    limit = limits.get(tf, 50)
                    future = executor.submit(
                        self._get_single_timeframe_data,
                        symbol, tf, limit, strategy_type
                    )
                    futures[tf] = future
                
                # Coletar na ordem hierárquica (Macro → Micro)
                for tf in timeframes:
                    try:
                        df = futures[tf].result(timeout=30)
                        results[tf] = df if df is not None else pd.DataFrame()
                        
                        if not df.empty:
                            print(f"   ✅ {tf}: {len(df)} candles")
                        else:
                            print(f"   ⚠️ {tf}: dados vazios")
                            
                    except Exception as e:
                        print(f"   ❌ {tf}: erro - {e}")
                        results[tf] = pd.DataFrame()
            
            # Validar que temos dados suficientes
            valid_timeframes = [tf for tf, df in results.items() if not df.empty]
            if len(valid_timeframes) >= 10:  # Mínimo 10 timeframes para análise
                self._cache[cache_key] = results
                self._cache[f"{cache_key}_timestamp"] = datetime.now()
                print(f"✅ Top-Down carregado: {symbol} {config['main_tf']} → {valid_timeframes}")
            else:
                print(f"⚠️ Dados insuficientes para análise Top-Down: {symbol}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro crítico no Top-Down para {symbol}: {e}")
            return {tf: pd.DataFrame() for tf in timeframes}

    def get_available_main_timeframes(self, strategy_type: str = None) -> List[str]:
        """
        Retorna lista de timeframes principais disponíveis
        Útil para UI/UX - mostrar apenas timeframes válidos
        """
        if strategy_type:
            strategies = [strategy_type] if strategy_type in self.TOP_DOWN_STRATEGIES else []
        else:
            strategies = self.TOP_DOWN_STRATEGIES.keys()
        
        main_timeframes = []
        for strategy in strategies:
            for config in self.TOP_DOWN_STRATEGIES[strategy]:
                main_timeframes.append(config['main_tf'])
        
        return sorted(list(set(main_timeframes)))

    def get_strategy_info(self, main_tf: str) -> Dict:
        """
        Retorna informações completas da estratégia para um timeframe principal
        """
        strategy_type = self._detect_strategy_from_timeframe(main_tf)
        strategy_configs = self.TOP_DOWN_STRATEGIES.get(strategy_type, [])
        
        for config in strategy_configs:
            if config['main_tf'] == main_tf:
                return {
                    'strategy_type': strategy_type,
                    'main_tf': config['main_tf'],
                    'confirmation_tf': config['confirmation_tf'],
                    'entry_tf': config['entry_tf'],
                    'all_timeframes': config['timeframes'],
                    'description': self._get_strategy_description(strategy_type)
                }
        
        return {}

    def _get_strategy_description(self, strategy_type: str) -> str:
        """Descrição amigável das estratégias"""
        descriptions = {
            'position_trading': 'Long-term trends (weeks to months)',
            'swing_trading': 'Medium-term swings (days to weeks)', 
            'day_trading': 'Intraday opportunities (hours to days)',
            'scalping': 'Short-term precision (minutes to hours)',
            'ultra_scalping': 'High-frequency opportunities (seconds to minutes)'
        }
        return descriptions.get(strategy_type, 'Unknown strategy')

    # MÉTODOS DE CONVENIÊNCIA (agora com nomes padronizados)
    def get_support_resistance_data(self, symbol: str, main_tf: str) -> Dict[str, pd.DataFrame]:
        """Dados para análise de Support/Resistance com confirmação hierárquica"""
        return self.get_top_down_data(symbol, main_tf)

    def get_trend_analysis_data(self, symbol: str, main_tf: str) -> Dict[str, pd.DataFrame]:
        """Dados para análise de tendência (Macro → Micro)"""
        return self.get_top_down_data(symbol, main_tf)

    def get_pattern_recognition_data(self, symbol: str, main_tf: str) -> Dict[str, pd.DataFrame]:
        """Dados para reconhecimento de padrões com confirmação multi-timeframe"""
        return self.get_top_down_data(symbol, main_tf)

    # MÉTODO SINGLE TIMEFRAME (base)
    def _get_single_timeframe_data(self, symbol: str, timeframe: str, limit: int, 
                                  operation_type: str) -> Optional[pd.DataFrame]:
        """Busca e processa dados de um único timeframe"""
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if cache_key in self._cache and self._is_cache_valid(cache_key, operation_type):
            return self._cache[cache_key]
        
        try:
            raw_data = self.data_client.fetch_candles(symbol, interval=timeframe, limit=limit)
            if not raw_data or "history" not in raw_data:
                return None
            
            processed_df = self.processor.process_raw_data_from_client(
                raw_data, symbol, operation_type
            )
            
            self._cache[cache_key] = processed_df
            return processed_df
            
        except Exception as e:
            print(f"❌ Erro em {symbol} {timeframe}: {e}")
            return None

    def _is_cache_valid(self, cache_key: str, strategy_type: str) -> bool:
        """Verifica validade do cache baseado na estratégia"""
        if cache_key not in self._cache:
            return False
            
        cache_time = self._cache.get(f"{cache_key}_timestamp")
        if not cache_time:
            return False
        
        ttl = self._cache_ttl.get(strategy_type, 300)
        return (datetime.now() - cache_time).total_seconds() < ttl

    def clear_strategy_cache(self, symbol: str = None, strategy_type: str = None):
        """Limpa cache de forma seletiva"""
        keys_to_remove = []
        
        for key in self._cache.keys():
            if key.startswith('_timestamp'):
                continue
                
            if symbol and symbol not in key:
                continue
                
            if strategy_type and strategy_type not in key:
                continue
                
            keys_to_remove.append(key)
            keys_to_remove.append(f"{key}_timestamp")
        
        for key in keys_to_remove:
            self._cache.pop(key, None)

# Instância global para uso em todos os módulos
data_manager = TopDownDataManager()
