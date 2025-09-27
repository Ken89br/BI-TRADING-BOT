import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from data.data_client import FallbackDataClient
from data_processor import MarketAwareDataProcessor

class TopDownDataManager:
    """
    Gerenciador de dados para análise Top-Down (Macro → Micro)
    """
    
    # CONFIGURAÇÃO TOP-DOWN HIERÁRQUICA
    TRADING_STYLE = {
        'position_trading': [
            {
                'trend_tf': '1W', 
                'confirmation_tf': '1D', 
                'entry_tf': '6H',
                'timeframes': ['1W', '1D', '6H'],
                'limits': {'1W': 52, '1D': 30, '6H': 40}
            }
        ],
        
        'swing_trading': [
            {
                'trend_tf': '1D', 
                'confirmation_tf': '4H', 
                'entry_tf': '1H',
                'timeframes': ['1D', '4H', '1H'],
                'limits': {'1D': 30, '4H': 42, '1H': 24}  # 1 mês, 1 semana, 1 dia
            }
        ],
        
        'day_trading': [
            {
                'trend_tf': '6H', 
                'confirmation_tf': '2H', 
                'entry_tf': '30min',
                'timeframes': ['6H', '2H', '30min'],
                'limits': {'6H': 20, '2H': 30, '30min': 48}  # 5 dias, 2.5 dias, 1 dia
            },
            {
                'trend_tf': '4H', 
                'confirmation_tf': '1H', 
                'entry_tf': '15min',
                'timeframes': ['4H', '1H', '15min'],
                'limits': {'4H': 30, '1H': 24, '15min': 96}  # 5 dias, 1 dia, 1 dia
            }
        ],
        
        'scalping': [
            {
                'trend_tf': '1H', 
                'confirmation_tf': '30min', 
                'entry_tf': '5min',
                'timeframes': ['1H', '30min', '5min'],
                'limits': {'1H': 24, '30min': 48, '5min': 144}  # 1 dia, 1 dia, 12 horas
            },
            {
                'trend_tf': '30min', 
                'confirmation_tf': '15min', 
                'entry_tf': '3min',
                'timeframes': ['30min', '15min', '3min'],
                'limits': {'30min': 48, '15min': 96, '3min': 160}  # 1 dia, 1 dia, 8 horas
            }
        ],
        
        'ultra_scalping': [
            {
                'trend_tf': '15min', 
                'confirmation_tf': '5min', 
                'entry_tf': '1min',
                'timeframes': ['15min', '5min', '1min'],
                'limits': {'15min': 96, '5min': 144, '1min': 240}  # 1 dia, 12 horas, 4 horas
            },
            {
                'trend_tf': '5min', 
                'confirmation_tf': '3min', 
                'entry_tf': '1min',
                'timeframes': ['5min', '3min', '1min'],
                'limits': {'5min': 144, '3min': 240, '1min': 360}  # 12 horas, 12 horas, 6 horas
            }
        ]
    }
    
    # MAPEAMENTO DE TIMEFRAMES PRINCIPAIS
    TIMEFRAME_TO_STRATEGY = {
        '1W': 'position_trading',
        '1D': 'swing_trading', 
        '6H': 'day_trading',
        '4H': 'day_trading',
        '1H': 'scalping',
        '30min': 'scalping',
        '15min': 'ultra_scalping',
        '5min': 'ultra_scalping'
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

    def get_top_down_data(self, symbol: str, trend_tf: str) -> Dict[str, pd.DataFrame]:
        """
        Obtém dados hierárquicos para análise Top-Down
        
        Args:
            symbol: Par a ser analisado (ex: 'EURUSD')
            trend_tf: Timeframe principal (ex: '1H', '4H', '15min')
        
        Returns:
            Dict com 3 timeframes: {trend_tf: DataFrame, confirmation_tf: DataFrame, entry_tf: DataFrame}
        """
        if trend_tf not in self.TIMEFRAME_TO_STRATEGY:
            raise ValueError(f"Timeframe principal inválido: {trend_tf}. "
                           f"Use: {list(self.TIMEFRAME_TO_STRATEGY.keys())}")
        
        strategy_type = self.TIMEFRAME_TO_STRATEGY[trend_tf]
        config = self._find_config(strategy_type, trend_tf)
        
        if not config:
            raise ValueError(f"Configuração não encontrada para {trend_tf}")
        
        return self._fetch_hierarchical_data(symbol, config, strategy_type)

    def _find_config(self, strategy_type: str, trend_tf: str) -> Optional[Dict]:
        for config in self.TOP_DOWN_STRATEGIES.get(strategy_type, []):
            if config['trend_tf'] == trend_tf:
                return config
        return None

    def _fetch_hierarchical_data(self, symbol: str, config: Dict, strategy_type: str) -> Dict[str, pd.DataFrame]:
        cache_key = f"topdown_{symbol}_{config['trend_tf']}"
        
        # Verificar cache
        if self._is_cache_valid(cache_key, strategy_type):
            return self._cache[cache_key]
        
        # Descrição da estratégia no log
        strategy_desc = self._get_strategy_description(strategy_type)
        print(f"🎯 {strategy_desc} | {symbol} | "
              f"Trend: {config['trend_tf']} | Confirmation: {config['confirmation_tf']} | "
              f"Entry: {config['entry_tf']}")
        
        results = {}
        
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                
                for tf in config['timeframes']:
                    limit = config['limits'].get(tf, 50)
                    future = executor.submit(self._fetch_single_timeframe, symbol, tf, limit, strategy_type)
                    futures[tf] = future
                
                # Coletar resultados
                for tf, future in futures.items():
                    try:
                        df = future.result(timeout=30)
                        results[tf] = df if df is not None else pd.DataFrame()
                        status = "✅" if not df.empty else "⚠️"
                        print(f"   {status} {tf}: {len(df)} candles")
                    except Exception as e:
                        print(f"   ❌ {tf}: {e}")
                        results[tf] = pd.DataFrame()
            
            # Validar e cachear resultados
            valid_timeframes = [tf for tf, df in results.items() if not df.empty]
            if len(valid_timeframes) >= 2:  # Mínimo 2 timeframes para análise
                self._cache[cache_key] = results
                self._cache[f"{cache_key}_timestamp"] = datetime.now()
                print(f"✅ Top-Down carregado: {symbol} {config['trend_tf']} → {valid_timeframes}")
            else:
                print(f"⚠️ Dados insuficientes para {symbol}")
            
            return results
            
        except Exception as e:
            print(f"❌ Erro no Top-Down para {symbol}: {e}")
            return {tf: pd.DataFrame() for tf in config['timeframes']}

    def _fetch_single_timeframe(self, symbol: str, timeframe: str, limit: int, 
                               strategy_type: str) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{timeframe}_{limit}"
        
        if cache_key in self._cache and self._is_cache_valid(cache_key, strategy_type):
            return self._cache[cache_key]
        
        try:
            raw_data = self.data_client.fetch_candles(symbol, interval=timeframe, limit=limit)
            if not raw_data or "history" not in raw_data:
                return None
            
            processed_df = self.processor.process_raw_data_from_client(raw_data, symbol, strategy_type)
            self._cache[cache_key] = processed_df
            return processed_df
            
        except Exception as e:
            print(f"❌ Erro em {symbol} {timeframe}: {e}")
            return None

    def get_available_trend_timeframes(self) -> List[str]:
        return list(self.TIMEFRAME_TO_STRATEGY.keys())

    def get_strategy_info(self, trend_tf: str) -> Dict:
        if trend_tf not in self.TIMEFRAME_TO_STRATEGY:
            return {}
        
        strategy_type = self.TIMEFRAME_TO_STRATEGY[trend_tf]
        config = self._find_config(strategy_type, trend_tf)
        
        if not config:
            return {}
        
        return {
            'strategy_type': strategy_type,
            'strategy_description': self._get_strategy_description(strategy_type),
            'trend_tf': config['trend_tf'],
            'confirmation_tf': config['confirmation_tf'],
            'entry_tf': config['entry_tf'],
            'timeframes': config['timeframes'],
            'limits': config['limits']
        }

    def _get_strategy_description(self, strategy_type: str) -> str:
        descriptions = {
            'position_trading': 'Long-term trends (weeks to months)',
            'swing_trading': 'Medium-term swings (days to weeks)', 
            'day_trading': 'Intraday opportunities (hours to days)',
            'scalping': 'Short-term precision (minutes to hours)',
            'ultra_scalping': 'High-frequency opportunities (seconds to minutes)'
        }
        return descriptions.get(strategy_type, 'Unknown strategy')

    def _is_cache_valid(self, cache_key: str, strategy_type: str) -> bool:
        if cache_key not in self._cache:
            return False
            
        cache_time = self._cache.get(f"{cache_key}_timestamp")
        if not cache_time:
            return False
        
        ttl = self._cache_ttl.get(strategy_type, 300)
        return (datetime.now() - cache_time).total_seconds() < ttl

    def clear_cache(self, symbol: str = None):
        if symbol:
            keys_to_remove = [k for k in self._cache.keys() if symbol in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
        else:
            self._cache.clear()

# Instância global
data_manager = TopDownDataManager()
