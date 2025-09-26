import os
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from requests.exceptions import RequestException

class TiingoClient:
    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        if not self.api_key:
            raise ValueError("TIINGO_API_KEY não encontrada nas variáveis de ambiente")
        
        self.base_url = "https://api.tiingo.com/tiingo/fx"
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "TradingBot"
        })
        self.logger = logging.getLogger(__name__)
        self.rate_limit_remaining = 500  # Valor inicial padrão

    def _validate_symbol(self, symbol: str) -> Optional[str]:
        """Valida e formata o símbolo para a API Tiingo"""
        if not symbol or len(symbol.replace("/", "")) != 6:
            self.logger.error(f"Formato de símbolo inválido: {symbol}")
            return None
        
        return symbol.lower().replace("/", "")

    def _calculate_date_range(self, interval: str, limit: int) -> Dict[str, str]:
        """Calcula o range de datas baseado no intervalo e limite"""
        interval_map = {
            '1min': timedelta(minutes=1),
            '5min': timedelta(minutes=5),
            '15min': timedelta(minutes=15),
            '30min': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        
        delta = interval_map.get(interval, timedelta(minutes=1))
        total_time = delta * limit
        end_date = datetime.utcnow()
        start_date = end_date - total_time
        
        return {
            "start": start_date.isoformat(timespec='seconds') + 'Z',
            "end": end_date.isoformat(timespec='seconds') + 'Z'
        }

    def _handle_rate_limit(self, headers: Dict) -> None:
        """Atualiza e monitora o rate limit"""
        self.rate_limit_remaining = int(headers.get('X-RateLimit-Remaining', 500))
        
        if self.rate_limit_remaining < 50:
            self.logger.warning(f"Rate limit baixo: {self.rate_limit_remaining} requisições restantes")

    def _parse_candle_data(self, data: List[Dict], limit: int) -> Optional[Dict]:
        """Processa e valida os dados dos candles"""
        if not data or not isinstance(data, list):
            self.logger.error("Dados de candles inválidos ou vazios")
            return None
        
        candles = []
        valid_items = 0
        
        for item in data[-limit:]:
            try:
                candle = {
                    "timestamp": int(datetime.fromisoformat(item["date"].replace('Z', '')).timestamp()),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item.get("volume", 0)) or 0
                }
                candles.append(candle)
                valid_items += 1
            except (KeyError, ValueError) as e:
                self.logger.warning(f"Erro ao processar candle: {e}")
                continue
        
        if valid_items == 0:
            self.logger.error("Nenhum candle válido encontrado")
            return None
            
        return {
            "history": candles,
            "close": candles[-1]["close"],
            "count": valid_items
        }

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1min",
        limit: int = 200,
        retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, List[Dict]]]:
        """
        Obtém dados históricos de candles do Tiingo
        
        Args:
            symbol: Par de moedas (ex: 'EURUSD' ou 'EUR/USD')
            interval: Intervalo (1min, 5min, 15min, 30min, 1h, 4h, 1d)
            limit: Número máximo de candles (1-5000)
            retries: Número de tentativas em caso de falha
            retry_delay: Atraso entre tentativas (segundos)
            
        Returns:
            Dicionário com:
            - history: Lista de candles
            - close: Último preço
            - count: Número de candles válidos
            ou None em caso de erro
        """
        # Validação inicial
        formatted_symbol = self._validate_symbol(symbol)
        if not formatted_symbol:
            return None
            
        limit = max(1, min(limit, 5000))
        date_range = self._calculate_date_range(interval, limit)
        
        params = {
            "tickers": formatted_symbol,
            "startDate": date_range["start"],
            "endDate": date_range["end"],
            "resampleFreq": interval,
            "format": "json",
            "token": self.api_key
        }

        self.logger.info(f"Buscando candles: {formatted_symbol} {interval} (limit={limit})")

        for attempt in range(1, retries + 1):
            try:
                # Verificação de rate limit
                if self.rate_limit_remaining < 10:
                    wait_time = 60
                    self.logger.warning(f"Aguardando {wait_time}s por rate limit")
                    time.sleep(wait_time)
                
                response = self.session.get(
                    f"{self.base_url}/prices",
                    params=params,
                    timeout=10
                )
                
                self._handle_rate_limit(response.headers)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    self.logger.warning(f"Rate limit excedido. Retry após {retry_after}s")
                    time.sleep(retry_after)
                    continue
                    
                if response.status_code != 200:
                    self.logger.warning(f"Tentativa {attempt}/{retries} - Status {response.status_code}")
                    time.sleep(retry_delay)
                    continue
                
                data = response.json()
                result = self._parse_candle_data(data, limit)
                
                if result:
                    self.logger.info(f"Sucesso: {result['count']} candles recebidos")
                    return result
                    
                time.sleep(retry_delay)
                
            except RequestException as e:
                self.logger.warning(f"Tentativa {attempt}/{retries} - Erro de rede: {str(e)}")
                time.sleep(retry_delay)
            except Exception as e:
                self.logger.error(f"Erro inesperado: {str(e)}", exc_info=True)
                time.sleep(retry_delay)

        self.logger.error(f"Falha após {retries} tentativas")
        return None

    def __del__(self):
        self.session.close()
