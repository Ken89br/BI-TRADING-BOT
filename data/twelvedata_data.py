import requests
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import logging
from requests.exceptions import RequestException

class TwelveDataClient:
    def __init__(self):
        self.api_key = os.getenv("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise ValueError("TWELVEDATA_API_KEY não encontrada nas variáveis de ambiente")
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TradingBot"})
        self.logger = logging.getLogger(__name__)

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Verifica e trata limites de requisição"""
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            self.logger.warning(f"Rate limit atingido. Retry após {retry_after}s")
            time.sleep(retry_after)
            return True
        return False

    def _parse_datetime(self, dt_str: str) -> Optional[int]:
        """Converte múltiplos formatos de data para timestamp"""
        formats = [
            "%Y-%m-%d %H:%M:%S",  # Com hora
            "%Y-%m-%d",           # Apenas data
            "%Y-%m-%dT%H:%M:%S",  # ISO format
            "%Y-%m-%d %H:%M:%S%z" # Com timezone
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                return int(dt.timestamp())
            except ValueError:
                continue
                
        self.logger.error(f"Formato de data não reconhecido: {dt_str}")
        return None

    def _validate_response(self, data: Dict) -> bool:
        """Valida a estrutura da resposta da API"""
        if not isinstance(data, dict):
            self.logger.error("Resposta da API não é JSON válido")
            return False
            
        if "code" in data and data["code"] != 200:
            self.logger.error(f"Erro na API: {data.get('message', 'Sem mensagem')}")
            return False
            
        if "values" not in data:
            self.logger.error("Resposta não contém dados de candles")
            return False
            
        return True

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1min",
        limit: int = 200,
        retries: int = 3,
        delay: float = 1.5
    ) -> Optional[Dict[str, List[Dict]]]:
        """
        Obtém dados históricos de candles
        
        Args:
            symbol: Par de moedas (ex: 'EUR/USD' ou 'EURUSD')
            interval: Intervalo (1min, 5min, 1h, etc)
            limit: Número máximo de candles (máx 5000)
            retries: Tentativas em caso de falha
            delay: Atraso entre tentativas (segundos)
            
        Returns:
            Dict com 'history' (lista de candles) e 'close' (último preço)
            ou None em caso de erro
        """
        limit = min(limit, 5000)
        formatted_symbol = symbol.replace("/", "") if "/" in symbol else symbol
        endpoint = f"{self.base_url}/time_series"
        
        params = {
            "symbol": formatted_symbol,
            "interval": interval,
            "outputsize": limit,
            "apikey": self.api_key
        }

        self.logger.info(f"Buscando candles: {formatted_symbol} {interval} (limit={limit})")

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(endpoint, params=params, timeout=15)
                
                if self._handle_rate_limit(response):
                    continue
                    
                if response.status_code != 200:
                    self.logger.warning(f"Tentativa {attempt}/{retries} - Status {response.status_code}")
                    time.sleep(delay)
                    continue
                    
                data = response.json()
                
                if not self._validate_response(data):
                    time.sleep(delay)
                    continue
                    
                candles = []
                for row in reversed(data["values"]):
                    ts = self._parse_datetime(row["datetime"])
                    if ts is None:
                        continue
                        
                    try:
                        candle = {
                            "timestamp": ts,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row.get("volume", 0))
                        }
                        candles.append(candle)
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Erro ao processar candle: {e}")
                        continue
                        
                if not candles:
                    self.logger.error("Nenhum candle válido encontrado")
                    return None
                    
                return {
                    "history": candles,
                    "close": candles[-1]["close"],
                    "symbol": formatted_symbol,
                    "interval": interval
                }
                
            except RequestException as e:
                self.logger.warning(f"Tentativa {attempt}/{retries} - Erro de rede: {str(e)}")
                time.sleep(delay)
            except Exception as e:
                self.logger.error(f"Erro inesperado: {str(e)}", exc_info=True)
                time.sleep(delay)

        self.logger.error(f"Falha após {retries} tentativas")
        return None

    def __del__(self):
        self.session.close()
