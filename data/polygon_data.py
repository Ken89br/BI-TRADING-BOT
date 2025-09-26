import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import requests
from requests.exceptions import RequestException

class PolygonClient:
    def __init__(self):
        self.api_key = os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY não encontrada nas variáveis de ambiente")
        
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TradingBot/2.0",
            "Authorization": f"Bearer {self.api_key}"
        })
        self.logger = logging.getLogger(__name__)
        self.rate_limit_remaining = 5  # Inicializa contador de rate limit

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Monitora e gerencia limites de requisição"""
        self.rate_limit_remaining = int(response.headers.get('x-ratelimit-remaining', 5))
        
        if response.status_code == 429:
            reset_time = int(response.headers.get('x-ratelimit-reset', 60))
            self.logger.warning(f"Rate limit excedido. Reset em {reset_time}s")
            time.sleep(reset_time + 1)
            return True
        return False

    def _normalize_symbol(self, symbol: str) -> Optional[str]:
        """Normaliza símbolos para formato Polygon"""
        if not symbol or not isinstance(symbol, str):
            self.logger.error("Símbolo inválido")
            return None

        symbol = symbol.upper().strip()
        
        # Já formatado (C: ou X:)
        if ":" in symbol:
            return symbol
        
        # Determina tipo pelo padrão
        if len(symbol) == 6 and symbol.isalpha():  # Forex (EURUSD)
            return f"C:{symbol}"
        elif len(symbol) <= 5:  # Stocks (AAPL)
            return f"X:{symbol}"
        
        self.logger.error(f"Formato de símbolo não suportado: {symbol}")
        return None

    def fetch_candles(
        self,
        symbol: str,
        interval: Union[int, str] = "1",
        limit: int = 200,
        retries: int = 3,
        multiplier: int = 1,
        timespan: str = "minute"
    ) -> Optional[Dict[str, Union[List[Dict], float]]]:
        """
        Obtém dados históricos de candles da API Polygon
        
        Args:
            symbol: Símbolo (ex: 'EURUSD' ou 'AAPL')
            interval: Tamanho da vela (1, 5, 15...)
            limit: Número máximo de candles (1-50,000)
            retries: Tentativas em caso de falha
            multiplier: Multiplicador do intervalo (ex: 2 com 'minute' = 2min)
            timespan: Unidade do intervalo (minute, hour, day, etc)
            
        Returns:
            Dicionário com:
            - history: Lista de candles
            - close: Último preço
            - symbol: Símbolo normalizado
            Ou None em caso de erro
        """
        try:
            limit = min(max(1, limit), 50000)
            formatted_symbol = self._normalize_symbol(symbol)
            if not formatted_symbol:
                return None

            # Calcula período de tempo
            end_dt = datetime.utcnow()

            # Corrigido: define dinamicamente os argumentos para timedelta
            delta_args = {}
            if timespan == "minute":
                delta_args["minutes"] = limit * multiplier
            elif timespan == "hour":
                delta_args["hours"] = limit * multiplier
            elif timespan == "day":
                delta_args["days"] = limit * multiplier
            else:
                self.logger.error(f"Timespan inválido: {timespan}")
                return None

            start_dt = end_dt - timedelta(**delta_args)

            endpoint = f"{self.base_url}/v2/aggs/ticker/{formatted_symbol}/range/{multiplier}/{timespan}/{start_dt:%Y-%m-%d}/{end_dt:%Y-%m-%d}"
            
            params = {
                "adjusted": "true",
                "sort": "asc",
                "limit": limit,
            }

            self.logger.info(f"Buscando candles: {formatted_symbol} {multiplier}{timespan[0]} (limit={limit})")

            for attempt in range(1, retries + 1):
                try:
                    if self.rate_limit_remaining <= 1:
                        time.sleep(1)  # Prevenção de rate limit

                    response = self.session.get(
                        endpoint,
                        params=params,
                        timeout=10
                    )

                    if self._handle_rate_limit(response):
                        continue

                    if response.status_code != 200:
                        self.logger.warning(f"Tentativa {attempt}/{retries} - Status {response.status_code}")
                        time.sleep(1.5 ** attempt)  # Backoff exponencial
                        continue

                    data = response.json()

                    if not data.get("results"):
                        self.logger.warning(f"Dados vazios para {formatted_symbol}")
                        return None

                    candles = [{
                        "timestamp": item["t"] // 1000,  # ms para segundos
                        "open": item["o"],
                        "high": item["h"],
                        "low": item["l"],
                        "close": item["c"],
                        "volume": item["v"],
                        "transactions": item.get("n", 0)
                    } for item in data["results"]]

                    return {
                        "history": candles,
                        "close": candles[-1]["close"],
                        "symbol": formatted_symbol,
                        "interval": f"{multiplier}{timespan[0]}"
                    }

                except RequestException as e:
                    self.logger.warning(f"Tentativa {attempt}/{retries} - Erro de rede: {e}")
                    time.sleep(2)
                except KeyError as e:
                    self.logger.error(f"Campo faltando na resposta: {e}")
                    time.sleep(1)
                except Exception as e:
                    self.logger.error(f"Erro inesperado: {e}", exc_info=True)
                    time.sleep(1)

            self.logger.error(f"Falha após {retries} tentativas")
            return None

        except Exception as e:
            self.logger.critical(f"Erro crítico: {e}", exc_info=True)
            return None

    def __del__(self):
        """Garante que a sessão seja fechada"""
        self.session.close()
        
