import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from iqoptionapi.stable_api import IQ_Option

class IQOptionClient:
    def __init__(self):
        self.email = "harryloype@gmail.com"
        self.password = "Kenet-Abreu@3646633"
        self.iq = None
        self.timeout = 15  # timeout em segundos
        self.max_retries = 3
        self._init_timeframe_mapping()
        self._connect()

    def _init_timeframe_mapping(self):
        """Mapeamento completo de timeframes incluindo OTC"""
        self.timeframe_map = {
            # Standard timeframes
            "S1": 1, "M1": 60, "M5": 300, "M15": 900,
            "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400,
            # Aliases alternativos
            "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
            "1h": 3600, "4h": 14400, "1d": 86400
        }

    def _connect(self):
        """Conexão robusta com tratamento de erro aprimorado"""
        self.iq = IQ_Option(self.email, self.password)
        
        if not self.iq.connect():
            raise ConnectionError("Falha na conexão inicial com IQ Option")
        
        if not self.iq.check_connect():
            raise ConnectionError("Falha na autenticação com IQ Option")

    def _ensure_connection(self):
        """Verifica e reconecta se necessário"""
        try:
            if not self.iq.check_connect():
                print("Reconectando à IQ Option...")
                self._connect()
                time.sleep(1)  # Pausa pós-reconexão
        except Exception as e:
            print(f"Erro na verificação de conexão: {str(e)}")
            self._connect()

    def _normalize_symbol(self, symbol: str) -> str:
        """Padroniza símbolos incluindo tratamento para OTC"""
        return (
            symbol.upper()
            .replace(" ", "")
            .replace("/", "")
            .replace("OTC", "OT")  # IQ Option usa 'OT' para mercados OTC
        )

    def fetch_candles(
        self, 
        symbol: str, 
        interval: str = "M1", 
        limit: int = 100,
        retries: int = None
    ) -> Optional[Dict[str, List[Dict]]]:
        retries = retries or self.max_retries
        symbol_clean = self._normalize_symbol(symbol)
        tf_seconds = self.timeframe_map.get(interval.upper(), 60)
        
        for attempt in range(1, retries + 1):
            try:
                self._ensure_connection()
                
                now = int(time.time())
                start_time = now - (tf_seconds * limit)
                
                candles = self.iq.get_candles(symbol_clean, tf_seconds, start_time, now)
                
                if not candles:
                    if attempt == retries:
                        return None
                    time.sleep(1)
                    continue
                
                history = []
                for candle in candles:
                    history.append({
                        "timestamp": candle["from"],
                        "datetime": datetime.fromtimestamp(candle["from"]).strftime('%Y-%m-%d %H:%M:%S'),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle.get("volume", 0))
                    })
                
                return {
                    "history": history,
                    "close": history[-1]["close"],
                    "provider": "IQOption",
                    "symbol": symbol,
                    "timeframe": interval
                }
                
            except Exception as e:
                print(f"⚠️ Tentativa {attempt}/{retries} falhou para {symbol} {interval}: {str(e)}")
                if attempt == retries:
                    return None
                time.sleep(1.5 ** attempt)  # Backoff exponencial
        
        return None

    def __del__(self):
        """Destruidor com tratamento de erro robusto"""
        if hasattr(self, 'iq') and self.iq:
            try:
                self.iq.close_connection()
            except Exception as e:
                print(f"Erro ao fechar conexão: {str(e)}")
