import os
import json
import time
import socket
from datetime import datetime
import websocket
from websocket import WebSocketTimeoutException, WebSocketConnectionClosedException
from typing import Optional, Dict, List

class PocketOptionAuthError(Exception):
    """Erro de autenticação/SSID inválido ou expirado."""
    pass

class PocketOptionNetworkError(Exception):
    """Erro de conexão com o servidor PocketOption."""
    pass

class PocketOptionClient:
    def __init__(self):
        # URL alternativa incluída como fallback
        self.ws_urls = [
            "wss://socket.pocketoption.com/socket.io/?EIO=4&transport=websocket",
            "wss://eu.socket.pocketoption.com/socket.io/?EIO=4&transport=websocket",
            "wss://us.socket.pocketoption.com/socket.io/?EIO=4&transport=websocket"
        ]
        self.ssid = os.getenv("POCKETOPTION_SSID")
        if not self.ssid:
            raise RuntimeError("POCKETOPTION_SSID não encontrado no .env!")
        self.timeout = 10  # timeout em segundos
        self.current_ws_url = None

    def _to_tf(self, interval: str) -> int:
        mapping = {
            "s30": 30,
            "m1": 60,
            "m2": 120,
            "m5": 300,
            "m15": 900,
            "m30": 1800,
            "h1": 3600,
            "h4": 14400,
            "d1": 86400
        }
        return mapping.get(interval.lower(), 60)

    def _create_connection(self) -> websocket.WebSocket:
        """Tenta criar conexão com fallback para URLs alternativas"""
        last_exception = None
        
        for url in self.ws_urls:
            try:
                self.current_ws_url = url
                ws = websocket.create_connection(
                    url,
                    timeout=self.timeout,
                    socket=socket.SOL_TCP  # Força uso do protocolo TCP padrão
                )
                return ws
            except (socket.gaierror, WebSocketTimeoutException, 
                   WebSocketConnectionClosedException) as e:
                last_exception = e
                continue
                
        raise PocketOptionNetworkError(
            f"Não foi possível conectar a nenhum servidor PocketOption. Último erro: {last_exception}"
        )

    def fetch_candles(self, symbol: str, interval: str = "m1", limit: int = 5, 
                     retries: int = 2) -> Optional[Dict[str, List[Dict]]]:
        symbol_api = symbol.lower().replace(" ", "").replace("/", "")
        tf_sec = self._to_tf(interval)
        
        for attempt in range(retries + 1):
            try:
                candles = self._fetch_ws_candles(symbol_api, tf_sec, limit)
                if candles:
                    return {"history": candles, "close": candles[-1]["close"]}
            except PocketOptionAuthError as e:
                raise  # Erros de auth não devem ser retried
            except (PocketOptionNetworkError, Exception) as e:
                if attempt == retries:
                    print(f"❌ PocketOption WS error (tentativa {attempt + 1}/{retries + 1}): {e}")
                time.sleep(1)  # Backoff simples
                
        return None

    def _fetch_ws_candles(self, asset: str, period: int, limit: int) -> Optional[List[Dict]]:
        """Lógica principal de obtenção de candles via WS"""
        ws = None
        try:
            ws = self._create_connection()
            
            # Handshake inicial
            init_msg = ws.recv()
            if not init_msg.startswith('0'):
                raise PocketOptionNetworkError("Resposta inicial inesperada do servidor")
            
            # Autenticação
            auth_payload = json.dumps({
                "session": self.ssid,
                "isDemo": 0,
                "platform": 2,
                "isFastHistory": True,
                "isOptimized": True
            })
            ws.send(f'42["auth",{auth_payload}]')
            
            # Verificação de autenticação
            auth_ok = False
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    msg = ws.recv()
                    if '"auth"' in msg:
                        if '"success":true' in msg or "authenticated" in msg:
                            auth_ok = True
                            break
                        elif '"success":false' in msg or "error" in msg:
                            raise PocketOptionAuthError("Autenticação falhou: SSID inválido ou expirado")
                except WebSocketTimeoutException:
                    continue
                    
            if not auth_ok:
                raise PocketOptionNetworkError("Timeout durante autenticação")
            
            # Solicitação de candles
            now = int(time.time())
            req_payload = json.dumps({
                "asset": asset,
                "period": period,
                "limit": limit,
                "end": now
            })
            ws.send(f'42["get-candles",{req_payload}]')
            
            # Processamento da resposta
            candles = []
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    msg = ws.recv()
                    if '"get-candles"' in msg:
                        try:
                            arr = json.loads(msg[2:])
                            data = arr[1].get("data", [])
                            candles = [
                                {
                                    "timestamp": int(c["time"]),
                                    "open": float(c["open"]),
                                    "high": float(c["high"]),
                                    "low": float(c["low"]),
                                    "close": float(c["close"]),
                                    "volume": float(c.get("volume", 0))
                                }
                                for c in data
                            ]
                            return candles
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            raise PocketOptionNetworkError(f"Erro ao parsear candles: {e}")
                except WebSocketTimeoutException:
                    continue
                    
            raise PocketOptionNetworkError("Timeout ao aguardar candles")
            
        finally:
            if ws:
                try:
                    ws.close()
                except:
                    pass

    def __del__(self):
        """Garante que a conexão seja fechada"""
        pass  # A conexão já é fechada no finally block
