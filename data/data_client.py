#data/data_client.py
import subprocess
import json
import os
import pandas as pd
import joblib
from datetime import datetime, timedelta
from data.iqoption_data import IQOptionClient
from data.pocketoption_data import PocketOptionClient, PocketOptionAuthError
from data.twelvedata_data import TwelveDataClient
from data.tiingo_data import TiingoClient
from data.polygon_data import PolygonClient
from strategy.train_model_historic import main as run_training
from data.google_drive_client import upload_or_update_file as upload_file, download_file, find_file_id, get_folder_id_for_file

# Adicione para alerta Telegram
from utils.telegram_alert import send_telegram_alert

LAST_RETRAIN_PATH = "last_retrain.txt"

def merge_and_save_csv(filepath, new_candles):
    if os.path.exists(filepath):
        df_old = pd.read_csv(filepath)
    else:
        df_old = pd.DataFrame()
    df_new = pd.DataFrame(new_candles)
    if df_new.empty:
        return
    if not df_old.empty:
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    df_all.sort_values('timestamp', inplace=True)
    df_all.to_csv(filepath, index=False)

def _map_symbol(symbol, provider):
    """
    Adapta o símbolo para o formato correto de cada provedor.
    """
    orig = symbol
    if provider == "PocketOptionClient":
        # Ex: EURUSD → eurusd, EURUSD OTC → eurusdotc (sem espaço/caixa baixa)
        s = symbol.lower().replace(" ", "")
        return s
    elif provider == "TwelveDataClient":
        # Ex: EURUSD → EUR/USD, EURUSD OTC → ignora OTC
        if "OTC" in symbol:
            # TwelveData não suporta OTC, devolve None
            return None
        if "/" in symbol:
            return symbol.upper().replace(" OTC", "")
        return f"{symbol[:3]}/{symbol[3:6]}"
    elif provider == "TiingoClient":
        # Ex: EURUSD → eurusd (sem OTC)
        if "OTC" in symbol:
            return None
        return symbol.lower().replace(" ", "").replace("/", "")
    elif provider == "PolygonClient":
        # Ex: EURUSD → C:EURUSD, EURUSD OTC → ignora OTC
        if "OTC" in symbol:
            return None
        base = symbol.replace(" ", "").replace("/", "")
        if len(base) == 6:
            return f"C:{base.upper()}"
        return None
    elif provider == "Dukascopy":
        # Dukascopy aceita OTC (em algumas plataformas) como eurusdotc ou eurusd (deve testar)
        return symbol.lower().replace(" ", "")
    # Default: retorna como está
    return symbol

def _map_timeframe(interval, provider):
    """
    Adapta o timeframe/intervalo para cada provedor.
    """
    tf_map = {
        "S1": "s1", "M1": "m1", "M5": "m5", "M15": "m15",
        "M30": "m30", "H1": "h1", "H4": "h4", "D1": "d1", "W1": "1w",
        "1min": "m1", "5min": "m5", "15min": "m15", "30min": "m30",
        "1h": "h1", "2h": "h2", "4h": "h4", "6h": "h6", "1d": "d1", "1w": "w1",
    }
    if provider == "TwelveDataClient":
        # TwelveData aceita "1min", "5min", etc
        return {
            "s1": "1min", "m1": "1min", "m5": "5min", "m15": "15min",
            "m30": "30min", "h1": "1h", "h2": "2h", "h4": "4h", "h6": "6h", "d1": "1d", "w1": "1w",
        }.get(tf_map.get(interval, interval), "1min")
    elif provider == "TiingoClient":
        # Aceita "1min", "5min", etc
        return {
            "s1": "1min", "m1": "1min", "m5": "5min", "m15": "15min",
            "m30": "30min", "h1": "1h", "h2": "2h", "h4": "4h", "h6": "6h", 
        }.get(tf_map.get(interval, interval), "1min")
    elif provider == "PolygonClient":
        # Só aceita "1", "5", "15", "30", "60"
        return {
            "s1": "1", "m1": "1", "m5": "5", "m15": "15", "m30": "30",
            "h1": "60", "h4": "240"
        }.get(tf_map.get(interval, interval), "1")
    elif provider == "PocketOptionClient":
        # Aceita "m1", "m5", etc (já no formato)
        return tf_map.get(interval, interval)
    elif provider == "Dukascopy":
        return tf_map.get(interval, interval)
    return interval

class FallbackDataClient:
    IN_ROWS_BEFORE_RETRAIN = 50
    def __init__(self):
        self.providers = [
            PocketOptionClient(),
            IQOptionClient(),
            TwelveDataClient(),
            TiingoClient(),
            PolygonClient()
        ]

    def fetch_candles(self, symbol, interval="1min", limit=5, prefer_pocket=False):
        tested = set()
        # 1. Sempre tente Dukascopy primeiro (até mesmo no prefer_pocket), se possível
        dsymbol = _map_symbol(symbol, "Dukascopy")
        dinterval = _map_timeframe(interval, "Dukascopy")
        print(f"📡 [Dukascopy] Tentando symbol={dsymbol}, interval={dinterval}, limit={limit}")
        try:
            candles = self._fetch_from_dukascopy(dsymbol, dinterval, limit)
            if candles and "history" in candles and candles["history"]:
                print("✅ Dukascopy succeeded.")
                self._save_to_csv(symbol, interval, candles["history"])
                self._maybe_retrain()
                return candles
        except Exception as e:
            print(f"❌ Dukascopy failed: {e}")

        # 2. Depois, tente os providers alternativos, respeitando OTC
        for i, provider in enumerate(self.providers):
            name = provider.__class__.__name__
            psymbol = _map_symbol(symbol, name)
            if psymbol is None:
                print(f"⚠️ [{name}] Não suporta esse símbolo: {symbol}")
                continue
            pinterval = _map_timeframe(interval, name)
            if (name, psymbol, pinterval) in tested:
                continue
            print(f"⚙️ Tentando {name} com symbol={psymbol}, interval={pinterval}, limit={limit}")
            try:
                result = provider.fetch_candles(psymbol, interval=pinterval, limit=limit)
                if result and "history" in result and result["history"]:
                    print(f"✅ Êxito de {name}")
                    self._save_to_csv(symbol, interval, result["history"])
                    return result
            except PocketOptionAuthError as e:
                msg = f"❗ <b>ERRO PocketOption SSID</b>\n{e}\nHora: {datetime.utcnow()}"
                print(msg)
                send_telegram_alert(msg)
            except Exception as e:
                print(f"❌ {name} error: {e}")
            tested.add((name, psymbol, pinterval))

        print("❌ Todos provedores falharam.")
        return None

    def _fetch_from_dukascopy(self, symbol, interval, limit):
        now = datetime.utcnow()
        filename = f"{symbol}_{interval}.csv"
        filepath = f"data/{filename}"
        if not os.path.exists(filepath):
            try:
                print(f"⬇️ Baixando {filename} do Google Drive...")
                download_file(filename, filepath, drive_folder_id=get_folder_id_for_file(filename))
                print(f"✅ Baixado {filename} do Google Drive.")
            except Exception as e:
                print(f"⚠️ Não foi possível baixar {filename}: {e}")
        from_dt = now - timedelta(minutes=limit)
        cmd = [
            "node", "--max-old-space-size=1024", "data/dukascopy_client.cjs",
            symbol, interval,
            from_dt.isoformat(), now.isoformat()
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        candles = json.loads(result.stdout)
        return {
            "history": candles,
            "close": candles[-1]["close"] if candles else None
        }

    def _save_to_csv(self, symbol, interval, candles):
        if not candles:
            return
        filename = f"{symbol.lower().replace(' ', '').replace('/', '')}_{_map_timeframe(interval, 'Dukascopy')}.csv"
        path = f"data/{filename}"
        merge_and_save_csv(path, candles)
        try:
            file_id = upload_file(path)
            print(f"☁️ Arquivo {filename} enviado ao Google Drive! ID: {file_id}")
        except Exception as e:
            print(f"⚠️ Falha ao enviar {filename} ao Google Drive: {e}")

    def _maybe_retrain(self):
        from strategy.train_model_historic import main as run_training
        now = datetime.utcnow()
        last = self._load_last_retrain_time()
        if not last or (now - last).total_seconds() >= 30:
            print("🧠 Triggering model retraining...")
            run_training()
            self._store_last_retrain_time(now)

    def _load_last_retrain_time(self):
        try:
            if not os.path.exists(LAST_RETRAIN_PATH):
                download_file(LAST_RETRAIN_PATH, LAST_RETRAIN_PATH, drive_folder_id=get_folder_id_for_file(LAST_RETRAIN_PATH))
        except Exception as e:
            pass
        if not os.path.exists(LAST_RETRAIN_PATH):
            return None
        try:
            with open(LAST_RETRAIN_PATH, "r") as f:
                ts = f.read().strip()
                return datetime.fromisoformat(ts)
        except:
            return None

    def _store_last_retrain_time(self, dt):
        with open(LAST_RETRAIN_PATH, "w") as f:
            f.write(dt.isoformat())
        try:
            upload_file(LAST_RETRAIN_PATH)
        except Exception as e:
            print(f"⚠️ Falha ao enviar {LAST_RETRAIN_PATH} ao Google Drive: {e}")
