#Função principal: Automatizar o fluxo completo de coleta de dados, disparo de treinamento, e upload para o Google Drive.
#O que faz:
#Busca dados de candles para vários símbolos/timeframes utilizando um cliente externo (dukascopy_client.cjs).
#Salva esses dados em CSV.
#Faz upload dos CSVs e modelos para o Google Drive.
#Periodicamente dispara o treinamento do modelo histórico (train_model_historic.main()).
#Roda em loop continuamente, mantendo os dados e modelos sempre atualizados.

#strategy/autotrainer.py
import os
import time
import json
import glob
from dotenv import load_dotenv
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from hashlib import md5

from strategy.train_model_historic import main as run_training
from config import CONFIG
from data.google_drive_client import upload_or_update_file as upload_file, get_folder_id_for_file
from data.data_client import FallbackDataClient

load_dotenv()

SYMBOLS = CONFIG["symbols"] + CONFIG.get("otc_symbols", [])
TIMEFRAMES = CONFIG["timeframes"]
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

BOOTSTRAP_FLAG = os.path.join(DATA_DIR, "autotrainer_bootstrap.flag")
LAST_RETRAIN_PATH = os.path.join(DATA_DIR, "last_retrain.txt")
LOG_FILE = os.path.join(DATA_DIR, "autotrainer.log")
FILE_HASHES_PATH = os.path.join(DATA_DIR, "autotrainer_uploaded_hashes.json")

NORMAL_LIMIT = 1000  # Limite de candles no fluxo normal (ciclos). Bootstrap pega 7 dias!

def setup_logging():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def file_md5(path):
    hash_md5 = md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Erro ao calcular hash de {path}: {str(e)}")
        return None

def load_uploaded_hashes():
    if os.path.exists(FILE_HASHES_PATH):
        try:
            with open(FILE_HASHES_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar hashes: {str(e)}")
    return {}

def save_uploaded_hashes(hashes):
    try:
        with open(FILE_HASHES_PATH, "w") as f:
            json.dump(hashes, f)
    except Exception as e:
        logger.error(f"Erro ao salvar hashes: {str(e)}")

uploaded_hashes = load_uploaded_hashes()

data_client = FallbackDataClient()
MIN_CANDLES = 50

def merge_and_save_csv(filepath, new_candles):
    import pandas as pd
    # Mantém histórico anterior e adiciona apenas candles novos, sem duplicar timestamp
    if os.path.exists(filepath):
        try:
            df_old = pd.read_csv(filepath)
        except Exception:
            df_old = pd.DataFrame()
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

def get_bootstrap_limit(tf):
    tf_map = {
        "s1": 1/60, "m1": 1, "m5": 5, "m15": 15, "m30": 30,
        "h1": 60, "h4": 240, "d1": 1440
    }
    tf_minutes = tf_map.get(tf.lower(), 1)
    return int(7 * 24 * 60 / tf_minutes)  # 7 dias

# Adicionado parâmetro 'limit' explicitamente. Se None, decide de acordo com prefer_pocket/bootstrap.
def fetch_and_save(symbol: str, from_dt: datetime, to_dt: datetime, tf: str, prefer_pocket=False, limit=None) -> bool:
    try:
        tf_map = {
            "S1": "s1", "M1": "m1", "M5": "m5", "M15": "m15",
            "M30": "m30", "H1": "h1", "H4": "h4", "D1": "d1"
        }
        interval = tf_map.get(tf.upper(), tf.lower())
        # Se estamos no bootstrap (prefer_pocket=true), usa get_bootstrap_limit.
        # Caso contrário, usa NORMAL_LIMIT.
        if limit is None:
            limit = get_bootstrap_limit(interval) if prefer_pocket else NORMAL_LIMIT
        candles_result = data_client.fetch_candles(symbol, interval=interval, limit=limit, prefer_pocket=prefer_pocket)
        candles = candles_result["history"] if candles_result and "history" in candles_result else None
        if not candles or len(candles) < MIN_CANDLES:
            logger.warning(f"Não foi possível obter candles válidos para {symbol} @ {tf} (obtidos: {0 if not candles else len(candles)})")
            return False

        symbol_clean = symbol.lower().replace(" ", "").replace("/", "")
        filename = f"{symbol_clean}_{interval}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        merge_and_save_csv(filepath, candles)
        logger.info(f"Saved/merged {len(candles)} rows to {filename}")
        return True

    except Exception as e:
        logger.error(f"Unexpected error with {symbol} @ {tf}: {str(e)}", exc_info=True)
    return False

# Adicionado parâmetro 'limit' explicitamente e propagado para fetch_and_save.
def fetch_all_symbols_timeframes(from_dt: datetime, to_dt: datetime, max_workers: int = 6, prefer_pocket=False, limit=None):
    jobs = []
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for symbol in SYMBOLS:
            for tf in TIMEFRAMES:
                job = executor.submit(fetch_and_save, symbol, from_dt, to_dt, tf, prefer_pocket, limit)
                jobs.append(((symbol, tf), job))
        for (symbol, tf), job in jobs:
            try:
                results[(symbol, tf)] = job.result()
            except Exception as e:
                logger.error(f"Erro no fetch {symbol} {tf}: {str(e)}")
    return results

def bootstrap_initial_data():
    if os.path.exists(BOOTSTRAP_FLAG):
        logger.info("Bootstrap já realizado anteriormente. Pulando bootstrap inicial.")
        return False  # não fez bootstrap agora
    logger.info("Bootstrapping initial data (last 7 days, prefer PocketOption)")
    now = datetime.utcnow()
    from_dt = now - timedelta(days=7)
    fetch_all_symbols_timeframes(from_dt, now, prefer_pocket=True, limit=None)  # Bootstrap: busca máximo
    with open(BOOTSTRAP_FLAG, "w") as f:
        f.write(now.isoformat())
    logger.info("Bootstrap complete.")
    return True  # fez bootstrap agora

def should_retrain() -> bool:
    if not os.path.exists(LAST_RETRAIN_PATH):
        return True
    try:
        with open(LAST_RETRAIN_PATH, "r") as f:
            last = datetime.fromisoformat(f.read().strip())
        return (datetime.utcnow() - last).total_seconds() >= 120
    except Exception as e:
        logger.error(f"Error reading last retrain time: {str(e)}")
        return True

def store_last_retrain_time():
    now = datetime.utcnow().isoformat()
    with open(LAST_RETRAIN_PATH, "w") as f:
        f.write(now)
    try:
        upload_file(LAST_RETRAIN_PATH)
        logger.info("Updated last retrain time on Drive")
    except Exception as e:
        logger.error(f"Failed to upload retrain time: {str(e)}")

def upload_files_parallel(pattern: str, description: str, max_workers: int = 4):
    files = glob.glob(pattern)
    jobs = []
    uploaded = 0
    changed_hashes = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for filepath in files:
            filename = os.path.basename(filepath)
            file_hash = file_md5(filepath)
            if not file_hash:
                continue
            if uploaded_hashes.get(filename) == file_hash:
                logger.info(f"Skip upload (not changed): {filename}")
                continue
            jobs.append(executor.submit(upload_file, filepath))
            uploaded_hashes[filename] = file_hash
            changed_hashes = True
        for idx, job in enumerate(jobs):
            try:
                file_id = job.result()
                logger.info(f"Uploaded {description}: {files[idx]} (ID: {file_id})")
                uploaded += 1
            except Exception as e:
                logger.error(f"Failed to upload {files[idx]}: {str(e)}")
    if changed_hashes:
        save_uploaded_hashes(uploaded_hashes)
    return uploaded

def main_loop():
    did_bootstrap = bootstrap_initial_data()
    if did_bootstrap:
        # Primeiro treinamento completo logo após bootstrap (usando todos os CSVs)
        try:
            logger.info("Primeiro treinamento com todo o histórico baixado (7 dias)")
            run_training()
            # Upload dos modelos logo após o primeiro treinamento
            uploaded_models = upload_files_parallel(f"{MODEL_DIR}/model_*.pkl", "model files")
            logger.info(f"Uploaded {uploaded_models} model files (bootstrap).")
            store_last_retrain_time()
        except Exception as e:
            logger.error(f"Erro no treinamento inicial após bootstrap: {str(e)}", exc_info=True)

    while True:
        cycle_start = datetime.utcnow()
        logger.info(f"Starting new cycle at {cycle_start.isoformat()}")

        fetch_start = cycle_start - timedelta(seconds=30)
        # Chamada do ciclo normal agora usa NORMAL_LIMIT
        fetch_results = fetch_all_symbols_timeframes(fetch_start, cycle_start, limit=NORMAL_LIMIT)
        success_count = sum(1 for v in fetch_results.values() if v)
        logger.info(f"Fetch complete: {success_count} datasets updated.")

        uploaded_csv = upload_files_parallel(f"{DATA_DIR}/*.csv", "CSV files")
        logger.info(f"Uploaded {uploaded_csv} CSV files.")

        if should_retrain():
            try:
                logger.info("Starting model training...")
                run_training()
                store_last_retrain_time()
                uploaded_models = upload_files_parallel(f"{MODEL_DIR}/model_*.pkl", "model files")
                logger.info(f"Uploaded {uploaded_models} model files.")
            except Exception as e:
                logger.error(f"Training failed: {str(e)}", exc_info=True)
        else:
            logger.info("Skipping training - not time yet")

        cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
        sleep_time = max(1, 30 - cycle_duration)
        logger.info(f"Cycle completed in {cycle_duration:.2f}s. Sleeping {sleep_time:.2f}s")
        time.sleep(sleep_time)

def main():
    try:
        logger.info("Starting AutoTrainer service")
        main_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
