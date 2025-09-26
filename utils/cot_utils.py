import os
import pandas as pd
from data.google_drive_client import download_file, list_files_in_drive_folder

COT_DATA_DIR = "cot_data"
GDRIVE_FOLDER_ID_CSV = "1Bv5rwzYMUVuRNSXKSz9zAFidDTCjY8g6"

def get_latest_cot_drive():
    """Busca o cot_processed_*.csv mais recente do Google Drive e faz download para cot_data/."""
    files = list_files_in_drive_folder(GDRIVE_FOLDER_ID_CSV)
    cot_files = [f for f in files if f['name'].startswith('cot_processed_') and f['name'].endswith('.csv')]
    if not cot_files:
        print("[COT UTILS] Nenhum arquivo cot_processed encontrado no Drive.")
        return None
    # Ordena pelo modifiedTime (YYYY-MM-DDTHH:MM:SS.sssZ ou similar)
    latest = max(cot_files, key=lambda f: f['modifiedTime'])
    local_path = os.path.join(COT_DATA_DIR, latest['name'])
    if not os.path.exists(COT_DATA_DIR):
        os.makedirs(COT_DATA_DIR, exist_ok=True)
    download_file(latest['name'], local_path, drive_folder_id=GDRIVE_FOLDER_ID_CSV)
    return local_path

def get_latest_cot(symbol):
    """
    Busca o valor COT mais recente para o símbolo informado.
    Tenta localmente, senão faz download do Drive.
    """
    if not os.path.exists(COT_DATA_DIR):
        os.makedirs(COT_DATA_DIR, exist_ok=True)
    # Busca localmente o mais recente
    files = [f for f in os.listdir(COT_DATA_DIR) if f.startswith("cot_processed_") and f.endswith(".csv")]
    if not files:
        csv_path = get_latest_cot_drive()
    else:
        latest_file = sorted(files)[-1]
        csv_path = os.path.join(COT_DATA_DIR, latest_file)
    if not csv_path or not os.path.exists(csv_path):
        print("[COT UTILS] Nenhum arquivo COT disponível localmente ou no Drive.")
        return None
    try:
        df = pd.read_csv(csv_path)
        df_symbol = df[df['my_symbol'] == symbol]
        if df_symbol.empty:
            return None
        row = df_symbol.sort_values("date").iloc[-1]
        # Calcule extremos e médias móveis (considerando colunas 'pct_long' e 'date')
        pct_long_series = df_symbol.sort_values("date")["pct_long"].tail(52)
        high_52w = pct_long_series.max() if len(pct_long_series) >= 10 else None
        low_52w = pct_long_series.min() if len(pct_long_series) >= 10 else None
        avg_4w = pct_long_series.tail(4).mean() if len(pct_long_series) >= 4 else None
        return {
            "net_position": row["net_position"],
            "pct_long": row["pct_long"],
            "open_interest": row["Open_Interest_All"],
            "date": row["date"],
            "52w_high": high_52w,
            "52w_low": low_52w,
            "4w_avg": avg_4w
        }
    except Exception as e:
        print(f"[COT UTILS] Erro ao ler COT: {e}")
        return None
