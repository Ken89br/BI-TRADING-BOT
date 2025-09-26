from datetime import datetime
import pytz

TZ_MAPUTO = pytz.timezone("Africa/Maputo")

def to_maputo_time(dt_utc):
    """Converte datetime UTC para o fuso horário de Moçambique."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
    return dt_utc.astimezone(TZ_MAPUTO)
