import pandas as pd

def resample_candles(df, freq='10S'):
    """
    Agrupa candles de 1s em janelas de freq (ex: '10S' para 10 segundos).
    Requer df com índice datetime e colunas open/high/low/close/volume.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    resampled = df.resample(freq).agg(ohlc_dict).dropna()
    resampled.reset_index(inplace=True)
    return resampled
