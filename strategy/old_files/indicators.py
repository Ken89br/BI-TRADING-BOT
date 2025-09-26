#O arquivo indicators.py serve como central de utilidades para cálculo de indicadores técnicos clássicos (RSI, MACD, ATR, ADX, médias móveis, volatilidade, volume, etc), normalmente usando bibliotecas como pandas e ta. Ele é ideal para:

#Calcular rapidamente indicadores para serem usados em várias estratégias diferentes (reutilização).
#Manter o código limpo, sem duplicidade de lógica de cálculo.
#Permitir que o ensemble ou qualquer estratégia complexa monte um "snapshot" completo do mercado, com todos os indicadores prontos.

import pandas as pd
import ta

def calc_rsi(close, period=14):
    """Calcula o RSI da lista de preços de fechamento."""
    return ta.momentum.RSIIndicator(pd.Series(close), window=period).rsi().iloc[-1]

def calc_macd(close):
    """Calcula o MACD, retornando histograma, linha MACD e linha de sinal."""
    macd = ta.trend.MACD(pd.Series(close))
    return macd.macd_diff().iloc[-1], macd.macd().iloc[-1], macd.macd_signal().iloc[-1]

def calc_bollinger(close, period=20):
    """Calcula as Bandas de Bollinger, retorna string explicativa, largura e posição."""
    bb = ta.volatility.BollingerBands(pd.Series(close), window=period)
    width = bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]
    pos = close[-1] - bb.bollinger_lband().iloc[-1]
    return (f"Bollinger width: {width:.5f}, Pos: {pos:.5f}", width, pos)

def calc_atr(high, low, close, period=14):
    """Calcula o ATR (Average True Range)."""
    return ta.volatility.AverageTrueRange(
        pd.Series(high), pd.Series(low), pd.Series(close), window=period
    ).average_true_range().iloc[-1]

def calc_adx(high, low, close, period=14):
    """Calcula o ADX (Average Directional Index)."""
    return ta.trend.ADXIndicator(
        pd.Series(high), pd.Series(low), pd.Series(close), window=period
    ).adx().iloc[-1]

def calc_moving_averages(close, fast=5, slow=20):
    """Compara médias móveis rápidas e lentas. Retorna 'buy', 'sell' ou 'neutral'."""
    fast_ma = pd.Series(close).rolling(window=fast).mean().iloc[-1]
    slow_ma = pd.Series(close).rolling(window=slow).mean().iloc[-1]
    if fast_ma > slow_ma:
        return "buy"
    elif fast_ma < slow_ma:
        return "sell"
    else:
        return "neutral"

def calc_oscillators(rsi, macd_hist):
    """Combina RSI e MACD histogram para sugerir 'buy', 'sell' ou 'neutral'."""
    if rsi > 70 and macd_hist < 0:
        return "sell"
    elif rsi < 30 and macd_hist > 0:
        return "buy"
    else:
        return "neutral"

def calc_volatility(close, period=14):
    """Determina se a volatilidade está alta ou baixa."""
    std = pd.Series(close).rolling(window=period).std().iloc[-1]
    median_std = pd.Series(close).rolling(window=period).std().median()
    return "High" if std > median_std else "Low"

def calc_volume_status(volume, period=20):
    """Classifica o status do volume."""
    vol = pd.Series(volume)
    ma = vol.rolling(window=period).mean().iloc[-1]
    if vol.iloc[-1] > ma * 1.5:
        return "Spiked"
    elif vol.iloc[-1] < ma * 0.7:
        return "Low"
    else:
        return "Normal"

def calc_sentiment(close):
    """Avalia sentimento simples baseado nos últimos candles."""
    if len(close) < 3:
        return "Neutral"
    if close[-1] > close[-2] > close[-3]:
        return "Optimistic"
    elif close[-1] < close[-2] < close[-3]:
        return "Pessimistic"
    else:
        return "Neutral"
