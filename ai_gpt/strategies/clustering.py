import numpy as np
import pandas as pd
import talib
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
import hashlib

# ==============================
# Cache Global
# ==============================
_cluster_cache = {}

def _get_cache_key(pair, timeframe, df, params):
    """Cria chave única para cache baseada no estado do mercado."""
    tail = df['Close'].tail(20).values if (df is not None and 'Close' in df.columns) else np.array([0.0])
    price_fingerprint = hashlib.md5(np.asarray(tail).tobytes()).hexdigest()
    return (
        pair,
        timeframe,
        price_fingerprint,
        int(params.get("min_distance_pips", 15)),
        int(params.get("min_cluster_size", 2))
    )

def _get_from_cache(cache_key, operation_type="daytrade"):
    """Tenta buscar clusters do cache se não expirou (TTL dinâmico por tipo de operação)."""
    ttl_map = {
        "scalping": timedelta(seconds=30),
        "daytrade": timedelta(minutes=5),
        "swing": timedelta(minutes=15),
        "position": timedelta(hours=1)
    }
    ttl = ttl_map.get(operation_type, timedelta(minutes=2))

    if cache_key in _cluster_cache:
        clustered, timestamp = _cluster_cache[cache_key]
        if datetime.now() - timestamp < ttl:
            return clustered
    return None

def _save_to_cache(cache_key, clustered):
    """Salva resultado no cache."""
    _cluster_cache[cache_key] = (clustered, datetime.now())

# ==============================
# Utilitários
# ==============================
def _volume_weighted_median(prices, volumes):
    """Calcula mediana ponderada por volume."""
    if not prices:
        return 0.0
    if not volumes or all(v == 0 for v in volumes):
        return float(np.median(prices))

    prices_arr = np.array(prices, dtype=float)
    volumes_arr = np.array(volumes, dtype=float)

    sorted_indices = np.argsort(prices_arr)
    sorted_prices = prices_arr[sorted_indices]
    sorted_volumes = volumes_arr[sorted_indices]

    cum_volumes = np.cumsum(sorted_volumes)
    total_volume = cum_volumes[-1] if len(cum_volumes) > 0 else 0.0
    if total_volume <= 0:
        return float(np.median(prices_arr))

    median_point = total_volume / 2.0
    idx = np.searchsorted(cum_volumes, median_point)
    idx = min(max(idx, 0), len(sorted_prices) - 1)
    return float(sorted_prices[idx])

def _find_fractal_candle_index(df, price, is_support=True, tol=1e-5):
    """Encontra o índice do candle onde o fractal ocorreu com tolerância numérica."""
    if df is None or df.empty:
        return None
    arr = df["Low"].values if is_support else df["High"].values
    matches = np.where(np.isclose(arr, price, rtol=tol, atol=tol))[0]
    return int(matches[0]) if len(matches) > 0 else None

def _safe_talib_atr(df, timeperiod):
    """(UTIL) Calcula ATR e devolve pd.Series; ainda disponível se quiseres usar fora deste módulo."""
    try:
        high = np.asarray(df['High'].values, dtype=float)
        low = np.asarray(df['Low'].values, dtype=float)
        close = np.asarray(df['Close'].values, dtype=float)
        atr_arr = talib.ATR(high, low, close, timeperiod=timeperiod)
        atr_series = pd.Series(atr_arr, index=df.index).fillna(0.0)
        return atr_series
    except Exception:
        return pd.Series(np.zeros(len(df)), index=df.index if df is not None else None)

# ==============================
# Clusterização ATR + Mediana (Forex)
# ==============================
def _cluster_fractals_atr_median(fractals, min_distance_pips, df, use_volume=True, atr_val=None):
    """
    Clusterização para Forex usando ATR + Mediana (ponderada por volume se confiável).
    Agora exige atr_val (float) — não faz cálculo interno de ATR.
    """
    if df is None or df.empty:
        return {"supports": [], "resistances": []}

    # Exigir ATR externo
    if atr_val is None:
        raise ValueError("_cluster_fractals_atr_median requires atr_val (float). Provide precomputed['atr'].")

    current_price = float(df["Close"].iloc[-1])
    raw_min = (float(min_distance_pips) / 10000.0) * current_price

    # ATR adaptativo (usa o atr_val fornecido)
    try:
        min_distance = float(np.clip(raw_min, 0.25 * atr_val, 2.5 * atr_val))
    except Exception:
        min_distance = raw_min

    # função para clusterizar um lado
    def cluster_side(levels, reverse=False, volumes=None):
        if not levels:
            return []
        lv = sorted(levels, reverse=reverse)
        cluster = [lv[0]]
        out = []
        cluster_volumes = [volumes[0]] if volumes else []
        for i, price in enumerate(lv[1:], start=1):
            if abs(price - cluster[-1]) < min_distance:
                cluster.append(price)
                if volumes:
                    cluster_volumes.append(volumes[i])
            else:
                if volumes and cluster_volumes:
                    out.append(_volume_weighted_median(cluster, cluster_volumes))
                else:
                    out.append(float(np.median(cluster)))
                cluster = [price]
                cluster_volumes = [volumes[i]] if volumes else []
        # último cluster
        if cluster:
            if volumes and cluster_volumes:
                out.append(_volume_weighted_median(cluster, cluster_volumes))
            else:
                out.append(float(np.median(cluster)))
        return out

    # volumes lookup (mantém fallback para volumes individuais, mas sem ATR interno)
    support_volumes, resistance_volumes = [], []
    if use_volume and "Volume" in df.columns and df['Volume'].notna().any():
        for s in fractals.get("supports", []):
            idx = _find_fractal_candle_index(df, s, is_support=True)
            support_volumes.append(float(df.iloc[idx]["Volume"]) if idx is not None else 1.0)
        for r in fractals.get("resistances", []):
            idx = _find_fractal_candle_index(df, r, is_support=False)
            resistance_volumes.append(float(df.iloc[idx]["Volume"]) if idx is not None else 1.0)

    supports_out = cluster_side(fractals.get("supports", []), reverse=False, volumes=support_volumes if support_volumes else None)
    resistances_out = cluster_side(fractals.get("resistances", []), reverse=True, volumes=resistance_volumes if resistance_volumes else None)

    return {
        "supports": supports_out,
        "resistances": resistances_out
    }

# ==============================
# Clusterização DBSCAN (Crypto/Índices)
# ==============================
def _cluster_dbscan(levels, df, params, atr_val=None):
    """Clusterização via DBSCAN melhorada para mercados não Forex."""
    if not levels:
        return []
    X = np.array(levels, dtype=float).reshape(-1, 1)

    # Se atr_val for fornecido, usa-o para epsilon; caso contrário, usa volatilidade recente (não é ATR)
    if atr_val and atr_val > 0:
        epsilon = float(max(atr_val * 0.6, 1e-9))
    else:
        recent_prices = df['Close'].tail(20).astype(float) if (df is not None and 'Close' in df.columns) else np.array([0.0])
        if len(recent_prices) > 1 and np.nanstd(recent_prices) > 0:
            epsilon = float(np.nanstd(recent_prices) * 0.5)
        else:
            # fallback mínimo baseado no preço atual (não cálculo de ATR)
            epsilon = 0.002 * float(df['Close'].iloc[-1]) if (df is not None and not df.empty) else 1e-6

    min_samples = int(params.get("min_cluster_size", 2))
    try:
        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(X)
    except Exception:
        # fallback simples: sem cluster, retorna median único por nível
        return [round(float(np.median([float(x) for x in levels])), 5)] if levels else []

    clusters = {}
    for i, label in enumerate(db.labels_):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(float(levels[i]))

    out = [round(float(np.median(c)), 5) for c in clusters.values()]
    out = sorted(out)
    return out

# ==============================
# Função principal (SEM CÁLCULO INTERNO DE ATR)
# ==============================
def cluster_levels_auto(all_supports, all_resistances, current_price, pair, params, df,
                        timeframe="default", use_volume=True, precomputed=None):
    """
    Decide automaticamente se usa ATR+Mediana (Forex) ou DBSCAN (Crypto/Índices).
    NÃO calcula ATR internamente: exige precomputed['atr'] (float) fornecido externamente.
    Retorna: (supports_list, resistances_list) — ambas listas ordenadas.
    """
    # validações iniciais
    if all_supports is None:
        all_supports = []
    if all_resistances is None:
        all_resistances = []
    if df is None:
        df = pd.DataFrame()

    # operation_type para TTL (opcional)
    operation_type = None
    if precomputed and isinstance(precomputed, dict):
        operation_type = precomputed.get('operation_type')

    cache_key = _get_cache_key(pair, timeframe, df, params)
    cached = _get_from_cache(cache_key, operation_type or "daytrade")
    if cached:
        supports = cached.get("supports", [])
        resistances = cached.get("resistances", [])
        return sorted(supports), sorted(resistances, reverse=True)

    # exigir ATR externo (sem cálculo interno)
    if not precomputed or not isinstance(precomputed, dict) or precomputed.get("atr") is None:
        raise ValueError("cluster_levels_auto requires precomputed['atr'] (float). "
                         "It does NOT compute ATR internally. Compute ATR outside and pass it in precomputed.")

    try:
        atr_val = float(precomputed.get("atr"))
    except Exception:
        raise ValueError("precomputed['atr'] must be a numeric float convertible value.")

    # heurística simples para detectar forex
    forex_pairs = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
    pair_up = (pair or "").upper()

    # decidir método
    try:
        is_forex = any(pair_up.endswith(fx) for fx in forex_pairs)
    except Exception:
        is_forex = False

    if is_forex:
        clustered = _cluster_fractals_atr_median(
            {"supports": all_supports, "resistances": all_resistances},
            params.get("min_distance_pips", 15),
            df,
            use_volume,
            atr_val
        )
    else:
        clustered = {
            "supports": _cluster_dbscan(all_supports, df, params, atr_val),
            "resistances": _cluster_dbscan(all_resistances, df, params, atr_val),
        }

    # Garantir listas e ordenar (supports asc, resistances desc)
    supports_out = clustered.get("supports", []) or []
    resistances_out = clustered.get("resistances", []) or []

    supports_out = sorted([float(x) for x in supports_out]) if supports_out else []
    resistances_out = sorted([float(x) for x in resistances_out], reverse=True) if resistances_out else []

    clustered_safe = {"supports": supports_out, "resistances": resistances_out}
    _save_to_cache(cache_key, clustered_safe)

    return supports_out, resistances_out
