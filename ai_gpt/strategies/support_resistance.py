import time
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import talib
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from strategies.clustering import cluster_levels_auto
import logging
from scipy.ndimage import gaussian_filter1d
import json
import hashlib
from Utility.cache_manager import global_cache, make_cache_key
import logging
from Utility.logging_config import get_trading_logger
from data_processor import MarketAwareDataProcessor

logger = get_trading_logger("support_resistance")

class AdvancedSRAnalyzer:
    def __init__(self, data_client):
        """
        Analisador avançado de S/R com:
        - Machine Learning para clusterização
        - Análise de liquidez em tempo real
        - Integração multi-timeframe
        - Confirmação por volume e fluxo de ordens
        - Pivot Points tradicionais e ponderados por volume
        """
        logger.info("Inicializando AdvancedSRAnalyzer")
        self.data_client = data_client

    def _get_optimal_params(self, operation_type: str) -> dict:
        """Parâmetros otimizados por tipo de operação e paridade"""
        logger.debug(f"Obtendo parâmetros para {operation_type}")
        params = {
            'scalping': {
                'base_tf': '1min',
                'aux_tf': '5min',
                'lookback': 120,  # 2 horas
                'tolerance_pct': 0.002,
                'volume_factor': 1.8,
                'min_cluster_size': 3,
                'min_touches': 2,
                'ema_window': 9,
                'liquidity_threshold': 1.5,
                'pivot_lookback': 30  # Número de candles para cálculo dos pivots
            },
            'daytrade': {
                'base_tf': '5min',
                'aux_tf': '15min',
                'lookback': 96,  # 8 horas
                'tolerance_pct': 0.005,
                'volume_factor': 1.5,
                'min_cluster_size': 3,
                'min_touches': 2,
                'ema_window': 21,
                'liquidity_threshold': 1.3,
                'pivot_lookback': 24
            },
            'swing': {
                'base_tf': '1H',
                'aux_tf': '4H',
                'lookback': 168,  # 1 semana
                'tolerance_pct': 0.01,
                'volume_factor': 1.2,
                'min_cluster_size': 2,
                'min_touches': 2,
                'ema_window': 50,
                'liquidity_threshold': 1.1,
                'pivot_lookback': 24
            }
        }.get(operation_type)
        
        if not params:
            logger.warning(f"Tipo de operação não encontrado: {operation_type}")
        else:
            logger.debug(f"Parâmetros carregados para {operation_type}")
            
        return params

    def _fetch_multi_timeframe_data(self, pair: str, params: dict) -> dict:
        """Obtém dados de múltiplos timeframes para análise conjunta (base, aux e pivot).
        Futuramente poderá incluir também um terceiro timeframe de sinal (signal_tf).
        """
        logger.debug(f"Buscando dados multi-timeframe para {pair}")
        
        try:
            # Base timeframe (principal)
            base_data = self.data_client.fetch_candles(
                pair, interval=params['base_tf'], limit=params['lookback']
            )

            # Aux timeframe (confirmação)
            aux_data = self.data_client.fetch_candles(
                pair, interval=params['aux_tf'], limit=params['lookback'] // 4
            )

            # Pivot timeframe (não é considerado timeframe principal, apenas para cálculos auxiliares)
            pivot_data = self.data_client.fetch_candles(
                pair, interval=params['base_tf'], limit=params['pivot_lookback']
            )

            # Futuro: terceiro timeframe de sinal (signal_tf)
            signal_data = None
            if 'signal_tf' in params and params['signal_tf']:
                try:
                    signal_data = self.data_client.fetch_candles(
                        pair, interval=params['signal_tf'], limit=params['lookback'] // 2
                    )
                    logger.debug(f"Signal timeframe carregado: {params['signal_tf']} ({len(signal_data)} candles)")
                except Exception as e:
                    logger.warning(f"Erro ao buscar signal_tf para {pair}: {e}")
                    signal_data = None

            # Usar MarketAwareDataProcessor no lugar do _process_raw_data
            result = {
                'base': MarketAwareDataProcessor.process_market_aware_data(base_data, pair, operation_type=params.get("operation_type", "daytrade")),
                'aux': MarketAwareDataProcessor.process_market_aware_data(aux_data, pair, operation_type=params.get("operation_type", "daytrade")),
                'pivot': MarketAwareDataProcessor.process_market_aware_data(pivot_data, pair, operation_type=params.get("operation_type", "daytrade")),
                'signal': MarketAwareDataProcessor.process_market_aware_data(signal_data, pair, operation_type=params.get("operation_type", "daytrade")) 
                          if signal_data is not None else pd.DataFrame()
            }

            logger.debug(
                f"Dados obtidos: base={len(result['base'])}, "
                f"aux={len(result['aux'])}, pivot={len(result['pivot'])}, "
                f"signal={len(result['signal']) if not result['signal'].empty else 0}"
            )
            return result

        except Exception as e:
            logger.error(f"Erro ao buscar dados para {pair}: {e}", exc_info=True)
            return {}
            
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcula pivot points tradicionais"""
        if len(df) < 2:
            logger.warning("DataFrame muito pequeno para calcular pivots")
            return {}
            
        try:
            # Usa o último candle completo
            last_candle = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            high = last_candle['High']
            low = last_candle['Low']
            close = last_candle['Close']
            
            pp = (high + low + close) / 3
            pivots = {
                'pivot': pp,
                'r1': 2*pp - low,
                's1': 2*pp - high,
                'r2': pp + (high - low),
                's2': pp - (high - low),
                'r3': high + 2*(pp - low),
                's3': low - 2*(high - pp),
                'pivot_high': high,
                'pivot_low': low
            }
            
            logger.debug(f"Pivot Points calculados: PP={pp:.5f}")
            return pivots
            
        except Exception as e:
            logger.error(f"Erro no cálculo de Pivot Points: {e}")
            return {}

    def _calculate_adaptive_vwap(
        self,
        df: pd.DataFrame,
        operation_type: str,
        reset_mode: str = "session",
        use_tick_volume: bool = True
    ) -> dict:
        """
        Calcula VWAP adaptativo
        """
        logger.debug(f"Calculando VWAP adaptativo para {operation_type}, modo {reset_mode}")
        
        try:
            # Configurações por tipo de operação
            tf_settings = {
                'scalping': {'std_period': 15, 'n_std_1': 1.0, 'n_std_2': 1.5},
                'daytrade': {'std_period': 20, 'n_std_1': 1.5, 'n_std_2': 2.0},
                'swing': {'std_period': 50, 'n_std_1': 2.0, 'n_std_2': 2.5}
            }.get(operation_type, {'std_period': 20, 'n_std_1': 1.5, 'n_std_2': 2.0})

            typical = (df['High'] + df['Low'] + df['Close']) / 3

            # Escolher volume a ser usado
            if use_tick_volume and 'TickVolume' in df.columns:
                volume = df['TickVolume']
                logger.debug("Usando TickVolume para VWAP")
            else:
                volume = df['Volume']
                logger.debug("Usando Volume para VWAP")

            # Cálculo do VWAP com reset
            if reset_mode == "continuous":
                cum_vol = volume.cumsum()
                cum_vol_price = (typical * volume).cumsum()
                cum_vol_safe = cum_vol.replace(0, np.nan).ffill().bfill()
                vwap_series = cum_vol_price / cum_vol_safe

            elif reset_mode == "daily":
                dates = pd.to_datetime(df['Date']).dt.date
                vwap_series = df.groupby(dates).apply(
                    lambda g: (g['Volume'] * typical.loc[g.index]).cumsum() / g['Volume'].cumsum()
                ).reset_index(level=0, drop=True)

            elif reset_mode == "session":
                if 'Session' not in df.columns:
                    df['Session'] = (df['Date'].diff() > pd.Timedelta('1h')).cumsum()
                vwap_series = df.groupby('Session').apply(
                    lambda g: (g['Volume'] * typical.loc[g.index]).cumsum() / g['Volume'].cumsum()
                ).reset_index(level=0, drop=True)

            else:
                raise ValueError("reset_mode deve ser 'continuous', 'daily' ou 'session'")

            # Detecção de tendência
            close_price = df['Close'].iloc[-1]
            vwap_now = vwap_series.iloc[-1]
            vwap_slope = vwap_series.iloc[-1] - vwap_series.iloc[-5] if len(vwap_series) > 5 else 0

            trend = "up" if (close_price > vwap_now and vwap_slope > 0) else \
                    "down" if (close_price < vwap_now and vwap_slope < 0) else "sideways"

            # Bandas de volatilidade
            std = typical.rolling(tf_settings['std_period']).std()
            bands = {
                'upper1': (vwap_series + tf_settings['n_std_1'] * std).iloc[-1],
                'lower1': (vwap_series - tf_settings['n_std_1'] * std).iloc[-1],
                'upper2': (vwap_series + tf_settings['n_std_2'] * std).iloc[-1],
                'lower2': (vwap_series - tf_settings['n_std_2'] * std).iloc[-1],
                'mid': vwap_now
            }

            result = {
                'vwap': vwap_now,
                'trend': trend,
                'bands': bands,
                'slope': vwap_slope
            }
            
            logger.debug(f"VWAP calculado: {vwap_now:.5f}, Tendência: {trend}")
            return result

        except Exception as e:
            logger.error(f"Erro no cálculo do VWAP: {e}")
            return {}

    def _calculate_volume_nodes(self, 
                                df: pd.DataFrame, 
                                mode: str = "auto", 
                                bins: int = 100, 
                                smooth_sigma: float = None, 
                                pair: str = "", 
                                operation_type: str = "daytrade") -> dict:
        """
        Calcula nós de volume (POC e Value Area)

        adaptativos:
            - "raw": histograma por volume (Volume Profile clássico institucional)
            - "smooth": histograma suavizado (elimina serrilhado e revela clusters)
            - "auto": detecta se o par é Forex ou Crypto e ajusta automaticamente

            REFORÇO: Seleção inteligente do tipo de volume por mercado:
            - Forex: Prioriza Tick Volume (nº de ticks/atualizações)
            - Crypto/Ações: Prioriza Volume Real (quantidade negociada)

            Args:
                df (pd.DataFrame): DataFrame com colunas ['Typical', 'Volume']
                mode (str): "raw", "smooth" ou "auto"
                bins (int): número máximo de bins (intervalos de preço)
                smooth_sigma (float): intensidade da suavização Gaussiana (para smooth/auto)
                pair (str): nome do ativo/par (ex: "EURUSD", "BTCUSDT")
                operation_type (str): tipo de operação (ex: scalping, daytrade, swing, position)

            Returns:
                dict: {
                    'poc': preço do Point of Control,
                    'value_area': (mínimo, máximo) da Value Area 70%,
                    'clusters': lista de preços representando picos (apenas no modo smooth),
                    'volume_distribution': lista de tuplas (preço, volume)
                }
            """
            
        logger.debug(f"Calculando Volume Nodes para {pair}, modo {mode}")
        
        # --------- CACHE CHECK ---------
        params = {"mode": mode, "bins": bins, "smooth_sigma": smooth_sigma}
        cache_key = make_cache_key(pair, f"volume_nodes_{operation_type}", params, cache_namespace="sr")
        cached = global_cache.get(cache_key)
        if cached is not None:
            logger.debug("Retornando Volume Nodes do cache")
            return cached
        # -------------------------------

        # Validação inicial
        if df is None or df.empty or 'Typical' not in df.columns:
            logger.warning("DataFrame inválido para Volume Nodes")
            return {'poc': None, 'value_area': (None, None), 'clusters': [], 'volume_distribution': []}

        try:
            # SELEÇÃO INTELIGENTE DE VOLUME
            forex_suffixes = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
            is_forex = any((pair or "").upper().endswith(sfx) for sfx in forex_suffixes)
            
            # Decisão: qual coluna de volume usar?
            volume_col_to_use = None
            
            # Prioridade para Forex: Tick Volume
            if is_forex:
                if 'TickVolume' in df.columns and df['TickVolume'].notnull().any():
                    volume_col_to_use = 'TickVolume'
                    logger.debug(f"Forex: usando TickVolume para {pair}")
                elif 'Volume' in df.columns and df['Volume'].notnull().any():
                    volume_col_to_use = 'Volume'
                    logger.debug(f"Forex: fallback para Volume para {pair}")
            else:
                # Para Crypto/Ações: Volume Real
                if 'Volume' in df.columns and df['Volume'].notnull().any():
                    volume_col_to_use = 'Volume'
                    logger.debug(f"Crypto/Ações: usando Volume para {pair}")
                elif 'TickVolume' in df.columns and df['TickVolume'].notnull().any():
                    volume_col_to_use = 'TickVolume'
                    logger.debug(f"Crypto/Ações: fallback para TickVolume para {pair}")
            
            # Fallback final
            if volume_col_to_use is None:
                volumes = np.ones(len(df))
                logger.warning(f"Usando volume uniforme para {pair} (dados insuficientes)")
            else:
                volumes = df[volume_col_to_use].values

            prices = df['Typical'].values

            if len(prices) < 10:
                logger.warning("Dados insuficientes para Volume Nodes")
                return {'poc': None, 'value_area': (None, None), 'clusters': [], 'volume_distribution': []}

            # Heurísticas automáticas
            if is_forex:
                price_decimals = 5
            else:
                last_price = float(prices[-1])
                if last_price > 1000:   # BTC, ETH etc.
                    price_decimals = 2
                elif last_price > 1:    # mid-cap tokens
                    price_decimals = 4
                else:                   # micro moedas (SHIB, DOGE, etc.)
                    price_decimals = 8

            if smooth_sigma is None:
                smooth_sigma = 1.8 if is_forex else 3.0

            # modo automático
            if mode == "auto":
                mode = "raw" if is_forex else "smooth"

            # Histograma por volume
            bins = max(20, min(len(prices) // 2, bins))
            hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            if hist.sum() == 0:
                result = {
                    'poc': None,
                    'value_area': (None, None),
                    'clusters': [],
                    'volume_distribution': [(float(round(c, price_decimals)), 0.0) for c in centers]
                }
                global_cache.set(cache_key, result, expiry=600)
                return result

            # Modo RAW
            if mode == "raw":
                poc_idx = np.argmax(hist)
                poc_price = centers[poc_idx]

                total_vol = hist.sum()
                target_vol = 0.7 * total_vol

                order = np.argsort(hist)[::-1]
                cum_vol = 0
                selected_bins = []

                for idx in order:
                    cum_vol += hist[idx]
                    selected_bins.append(centers[idx])
                    if cum_vol >= target_vol:
                        break

                va_low, va_high = min(selected_bins), max(selected_bins)

                result = {
                    'poc': float(round(poc_price, price_decimals)),
                    'value_area': (float(round(va_low, price_decimals)), float(round(va_high, price_decimals))),
                    'clusters': [],
                    'volume_distribution': [(float(round(c, price_decimals)), float(v)) for c, v in zip(centers, hist)],
                    'volume_type_used': volume_col_to_use
                }

            # Modo SMOOTH
            elif mode == "smooth":
                smooth_hist = gaussian_filter1d(hist, sigma=smooth_sigma)

                poc_idx = np.argmax(smooth_hist)
                poc_price = centers[poc_idx]

                total_vol = smooth_hist.sum()
                target_vol = 0.7 * total_vol

                order = np.argsort(smooth_hist)[::-1]
                cum_vol = 0
                selected_bins = []

                for idx in order:
                    cum_vol += smooth_hist[idx]
                    selected_bins.append(centers[idx])
                    if cum_vol >= target_vol:
                        break

                va_low, va_high = min(selected_bins), max(selected_bins)

                # Clusters = picos locais
                peaks = np.where(
                    (smooth_hist[1:-1] > smooth_hist[:-2]) &
                    (smooth_hist[1:-1] > smooth_hist[2:])
                )[0] + 1
                clusters = [float(round(centers[i], price_decimals)) for i in peaks]

                result = {
                    'poc': float(round(poc_price, price_decimals)),
                    'value_area': (float(round(va_low, price_decimals)), float(round(va_high, price_decimals))),
                    'clusters': clusters,
                    'volume_distribution': [(float(round(c, price_decimals)), float(v)) for c, v in zip(centers, smooth_hist)],
                    'volume_type_used': volume_col_to_use
                }

            else:
                raise ValueError("Modo inválido em _calculate_volume_nodes. Use 'raw', 'smooth' ou 'auto'.")

            # Cache
            expiry_map = {
                'scalping': 300,      # 5 minutos
                'daytrade': 900,      # 15 minutos
                'swing': 3600,        # 1 hora
                'position': 21600     # 6 horas
            }
            expiry = expiry_map.get(operation_type, 900)
            global_cache.set(cache_key, result, expiry=expiry)

            logger.debug(f"Volume Nodes calculado: POC={result['poc']}, Clusters={len(result.get('clusters', []))}")
            return result

        except Exception as e:
            logger.error(f"Erro no cálculo de Volume Nodes: {e}")
            return {'poc': None, 'value_area': (None, None), 'clusters': [], 'volume_distribution': []}
            
    def _find_liquidity_zones(self, df: pd.DataFrame, params: dict, pair: str = "", operation_type: str = "daytrade") -> dict:
        """
        Detecta zonas de liquidez (order blocks) com mitigação, força adaptativa
        e opção de range configurável.
        
        Args:
            df (pd.DataFrame): OHLCV com colunas mínimas ['Open','High','Low','Close','Volume']
            params (dict): parâmetros de configuração (opcionais)
                - liquidity_threshold (float): múltiplo da média de volume (default 1.5)
                - range_mode (str): "body" ou "full" (candle inteiro) (default "body")
                - min_strength (float): mínimo de força p/ considerar zona (default 0.8)
                - use_vwap (float|None): VWAP atual (opcional, p/ confluência)
                - use_pivot (float|None): Pivot atual (opcional, p/ confluência)
                - vol_ma_period (int): janela do moving average do volume (default 20)
                - mitigation_lookahead (int|None): número de candles após a zona a checar mitigação (None = até o fim)
                - max_zones (int): máximo de zonas retornadas (default 5)

        Returns:
            dict: {'zones': [ {type, range, price, time, index, volume, strength} ]}
        """

        logger.debug(f"Buscando zonas de liquidez para {pair} ({operation_type})")
        
        # --------- CACHE CHECK ---------
        cache_key = make_cache_key(pair, f"liqzones_{operation_type}", params, cache_namespace="sr")
        cached = global_cache.get(cache_key)
        if cached is not None:
            logger.debug("Retornando zonas de liquidez do cache")
            return cached
        # -------------------------------
        
        # Defensive checks
        if df is None or df.empty:
            logger.warning("DataFrame vazio para zonas de liquidez")
            return {'zones': []}

        try:
            # Work on a copy to avoid side-effects
            df = df.copy()

            # Required price columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Colunas necessárias não encontradas no DataFrame")
                return {'zones': []}

            # Ensure Date column
            if 'Date' not in df.columns:
                if 'timestamp' in df.columns:
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                else:
                    try:
                        df['Date'] = pd.to_datetime(df.index)
                    except Exception:
                        df['Date'] = pd.to_datetime(df.index, errors='coerce')

            # Força tipos numéricos
            for c in required_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].ffill().bfill()

            # Params com defaults
            vol_ma_period = int(params.get('vol_ma_period', 20))
            liquidity_threshold = float(params.get('liquidity_threshold', 1.5))
            range_mode = params.get('range_mode', 'body')
            min_strength = float(params.get('min_strength', 0.8))
            mitigation_lookahead = params.get('mitigation_lookahead', None)  # number of candles to check after zone (None => until end)
            max_zones = int(params.get('max_zones', 5))

            logger.debug(f"Parâmetros: vol_ma_period={vol_ma_period}, liquidity_threshold={liquidity_threshold}, range_mode={range_mode}")

            # Confluence levels
            ref_levels = [params.get("use_vwap"), params.get("use_pivot")]
            ref_levels = [lvl for lvl in ref_levels if lvl is not None and not (isinstance(lvl, float) and np.isnan(lvl))]

            # Volume moving average
            vol_ma = df['Volume'].rolling(window=vol_ma_period, min_periods=1).mean().replace(0, 1).fillna(1)

            liquidity_zones = []
            n = len(df)

            logger.debug(f"Analisando {n} candles para zonas de liquidez")

            # Percorre candles com espaço para olhar adiante
            for i in range(3, n - 2):
                open_i = df['Open'].iat[i]
                close_i = df['Close'].iat[i]
                high_i = df['High'].iat[i]
                low_i = df['Low'].iat[i]
                close_i1 = df['Close'].iat[i + 1]
                close_i2 = df['Close'].iat[i + 2]

                current_vol = df['Volume'].iat[i]
                vol_ma_i = vol_ma.iat[i] if not np.isnan(vol_ma.iat[i]) else 1.0
                vol_ma_i = max(vol_ma_i, 1e-9)

                # Filtro simples de volume acima da média
                if current_vol <= vol_ma_i * liquidity_threshold:
                    continue

                # ------------------------------
                # Bullish Order Block (OB)
                # condição: candle de rejeição (fechamento < abertura) seguido por candles que rompem acima do high
                # ------------------------------
                if (close_i < open_i) and (close_i1 > high_i) and (close_i2 > high_i):
                    # range dependendo do modo
                    if range_mode == "body":
                        zone_low = min(low_i, open_i)
                        zone_high = max(low_i, open_i)
                    else:  # full candle
                        zone_low = low_i
                        zone_high = high_i

                    # breakout strength (seguro para divisão por zero)
                    breakout_strength = 0.0
                    if abs(high_i) > 1e-9:
                        breakout_strength = (close_i1 - high_i) / abs(high_i)

                    # força combinada: volume relativo + força do rompimento (+ confluência)
                    strength = 0.5 * (current_vol / vol_ma_i) + 0.3 * breakout_strength

                    if ref_levels:
                        dists = []
                        for lvl in ref_levels:
                            if lvl and abs(lvl) > 1e-9:
                                dists.append(abs(close_i - lvl) / abs(lvl))
                        if dists:
                            dist = min(dists)
                            strength += max(0, 0.2 * (1 - dist))  # massimo +0.2 quando muito próximo

                    strength = max(0.0, float(strength))

                    liquidity_zones.append({
                        'type': 'bullish',
                        'range': (round(zone_low, 8), round(zone_high, 8)),
                        'price': round(zone_low, 8),  # base da zona (suporte esperado)
                        'time': pd.to_datetime(df['Date'].iat[i]),
                        'index': int(i),
                        'volume': float(current_vol),
                        'strength': round(strength, 3)
                    })

                # ------------------------------
                # Bearish Order Block (OB)
                # condição: candle de rejeição altista (fechamento > abertura) seguido por candles que rompem abaixo do low
                # ------------------------------
                elif (close_i > open_i) and (close_i1 < low_i) and (close_i2 < low_i):
                    if range_mode == "body":
                        zone_low = open_i
                        zone_high = high_i
                    else:
                        zone_low = low_i
                        zone_high = high_i

                    breakout_strength = 0.0
                    if abs(low_i) > 1e-9:
                        breakout_strength = (low_i - close_i1) / abs(low_i)

                    strength = 0.5 * (current_vol / vol_ma_i) + 0.3 * breakout_strength

                    if ref_levels:
                        dists = []
                        for lvl in ref_levels:
                            if lvl and abs(lvl) > 1e-9:
                                dists.append(abs(close_i - lvl) / abs(lvl))
                        if dists:
                            dist = min(dists)
                            strength += max(0, 0.2 * (1 - dist))

                    strength = max(0.0, float(strength))

                    liquidity_zones.append({
                        'type': 'bearish',
                        'range': (round(zone_low, 8), round(zone_high, 8)),
                        'price': round(zone_high, 8),  # topo da zona (resistência esperada)
                        'time': pd.to_datetime(df['Date'].iat[i]),
                        'index': int(i),
                        'volume': float(current_vol),
                        'strength': round(strength, 3)
                    })

            # ------------------------------
            # Mitigação: remove zonas que foram tocadas depois do aparecimento (optionally limit lookahead)
            # ------------------------------
            non_mitigated = []
            for z in liquidity_zones:
                start_idx = z.get('index', None)
                if start_idx is None:
                    non_mitigated.append(z)
                    continue

                lookahead_end = n if mitigation_lookahead is None else min(n, start_idx + 1 + int(mitigation_lookahead))
                after_zone = df.iloc[start_idx + 1: lookahead_end]

                if after_zone.empty:
                    non_mitigated.append(z)
                    continue

                zone_low, zone_high = float(z['range'][0]), float(z['range'][1])
                touched_mask = (after_zone['Low'] <= zone_high) & (after_zone['High'] >= zone_low)
                touched = touched_mask.any()

                if not touched:
                    non_mitigated.append(z)

            # Filtrar por força mínima
            strong_zones = [z for z in non_mitigated if z.get('strength', 0) >= min_strength]
            strong_zones = sorted(strong_zones, key=lambda x: x['time'])
            result = {'zones': strong_zones[-max_zones:]}

            logger.debug(f"Zonas de liquidez encontradas: {len(result['zones'])} (força >= {min_strength})")

            # Cache
            expiry_map = {
                'scalping': 300,
                'daytrade': 900,
                'swing': 3600,
                'position': 21600
            }
            expiry = expiry_map.get(operation_type, 900)
            global_cache.set(cache_key, result, expiry=expiry)

            return result

        except Exception as e:
            logger.error(f"Erro na detecção de zonas de liquidez: {e}", exc_info=True)
            return {'zones': []}
            
    def _find_valid_fractals(self, df: pd.DataFrame, params: dict, pair: str = "", operation_type: str = "daytrade") -> dict:
        """
        ENCONTRA FRACTAIS VÁLIDOS com validações
        """
        logger.debug(f"Buscando fractais válidos para {pair} ({operation_type})")
        
        # --------- CACHE CHECK ---------
        cache_key = make_cache_key(pair, f"fractals_{operation_type}", params, cache_namespace="sr")
        cached = global_cache.get(cache_key)
        if cached is not None:
            logger.debug("Retornando fractais do cache")
            return cached
        # -------------------------------
        
        # Retorno padrão em caso de erro
        default_return = {
            'fractals': {'supports': [], 'resistances': []},
            'details': {'supports_info': [], 'resistances_info': []},
            'metadata': {
                'total_supports': 0, 
                'total_resistances': 0,
                'operation_type': params.get('operation_type', 'daytrade'),
                'timestamp': datetime.now(),
                'params_used': {}
            }
        }
        
        try:
            # VALIDAÇÃO: DataFrame vazio ou insuficiente
            if df is None or len(df) < 10:
                logger.warning("DataFrame vazio ou insuficiente para fractais")
                return default_return
                
            # VALIDAÇÃO: Colunas necessárias
            required_cols = ['Low', 'High', 'Volume', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Colunas necessárias não encontradas no DataFrame")
                return default_return

            low_values = df['Low'].values
            high_values = df['High'].values
            volume_values = df['Volume'].values
            close_values = df['Close'].values

            # Configurações adaptativas
            op_type = params.get('operation_type', 'daytrade')
            adaptive_params = {
                'scalping': {
                    'vol_ma_period': 3, 'volume_factor': 2.0,
                    'size_factor': 0.6, 'volatility_factor': 0.10, 'require_confirmation': True,
                    'min_distance_pips': 8, 'lookback_candles': 3, 'trend_weight': 0.4
                },
                'daytrade': {
                    'vol_ma_period': 5, 'volume_factor': 1.5,
                    'size_factor': 0.7, 'volatility_factor': 0.15, 'require_confirmation': True,
                    'min_distance_pips': 15, 'lookback_candles': 5, 'trend_weight': 0.3
                },
                'swing': {
                    'vol_ma_period': 8, 'volume_factor': 1.2,
                    'size_factor': 0.8, 'volatility_factor': 0.20, 'require_confirmation': False,
                    'min_distance_pips': 30, 'lookback_candles': 8, 'trend_weight': 0.2
                },
                'position': {
                    'vol_ma_period': 13, 'volume_factor': 1.0,
                    'size_factor': 0.9, 'volatility_factor': 0.25, 'require_confirmation': False,
                    'min_distance_pips': 50, 'lookback_candles': 13, 'trend_weight': 0.1
                }
            }
            ap = adaptive_params.get(op_type, adaptive_params['daytrade'])

            logger.debug(f"Configuração adaptativa: {ap}")

            # Parâmetros finais
            vol_ma_period = params.get('vol_ma_period', ap['vol_ma_period'])
            volume_factor = params.get('volume_factor', ap['volume_factor'])
            size_factor = params.get('size_factor', ap['size_factor'])
            volatility_factor = params.get('volatility_factor', ap['volatility_factor'])
            require_confirmation = params.get('require_confirmation', ap['require_confirmation'])
            min_distance_pips = params.get('min_distance_pips', ap['min_distance_pips'])
            lookback_candles = params.get('lookback_candles', ap['lookback_candles'])
            trend_weight = params.get('trend_weight', ap['trend_weight'])

            # Volume MA
            try:
                if 'vol_ma' in params:
                    vol_ma_values = params['vol_ma']
                else:
                    vol_ma_values = df['Volume'].rolling(vol_ma_period, min_periods=1).mean()
                    vol_ma_values = vol_ma_values.fillna(0).values
            except Exception as e:
                logger.warning(f"Erro no cálculo do Volume MA: {e}")
                vol_ma_values = np.zeros(len(df))

            # Price MA
            try:
                if 'price_ma' in params:
                    price_ma_values = params['price_ma']
                else:
                    price_ma_period = 20 if op_type == 'daytrade' else 50
                    price_ma_values = df['Close'].rolling(price_ma_period, min_periods=1).mean()
                    price_ma_values = price_ma_values.fillna(0).values
            except Exception as e:
                logger.warning(f"Erro no cálculo do Price MA: {e}")
                price_ma_values = np.zeros(len(df))

            # Tamanho médio das velas
            try:
                candle_sizes = high_values - low_values
                avg_candle_size = pd.Series(candle_sizes).rolling(
                    window=lookback_candles, min_periods=1
                ).mean().values
            except Exception as e:
                logger.warning(f"Erro no cálculo do tamanho médio das velas: {e}")
                avg_candle_size = np.zeros(len(df))

            # =========================
            # Garantir ATR disponível (valor escalar para cluster_utils)
            # =========================
            atr_val_float = None
            try:
                if 'atr' in params and params['atr'] is not None:
                    atr_param = params['atr']
                    if hasattr(atr_param, '__len__'):
                        atr_series_local = pd.Series(atr_param).fillna(method='bfill').fillna(0)
                        atr_val_float = float(atr_series_local.iloc[-1])
                        params['atr'] = atr_series_local.values
                    else:
                        atr_val_float = float(atr_param)
                else:
                    try:
                        atr_arr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
                        atr_series_local = pd.Series(atr_arr, index=df.index).fillna(method='bfill').fillna(0)
                        params['atr'] = atr_series_local.values
                        atr_val_float = float(atr_series_local.iloc[-1]) if len(atr_series_local) > 0 else 0.0
                    except Exception:
                        recent_range = (df['High'] - df['Low']).rolling(14, min_periods=1).mean().iloc[-1]
                        atr_val_float = float(recent_range if not np.isnan(recent_range) else 0.0)
                        params['atr'] = np.full(len(df), atr_val_float)
            except Exception as e:
                logger.warning(f"Erro ao garantir ATR: {e}")
                atr_val_float = 0.0
                params['atr'] = np.zeros(len(df))

            fractals = {'supports': [], 'resistances': []}
            fractal_details = {'supports_info': [], 'resistances_info': []}

            # DETECÇÃO DE FRACTAIS CRUS
            fractal_lookback = params.get('fractal_lookback', 200)
            start_idx = max(2, len(df) - fractal_lookback)
            if start_idx >= len(df) - 2:
                start_idx = max(2, len(df) - 10)

            logger.debug(f"Analisando fractais de {start_idx} a {len(df) - 2}")

            valid_fractals_count = 0
            for i in range(start_idx, len(df) - 2):
                try:
                    if i >= len(low_values) or i >= len(high_values) or i >= len(vol_ma_values):
                        continue

                    current_low = low_values[i]
                    current_high = high_values[i]
                    current_vol = volume_values[i]

                    # filtro de volume
                    if i >= len(vol_ma_values) or current_vol < vol_ma_values[i] * volume_factor:
                        continue

                    # Validação de tendência
                    trend_strength = 1.0
                    if trend_weight > 0 and i < len(price_ma_values) and price_ma_values[i] != 0:
                        price_vs_ma = close_values[i] / price_ma_values[i]
                        if price_vs_ma > 1.02:  # mercado pode estar em alta
                            trend_strength += trend_weight * 0.5
                        elif price_vs_ma < 0.98:  # mercado pode estar em baixa
                            trend_strength -= trend_weight * 0.5

                        # 2) Confirmação con ADX (se disponível)
                        if 'adx' in params and params['adx'] is not None and i < len(params['adx']):
                            adx_value = params['adx'][i]
                            if adx_value < 20:  # mercado lateral
                                trend_strength -= trend_weight
                            elif adx_value > 25:  # tendência mais forte
                                trend_strength += trend_weight * 0.3

                        # 3) Confirmação con médias móveis rápidas/lentas
                        if ('ema_fast' in params and 'ema_slow' in params and 
                            i < len(params['ema_fast']) and i < len(params['ema_slow'])):
                            if params['ema_fast'][i] > params['ema_slow'][i]:
                                trend_strength += trend_weight * 0.3
                            else:
                                trend_strength -= trend_weight * 0.3

                        # 4) RSI como filtro
                        if 'rsi' in params and params['rsi'] is not None and i < len(params['rsi']):
                            rsi_value = params['rsi'][i]
                            if rsi_value > 70 or rsi_value < 30:
                                trend_strength -= trend_weight * 0.2

                    # Validação de tamanho/volatilidade
                    candle_size = current_high - current_low
                    
                    size_ok = (i < len(avg_candle_size) and 
                              candle_size >= avg_candle_size[i] * size_factor)
                    
                    volatility_ok = False
                    if 'atr' in params and params['atr'] is not None and i < len(params['atr']):
                        try:
                            volatility_ok = candle_size >= params['atr'][i] * volatility_factor
                        except Exception:
                            volatility_ok = False
                    else:
                        # Cálculo fallback de volatilidade
                        start_idx_range = max(0, i - lookback_candles)
                        recent_high = np.max(high_values[start_idx_range:i + 1])
                        recent_low = np.min(low_values[start_idx_range:i + 1])
                        recent_range = recent_high - recent_low
                        volatility_ok = (recent_range > 0 and 
                                        candle_size >= recent_range * volatility_factor)

                    if not (size_ok and volatility_ok and trend_strength >= 0.7):
                        continue

                    # DETECÇÃO DE SUPORTE
                    support_condition = (
                        i >= 2 and i < len(df) - 2 and
                        current_low < low_values[i - 1] and 
                        current_low < low_values[i - 2] and
                        current_low < low_values[i + 1] and 
                        current_low < low_values[i + 2]
                    )

                    if support_condition:
                        if require_confirmation and op_type in ['scalping', 'daytrade']:
                            confirm_candles = 2 if op_type == 'scalping' else 3
                            if i < len(df) - confirm_candles:
                                # Verificar próximos candles de confirmação
                                confirmation_ok = True
                                for j in range(1, confirm_candles + 1):
                                    if i + j >= len(close_values) or close_values[i + j] <= current_low:
                                        confirmation_ok = False
                                        break
                                
                                if not confirmation_ok:
                                    continue

                        fractals['supports'].append(current_low)
                        fractal_details['supports_info'].append({
                            'index': i, 
                            'price': current_low, 
                            'volume': current_vol,
                            'timestamp': df.index[i] if i < len(df.index) else pd.Timestamp.now(),
                            'operation_type': op_type,
                            'strength': self._calculate_fractal_strength_fast(df, i, 'support', params)
                        })
                        valid_fractals_count += 1

                    # DETECÇÃO DE RESISTÊNCIA
                    resistance_condition = (
                        i >= 2 and i < len(df) - 2 and
                        current_high > high_values[i - 1] and 
                        current_high > high_values[i - 2] and
                        current_high > high_values[i + 1] and 
                        current_high > high_values[i + 2]
                    )

                    if resistance_condition:
                        if require_confirmation and op_type in ['scalping', 'daytrade']:
                            confirm_candles = 2 if op_type == 'scalping' else 3
                            if i < len(df) - confirm_candles:
                                # Verificar próximos candles de confirmação
                                confirmation_ok = True
                                for j in range(1, confirm_candles + 1):
                                    if i + j >= len(close_values) or close_values[i + j] >= current_high:
                                        confirmation_ok = False
                                        break
                                
                                if not confirmation_ok:
                                    continue

                        fractals['resistances'].append(current_high)
                        fractal_details['resistances_info'].append({
                            'index': i, 
                            'price': current_high, 
                            'volume': current_vol,
                            'timestamp': df.index[i] if i < len(df.index) else pd.Timestamp.now(),
                            'operation_type': op_type,
                            'strength': self._calculate_fractal_strength_fast(df, i, 'resistance', params)
                        })
                        valid_fractals_count += 1

                except Exception as e:
                    logger.warning(f"Erro processando fractal no índice {i}: {e}")
                    continue

            logger.debug(f"Fractais crus encontrados: {valid_fractals_count}")

            # Clusterização de fractais (híbrida, sem duplicar processos)
            try:
                current_price = float(df['Close'].iloc[-1]) if 'Close' in df.columns and not df['Close'].empty else None

                cluster_params = {
                    'min_distance_pips': min_distance_pips,
                    'min_cluster_size': int(params.get('min_cluster_size', ap.get('min_cluster_size', 2))),
                    'tolerance_pct': float(params.get('tolerance_pct', ap.get('tolerance_pct', 0.002)))
                }
                timeframe_for_cluster = params.get('timeframe', params.get('base_tf', 'default'))
                precomputed = {
                    'atr': float(atr_val_float) if atr_val_float else 0.0,
                    'operation_type': operation_type
                }

                clustered_supports, clustered_resistances = cluster_levels_auto(
                    fractals.get('supports', []),
                    fractals.get('resistances', []),
                    current_price,
                    pair,
                    cluster_params,
                    df,
                    timeframe=timeframe_for_cluster,
                    use_volume=True,
                    precomputed=precomputed
                )

                # Híbrido: se cluster_levels_auto falhar, tenta _cluster_price_levels
                if not clustered_supports and fractals['supports']:
                    clustered_supports = self._cluster_price_levels(fractals['supports'], current_price, params)
                if not clustered_resistances and fractals['resistances']:
                    clustered_resistances = self._cluster_price_levels(fractals['resistances'], current_price, params)

                # Se algum cluster der resultado, substitui
                if clustered_supports:
                    fractals['supports'] = clustered_supports
                if clustered_resistances:
                    fractals['resistances'] = clustered_resistances

                # Remapeamento dos detalhes apenas se houver cluster válido
                def remap_details(details_list, clustered_levels):
                    if not clustered_levels:
                        return details_list
                    remapped = []
                    for d in details_list:
                        closest = min(clustered_levels, key=lambda lvl: abs(lvl - d['price']))
                        d_copy = d.copy()
                        d_copy['clustered_level'] = closest
                        remapped.append(d_copy)
                    return remapped

                fractal_details['supports_info'] = remap_details(fractal_details['supports_info'], fractals['supports'])
                fractal_details['resistances_info'] = remap_details(fractal_details['resistances_info'], fractals['resistances'])

                logger.debug(f"Fractais clusterizados (híbrido): {len(fractals['supports'])}S/{len(fractals['resistances'])}R")

            except Exception as e:
                logger.error(f"Erro na clusterização de fractais: {e}")
                # fallback: mantém fractais crus

            # RESULTADO FINAL
            result = {
                'fractals': fractals,
                'details': fractal_details,
                'metadata': {
                    'total_supports': len(fractals['supports']),
                    'total_resistances': len(fractals['resistances']),
                    'operation_type': op_type,
                    'timestamp': datetime.now(),
                    'params_used': ap
                }
            }

        except Exception as e:
            logger.error(f"Erro crítico em _find_valid_fractals: {e}", exc_info=True)
            return default_return

        # CACHE
        expiry_map = {
            'scalping': 300, 'daytrade': 900, 'swing': 3600, 'position': 21600
        }
        expiry = expiry_map.get(operation_type, 900)
        global_cache.set(cache_key, result, expiry=expiry)

        logger.info(f"Fractais encontrados: {len(result['fractals']['supports'])} suportes, {len(result['fractals']['resistances'])} resistências")
        return result

    def _calculate_fractal_strength_fast(
            self, df: pd.DataFrame, index: int, fractal_type: str, params: dict, clustered_level: float = None
        ) -> float:
        """
        Calcula força do fractal com:
        - Volume relativo (volume_score)
        - Tendência (trend_score, via EMAs se disponíveis)
        - Recência do fractal (recency_score)
        - Número de toques no nível (touch_score dinâmico)
        - Ajustado para considerar nível clusterizado (quando fornecido)

        Args:
            df (pd.DataFrame): DataFrame OHLCV.
            index (int): Índice do fractal no DataFrame.
            fractal_type (str): "support" ou "resistance".
            params (dict): parâmetros de operação (inclui operation_type, ema_fast, ema_slow, etc.).
            clustered_level (float, opcional): nível clusterizado, se já disponível.

        Returns:
            float: força do fractal (0.5 a 5.0).
        """
        try:
            # Seleção de pesos por tipo de operação
            op_type = params.get('operation_type', 'daytrade')
            weight_config = {
                'scalping': {'volume': 0.4, 'trend': 0.3, 'touch': 0.2, 'recency': 0.1},
                'daytrade': {'volume': 0.35, 'trend': 0.25, 'touch': 0.25, 'recency': 0.15},
                'swing': {'volume': 0.3, 'trend': 0.2, 'touch': 0.3, 'recency': 0.2},
                'position': {'volume': 0.25, 'trend': 0.15, 'touch': 0.35, 'recency': 0.25},
            }.get(op_type, {'volume': 0.35, 'trend': 0.25, 'touch': 0.25, 'recency': 0.15})

            # ================================
            # 1. Volume ratio
            # ================================
            try:
                vol_ma_20 = df['Volume'].rolling(20, min_periods=1).mean().fillna(1)
                base_vol = vol_ma_20.iloc[index] if index < len(df) else 1.0

                if base_vol > 0:
                    volume_ratio = min(df['Volume'].iloc[index] / base_vol, 3.0)
                else:
                    volume_ratio = 1.0
            except Exception:
                volume_ratio = 1.0

            # Garante que não fique negativo
            raw_volume_score = (volume_ratio - 1) * weight_config['volume']
            volume_score = max(0.0, raw_volume_score)

            # ================================
            # 2. Tendência (se disponível)
            # ================================
            trend_score = 0.0
            if 'ema_fast' in params and 'ema_slow' in params:
                try:
                    if index < len(params['ema_fast']) and index < len(params['ema_slow']):
                        if params['ema_fast'][index] > params['ema_slow'][index]:
                            trend_score = 1.0 * weight_config['trend']
                        else:
                            trend_score = 0.0
                    else:
                        trend_score = 0.5 * weight_config['trend']
                except Exception:
                    trend_score = 0.5 * weight_config['trend']
            else:
                trend_score = 0.5 * weight_config['trend']

            # ================================
            # 3. Recency score
            # ================================
            if len(df) > 1:
                position = index / (len(df) - 1)
                recency_score = (1.0 - position) * weight_config['recency']
            else:
                recency_score = 0.0

            # ================================
            # 4. Touch score dinâmico (com híbrido)
            # ================================
            # Se temos clustered_level, usamos ele; senão, usamos o preço cru
            if clustered_level is not None:
                price_level = clustered_level
            else:
                price_level = df['Low'].iloc[index] if fractal_type == 'support' else df['High'].iloc[index]

            # Tolerância dinâmica:
            # - Base: 0.05% do preço
            # - Se ATR estiver disponível, usamos fração dele
            tolerance = price_level * 0.0005
            if 'atr' in params and params['atr'] is not None:
                try:
                    if index < len(params['atr']):
                        tolerance = max(tolerance, float(params['atr'][index]) * 0.1)
                except Exception:
                    pass

            touches = 0
            for i in range(index + 1, len(df)):
                if fractal_type == 'support':
                    if df['Low'].iloc[i] <= price_level + tolerance:
                        touches += 1
                elif fractal_type == 'resistance':
                    if df['High'].iloc[i] >= price_level - tolerance:
                        touches += 1

            # normaliza: até 5 toques contam
            touch_factor = min(touches, 5) / 5.0
            touch_score = touch_factor * weight_config['touch']

            # ================================
            # Força final
            # ================================
            strength = 1.0 + volume_score + trend_score + recency_score + touch_score
            return round(min(max(strength, 0.5), 5.0), 2)

        except Exception as e:
            print(f"⚠️ Erro cálculo força fractal: {e}")
            return 1.0  # Fallback seguro
            
    def _get_adaptive_ema_span(self, timeframe: str, operation_type: str) -> int:
        """Retorna o EMA span ideal baseado no timeframe e tipo de operação"""
        logger.debug(f"Obtendo EMA span para timeframe: {timeframe}, operation: {operation_type}")
        
        timeframe_settings = {
            '30s': {'scalping': 5, 'daytrade': 8, 'swing': 12},
            '1min': {'scalping': 8, 'daytrade': 12, 'swing': 20},
            '2min': {'scalping': 10, 'daytrade': 15, 'swing': 25},
            '5min': {'scalping': 12, 'daytrade': 20, 'swing': 30},
            '10min': {'scalping': 15, 'daytrade': 25, 'swing': 40},
            '15min': {'scalping': 20, 'daytrade': 30, 'swing': 50},
            '30min': {'scalping': 25, 'daytrade': 40, 'swing': 60},
            '1H': {'scalping': 30, 'daytrade': 50, 'swing': 80},
            '4H': {'scalping': 40, 'daytrade': 60, 'swing': 100},
            '1D': {'scalping': 50, 'daytrade': 70, 'swing': 120}
        }

        default_settings = {'scalping': 15, 'daytrade': 25, 'swing': 40}
        
        if timeframe not in timeframe_settings:
            result = default_settings.get(operation_type, 20)
            logger.debug(f"Timeframe {timeframe} não encontrado, usando default: {result}")
            return result
        
        result = timeframe_settings[timeframe].get(operation_type, 20)
        logger.debug(f"EMA span definido: {result}")
        return result

    def _calculate_fibonacci_extensions(
        self,
        df: pd.DataFrame,
        supports: List[float],
        resistances: List[float],
        timeframe: str = "15min",
        operation_type: str = "daytrade",
        pair: str = ""
    ) -> Dict:
        """
        Calcula níveis de Fibonacci (retracements + extensions) de forma robusta.
            Retorna sempre um dict com as chaves:
            - retracements: dict (pode estar vazio)
            - extensions: dict (pode estar vazio)
            - swing_high: float | None
            - swing_low: float | None
            - trend: "up"|"down"|"sideways"
            - ema_value: float | None
            - ema_span_used: int
            - timeframe: str
            """
        logger.debug(f"Calculando Fibonacci para {pair} ({operation_type}, {timeframe})")
        
        # --------- CACHE CHECK ---------
        try:
            params = {"timeframe": timeframe, "supports": supports, "resistances": resistances}
            cache_key = make_cache_key(pair, f"fib_{operation_type}", params, cache_namespace="sr")
            cached = global_cache.get(cache_key)
            if cached is not None:
                logger.debug("Retornando Fibonacci do cache")
                return cached
        except Exception as e:
            logger.warning(f"Erro no cache check do Fibonacci: {e}")
        # -------------------------------

        empty_result = {
            'retracements': {},
            'extensions': {},
            'swing_high': None,
            'swing_low': None,
            'trend': 'sideways',
            'ema_value': None,
            'ema_span_used': None,
            'timeframe': timeframe
        }

        # Validações básicas do df
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            logger.warning("DataFrame vazio ou inválido para Fibonacci")
            return empty_result

        # Verificar colunas essenciais
        for col in ['High', 'Low', 'Close']:
            if col not in df.columns:
                logger.warning(f"Coluna obrigatória ausente: {col}")
                return empty_result

        # Determina ema span
        try:
            ema_span = int(self._get_adaptive_ema_span(timeframe, operation_type) or 20)
            if ema_span <= 0:
                ema_span = 20
            logger.debug(f"EMA span utilizado: {ema_span}")
        except Exception as e:
            logger.warning(f"Erro ao determinar EMA span: {e}")
            ema_span = 20

        # Serie de closes
        close_series = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill').fillna(method='bfill')
        if close_series.isna().all():
            logger.warning("Série Close contém apenas NaN")
            return empty_result

        try:
            ema_val = float(close_series.ewm(span=ema_span, adjust=False).mean().iloc[-1])
            current_price = float(close_series.iloc[-1])
            logger.debug(f"Preço atual: {current_price}, EMA: {ema_val}")
        except Exception as e:
            logger.error(f"Erro ao calcular EMA/preço: {e}")
            return empty_result

        # Determina tendência básica
        if current_price > ema_val:
            trend = "up"
        elif current_price < ema_val:
            trend = "down"
        else:
            trend = "sideways"
        
        logger.debug(f"Tendência detectada: {trend}")

        # Função auxiliar para limpar níveis
        def _clean_levels(levels, price_ref: float, rel_tol: float = 1e-5) -> List[float]:
            if not levels:
                return []

            try:
                iterator = list(levels)
            except Exception:
                return []

            cleaned = []
            for x in iterator:
                try:
                    if x is None:
                        continue
                    xv = float(x)
                    if np.isfinite(xv):
                        cleaned.append(xv)
                except Exception:
                    continue

            if not cleaned:
                return []

            cleaned_sorted = sorted(cleaned)

            # remover duplicatas muito próximas
            tol = max(abs(price_ref) * rel_tol, 1e-12)
            merged = []
            current = cleaned_sorted[0]
            for v in cleaned_sorted[1:]:
                if abs(v - current) <= tol:
                    current = (current + v) / 2.0
                else:
                    merged.append(current)
                    current = v
            merged.append(current)
            return merged

        supports_clean = _clean_levels(supports, current_price)
        resistances_clean = _clean_levels(resistances, current_price)
        
        logger.debug(f"Suportes limpos: {len(supports_clean)}, Resistências limpas: {len(resistances_clean)}")

        swing_high: Optional[float] = None
        swing_low: Optional[float] = None

        # Tentativa: usar fractais clusterizados alinhados con a tendência
        try:
            if trend == "up" and supports_clean:
                last_n = supports_clean[-min(5, len(supports_clean)):]
                swing_low = float(min(last_n))

                valid_highs = [r for r in resistances_clean if r > swing_low]
                if valid_highs:
                    highs_above_price = [r for r in valid_highs if r > current_price]
                    candidate_highs = highs_above_price if highs_above_price else valid_highs
                    swing_high = float(max(candidate_highs)) if candidate_highs else None
                    logger.debug(f"Swing UP: low={swing_low}, high={swing_high}")

            elif trend == "down" and resistances_clean:
                last_n = resistances_clean[-min(5, len(resistances_clean)):]
                swing_high = float(max(last_n))

                valid_lows = [s for s in supports_clean if s < swing_high]
                if valid_lows:
                    lows_below_price = [s for s in valid_lows if s < current_price]
                    candidate_lows = lows_below_price if lows_below_price else valid_lows
                    swing_low = float(min(candidate_lows)) if candidate_lows else None
                    logger.debug(f"Swing DOWN: high={swing_high}, low={swing_low}")

            else:
                if supports_clean and resistances_clean:
                    swing_low = float(min(supports_clean[-min(5, len(supports_clean)):]))
                    swing_high = float(max(resistances_clean[-min(5, len(resistances_clean)):]))
                    logger.debug(f"Swing SIDEWAYS: low={swing_low}, high={swing_high}")
                    
        except Exception as e:
            logger.warning(f"Erro ao derivar swings a partir de fractais: {e}")

        # Fallback adaptativo: rolling window
        if swing_high is None or swing_low is None:
            try:
                window = min(max(10, len(df) // 5), len(df))
                window = max(window, 3)
                swing_high_fallback = float(df['High'].rolling(window=window, min_periods=1).max().iloc[-1])
                swing_low_fallback = float(df['Low'].rolling(window=window, min_periods=1).min().iloc[-1])
                swing_high = swing_high or swing_high_fallback
                swing_low = swing_low or swing_low_fallback
                logger.debug(f"Fallback rolling: low={swing_low}, high={swing_high}")
            except Exception as e:
                logger.error(f"Fallback rolando janelas falhou: {e}")
                return empty_result

        # Validações finais dos swings
        try:
            if swing_high is None or swing_low is None:
                logger.warning("Swings inválidos após fallback")
                return empty_result

            swing_high = float(swing_high)
            swing_low = float(swing_low)

            if swing_high <= swing_low:
                if np.isclose(swing_high, swing_low, rtol=1e-8, atol=1e-12):
                    logger.warning("Swing high ≈ swing low (praticamente igual)")
                    return empty_result
                else:
                    swing_high, swing_low = max(swing_high, swing_low), min(swing_high, swing_low)
                    logger.debug(f"Swings invertidos corrigidos: low={swing_low}, high={swing_high}")

            diff = swing_high - swing_low
            if diff <= 0 or not np.isfinite(diff):
                logger.warning("Diferença inválida entre swings")
                return empty_result
                
            logger.debug(f"Diferença entre swings: {diff}")
                
        except Exception as e:
            logger.error(f"Erro na validação final dos swings: {e}")
            return empty_result

        # Cálculo dos níveis Fibonacci
        try:
            retracements = {}
            extensions = {}

            fibs = {
                '236': 0.236,
                '382': 0.382,
                '500': 0.5,
                '618': 0.618,
                '786': 0.786
            }

            if trend == "up":
                retracements = {k: swing_high - diff * v for k, v in fibs.items()}
                extensions = {
                    '1000': swing_high,
                    '1272': swing_high + diff * 0.272,
                    '1414': swing_high + diff * 0.414,
                    '1618': swing_high + diff * 0.618,
                    '2000': swing_high + diff * 1.0,
                    '2618': swing_high + diff * 1.618,
                    '3618': swing_high + diff * 2.618
                }
            elif trend == "down":
                retracements = {k: swing_low + diff * v for k, v in fibs.items()}
                extensions = {
                    '1000': swing_low,
                    '1272': swing_low - diff * 0.272,
                    '1414': swing_low - diff * 0.414,
                    '1618': swing_low - diff * 0.618,
                    '2000': swing_low - diff * 1.0,
                    '2618': swing_low - diff * 1.618,
                    '3618': swing_low - diff * 2.618
                }
            else:
                retracements = {k: swing_high - diff * v for k, v in fibs.items()}
                extensions = {}

            logger.debug(f"Fibonacci calculado: {len(retracements)} retracements, {len(extensions)} extensions")

        except Exception as e:
            logger.error(f"Erro ao calcular níveis Fibonacci: {e}")
            return empty_result

        # Arredondamento consistente
        def _round_dict(d: dict) -> dict:
            return {k: round(float(v), 8) for k, v in d.items()}

        result = {
            'retracements': _round_dict(retracements),
            'extensions': _round_dict(extensions),
            'swing_high': round(float(swing_high), 8),
            'swing_low': round(float(swing_low), 8),
            'trend': trend,
            'ema_value': round(float(ema_val), 8),
            'ema_span_used': int(ema_span),
            'timeframe': timeframe
        }

        # Cache
        try:
            expiry_map = {
                'scalping': 300,
                'daytrade': 900,
                'swing': 3600,
                'position': 21600
            }
            expiry = expiry_map.get(operation_type, 900)
            global_cache.set(cache_key, result, expiry=expiry)
            logger.debug("Fibonacci salvo no cache")
        except Exception as e:
            logger.warning(f"Erro ao salvar Fibonacci no cache: {e}")

        logger.info(f"Fibonacci concluído: {len(result['retracements'])}R {len(result['extensions'])}E")
        return result
            
    def _calculate_volume_stats(self, df: pd.DataFrame) -> dict:
        """Calcula estatísticas de volume para uso nos sinais"""
        logger.debug("Calculando estatísticas de volume")
        
        if len(df) < 20:
            logger.warning("Dados insuficientes para estatísticas de volume")
            return {'current': 0, 'mean': 1, 'change': 0}
        
        try:
            current_volume = df['Volume'].iloc[-1]
            mean_volume = df['Volume'].rolling(20).mean().iloc[-1]
            
            if mean_volume <= 0:
                mean_volume = 1
                
            volume_change = (current_volume / mean_volume - 1) * 100 if mean_volume > 0 else 0
            
            result = {
                'current': current_volume,
                'mean': mean_volume,
                'change': volume_change
            }
            
            logger.debug(f"Estatísticas de volume: {current_volume} atual, {mean_volume:.1f} média, {volume_change:.1f}% variação")
            return result
            
        except Exception as e:
            logger.error(f"Erro no cálculo de estatísticas de volume: {e}")
            return {'current': 0, 'mean': 1, 'change': 0}
            
    def analyze(self, pair: str, operation_type: str) -> Optional[Dict]:
        """Executa análise completa de S/R"""
        
        logger.info(f"Iniciando análise S/R para {pair} ({operation_type})")
        
        # ==============================
        # Validação inicial
        # ==============================
        if not pair or not operation_type:
            logger.warning("Par ou tipo de operação não especificado")
            return None

        params = self._get_optimal_params(operation_type)
        if not params:
            logger.error(f"Parámetros não encontrados para {operation_type}")
            return None

        # ==============================
        # Cache Key Global (multi-timeframes)
        # ==============================
        timeframes = [
            params.get("base_tf"),
            params.get("aux_tf"),
            # futuramente: params.get("signal_tf")  # quando o 3º timeframe for adicionado
        ]
        cached = global_cache.get_for(
            pair, operation_type, params,
            timeframes=timeframes,
            current_price=None
        )
        if cached is not None:
            logger.debug(f"Retornando análise do cache para {pair}")
            return cached

        try:
            logger.debug(f"Buscando dados multi-timeframe para {pair}")
            # ==============================
            # Dados multi-timeframe
            # ==============================
            data = self._fetch_multi_timeframe_data(pair, params)
            if (not data or 'base' not in data or 'aux' not in data or
                    data['base'].empty or data['aux'].empty):
                logger.warning(f"Dados insuficientes (base/aux) para {pair} {operation_type}")
                return None

            df_base = data['base']
            df_aux = data['aux']
            df_pivot = data.get('pivot', df_base)

            current_price = df_base['Close'].iloc[-1]
            logger.debug(f"Preço atual: {current_price}")

            # ==============================
            # Pré-calcula ATRs (fractal e cluster)
            # ==============================
            atr_dict = {}
            atr_periods = {
                "fractal": params.get("fractal_atr_period", 14),
                "cluster": params.get("cluster_atr_period", 50),
            }
            
            for key, period in atr_periods.items():
                try:
                    atr_series = talib.ATR(
                        df_base['High'], df_base['Low'], df_base['Close'],
                        timeperiod=period
                    )
                    atr_dict[key] = {
                        "series": atr_series.fillna(0).values,
                        "last": float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0
                    }
                    logger.debug(f"ATR {key} calculado: {atr_dict[key]['last']}")
                except Exception as e:
                    logger.warning(f"Erro no cálculo do ATR {key}: {e}")
                    atr_dict[key] = {"series": np.zeros(len(df_base)), "last": 0.0}

            # ==============================
            # 1. Volume Profile (KDE)
            # ==============================
            try:
                logger.debug("Calculando Volume Profile...")
                volume_nodes = self._calculate_volume_nodes(df_base, pair=pair) or {}
                if volume_nodes.get('poc'):
                    logger.debug(f"POC encontrado: {volume_nodes['poc']}")
            except Exception as e:
                logger.error(f"Erro no cálculo de Volume Profile: {e}")
                volume_nodes = {}

            # ==============================
            # 2. Liquidity Zones
            # ==============================
            try:
                logger.debug("Buscando Liquidity Zones...")
                liquidity = self._find_liquidity_zones(df_base, params, pair=pair, operation_type=operation_type) or {}
                if liquidity.get('zones'):
                    logger.debug(f"{len(liquidity['zones'])} zonas de liquidez encontradas")
            except Exception as e:
                logger.error(f"Erro no cálculo de Liquidity Zones: {e}")
                liquidity = {}

            # ==============================
            # 3. Estatísticas de volume
            # ==============================
            volume_stats = self._calculate_volume_stats(df_base)
            logger.debug(f"Volume atual: {volume_stats['current']} (Média: {volume_stats['mean']})")

            # ==============================
            # Fractal params e indicadores
            # ==============================
            fractal_params = {
                'operation_type': operation_type,
                'volume_factor': params.get('volume_factor', 1.5),
                'atr': atr_dict["fractal"]["series"]
            }

            df_base = df_base.fillna(method='ffill').fillna(method='bfill')

            if len(df_base) >= 10:
                try:
                    logger.debug("Calculando indicadores técnicos...")
                    fractal_params.update({
                        'rsi': talib.RSI(df_base['Close'], timeperiod=14).fillna(50).values,
                        'ema_fast': df_base['Close'].ewm(span=9).mean().fillna(method='bfill').values,
                        'ema_slow': df_base['Close'].ewm(span=21).mean().fillna(method='bfill').values,
                        'adx': talib.ADX(df_base['High'], df_base['Low'], df_base['Close'], timeperiod=14).fillna(20).values,
                        'vol_ma': df_base['Volume'].rolling(params.get('pivot_lookback', 20)).mean().fillna(0).values
                    })
                except Exception as e:
                    logger.warning(f"Erro no cálculo de indicadores técnicos: {e}")

            # ==============================
            # 4. Detectar fractais
            # ==============================
            try:
                logger.debug("Detectando fractais no timeframe base...")
                base_fractals_result = self._find_valid_fractals(df_base, fractal_params, pair=pair, operation_type=operation_type) or {}
                
                logger.debug("Detectando fractais no timeframe auxiliar...")
                aux_fractals_result = self._find_valid_fractals(df_aux, fractal_params, pair=pair, operation_type=operation_type) or {}
                
                base_supports = base_fractals_result.get('fractals', {}).get('supports', [])
                base_resistances = base_fractals_result.get('fractals', {}).get('resistances', [])
                aux_supports = aux_fractals_result.get('fractals', {}).get('supports', [])
                aux_resistances = aux_fractals_result.get('fractals', {}).get('resistances', [])
                
                logger.debug(f"Fractais encontrados - Base: {len(base_supports)}S/{len(base_resistances)}R, Aux: {len(aux_supports)}S/{len(aux_resistances)}R")
                
            except Exception as e:
                logger.error(f"Erro na detecção de fractais: {e}", exc_info=True)
                return None

            if (not base_fractals_result or 'fractals' not in base_fractals_result or
                    not aux_fractals_result or 'fractals' not in aux_fractals_result):
                logger.warning(f"Fractais inválidos para {pair}")
                return None

            base_fractals = base_fractals_result['fractals']
            aux_fractals = aux_fractals_result['fractals']

            all_supports = base_fractals.get('supports', []) + aux_fractals.get('supports', [])
            all_resistances = base_fractals.get('resistances', []) + aux_fractals.get('resistances', [])

            fractal_details = base_fractals_result.get('details', {})
            fractal_metadata = base_fractals_result.get('metadata', {})

            # ==============================
            # 5. Clusteriza níveis
            # ==============================
            try:
                logger.debug(f"Clusterizando {len(all_supports)} suportes e {len(all_resistances)} resistências...")
                clustered_supports, clustered_resistances = cluster_levels_auto(
                    all_supports=all_supports,
                    all_resistances=all_resistances,
                    current_price=current_price,
                    pair=pair,
                    params=params,
                    df=df_base,
                    timeframe=params.get("base_tf", "default"),
                    use_volume=True,
                    precomputed={
                        'atr': atr_dict["cluster"]["last"],
                        'operation_type': operation_type
                    }
                )
                logger.debug(f"Clusterização concluída: {len(clustered_supports)}S/{len(clustered_resistances)}R")
                
            except Exception as e:
                logger.error(f"Erro na clusterização de níveis: {e}", exc_info=True)
                clustered_supports, clustered_resistances = [], []

            # ==============================
            # Fibonacci dinâmico
            # ==============================
            try:
                logger.debug("Calculando níveis Fibonacci...")
                fib_levels = self._calculate_fibonacci_extensions(
                    df_base,
                    clustered_supports,
                    clustered_resistances,
                    timeframe=params.get('base_tf', '15min'),
                    operation_type=operation_type
                ) or {}
                if fib_levels.get('retracements'):
                    logger.debug(f"Fibonacci calculado: {len(fib_levels['retracements'])} retracements")
            except Exception as e:
                logger.error(f"Erro no cálculo Fibonacci: {e}")
                fib_levels = {}

            # ==============================
            # 7. Outros níveis
            # ==============================
            ema = df_base['MA'].iloc[-1] if 'MA' in df_base.columns and not df_base['MA'].isna().all() else None
            if ema:
                logger.debug(f"EMA: {ema}")

            try:
                pivot_points = self._calculate_pivot_points(df_pivot) or {}
                if pivot_points:
                    logger.debug(f"Pivot Points calculados: PP={pivot_points.get('pivot')}")
            except Exception as e:
                logger.warning(f"Erro no cálculo de Pivot Points: {e}")
                pivot_points = {}

            try:
                vwap_dynamic = self._calculate_adaptive_vwap(
                    df_base,
                    operation_type=operation_type,
                    reset_mode="daily",
                    use_tick_volume=True
                ) or {}
                if vwap_dynamic.get('vwap'):
                    logger.debug(f"VWAP: {vwap_dynamic['vwap']} (Tendência: {vwap_dynamic.get('trend')})")
            except Exception as e:
                logger.warning(f"Erro no cálculo do VWAP: {e}")
                vwap_dynamic = {}

            # ==============================
            # 8. Monta níveis finais
            # ==============================
            min_touches = params.get('min_touches', 2)
            def last_n(lst, n):
                return lst[-n:] if lst and len(lst) >= n else lst
            
            levels = {
                'supports': clustered_supports[-min_touches:] if clustered_supports else [],
                'resistances': clustered_resistances[-min_touches:] if clustered_resistances else [],
                'fractal_details': fractal_details,
                'fractal_metadata': fractal_metadata,
                'poc': volume_nodes.get('poc') if isinstance(volume_nodes, dict) else None,
                'value_area': volume_nodes.get('value_area') if isinstance(volume_nodes, dict) else None,
                'liquidity_zones': liquidity,
                'fibonacci': fib_levels,
                'ema': ema,
                'ema_window': params.get('ema_window'),
                'pivot_points': pivot_points,
                'vwap_dynamic': vwap_dynamic,
                'volume_stats': volume_stats,
                'current_price': current_price,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'timeframe': params.get('base_tf'),
                'operation_type': operation_type,
                'support': clustered_supports[-1] if clustered_supports else None,
                'resistance': clustered_resistances[-1] if clustered_resistances else None,
                'secondary_supports': last_n(clustered_supports, 3) if clustered_supports else [],
                'secondary_resistances': last_n(clustered_resistances, 3) if clustered_resistances else [],
            }

            # ==============================
            # Atualiza cache global (multi-timeframes)
            # ==============================
            expiry_map = {
                'scalping': 300,      # 5 minutos
                'daytrade': 900,      # 15 minutos
                'swing': 3600,        # 1 hora
                'position': 21600     # 6 horas
            }
            expiry = expiry_map.get(operation_type, 900)

            global_cache.set_for(
                pair, operation_type, params, levels,
                timeframes=timeframes,
                expiry=expiry,
                last_price=current_price,
                price_tolerance=params.get("tolerance_pct", 0.005)
            )

            logger.info(f"Análise concluída para {pair}: {len(levels['supports'])} suportes, "
                       f"{len(levels['resistances'])} resistências, Preço: {current_price}")

            return levels

        except Exception as e:
            logger.critical(f"Erro crítico na análise de {pair}/{operation_type}", exc_info=True)
            return None

    def _calculate_confluence_confidence(self, level: float, level_type: str, levels: dict) -> float:
        """
        Calcula confiança de um nível de suporte/resistência baseado em múltiplas confluências.
        Retorna score contínuo entre 0.0 and 1.0.
        """

        logger.debug(f"Calculando confiança para nível {level_type} em {level}")
        
        current_price = levels['current_price']
        score = 0.2  # Base mínima para qualquer nível detectado

        # --- Pesos relativos (podem ser calibrados com backtest) ---
        weights = {
            "poc": 0.2,
            "value_area": 0.15,
            "fibonacci": 0.2,
            "pivot": 0.1,
            "liquidity": 0.25,
        }

        # 1. Proximidade com POC
        poc = levels.get('poc')
        if poc:
            distance = abs(level - poc) / current_price
            if distance < 0.005:  # até 0.5%
                score += weights["poc"] * max(0, (0.005 - distance) / 0.005)

        # 2. Value Area
        va = levels.get('value_area')
        if va:
            if va[0] <= level <= va[1]:
                score += weights["value_area"]

        # 3. Fibonacci
        if levels.get('fibonacci'):
            fibs = list(levels['fibonacci'].get('retracements', {}).values()) + \
                   list(levels['fibonacci'].get('extensions', {}).values())
            for f in fibs:
                distance = abs(level - f) / current_price
                if distance < 0.003:  # até 0.3%
                    score += weights["fibonacci"] * max(0, (0.003 - distance) / 0.003)
                    break  # basta um fib próximo

        # 4. Pivot Points
        pp = levels.get('pivot_points', {})
        if pp and level in pp.values():
            score += weights["pivot"]

        # 5. Liquidity Zones
        liquidity = levels.get('liquidity_zones', {})
        # Caso clássico
        if isinstance(liquidity, dict) and ('ob_up' in liquidity or 'ob_down' in liquidity):
            ob_level = liquidity.get('ob_up') if level_type == 'support' else liquidity.get('ob_down')
            if ob_level:
                distance = abs(level - ob_level) / current_price
                if distance < 0.002:
                    score += weights["liquidity"] * max(0, (0.002 - distance) / 0.002)
        # Caso moderno (lista de zonas com price)
        zones = liquidity.get('zones') if isinstance(liquidity, dict) else None
        if zones:
            zone_prices = [z['price'] for z in zones if 'price' in z]
            if zone_prices:
                closest = min(zone_prices, key=lambda p: abs(level - p))
                distance = abs(level - closest) / current_price
                if distance < 0.002:
                    score += weights["liquidity"] * max(0, (0.002 - distance) / 0.002)

        final_score = round(min(score, 1.0), 3)
        logger.debug(f"Confiança calculada para {level}: {final_score}")
        return final_score

    def generate_trading_signals(self, levels: Dict) -> Dict:
        """Gera sinais de trading baseados nos níveis identificados (S/R, Volume Profile, VWAP, Pivots)"""
        
        if not levels:
            logger.warning("Níveis vazios - nenhum sinal gerado")
            return {}
            
        logger.info(f"Gerando sinais para {levels.get('operation_type')} - Preço: {levels.get('current_price')}")
        
        signals = {}
        current_price = levels['current_price']
        op_type = levels['operation_type']

        # ---------------------------
        # 1. Volume stats
        # ---------------------------
        if 'volume_stats' in levels:
            volume_change = (levels['volume_stats']['current'] / levels['volume_stats']['mean'] - 1) * 100
        else:
            volume_change = 0

        # ---------------------------
        # 2. Suportes / Resistências clusterizados
        # ---------------------------
        valid_supports = [s for s in levels['supports'] if s < current_price]
        if valid_supports:
            closest_support = max(valid_supports)
            signals['potential_call'] = {
                'entry': closest_support * (1.0003 if op_type == 'scalping' else 1.0008),
                'stop_loss': closest_support * 0.9995,
                'take_profit': [
                    current_price + (current_price - closest_support) * 0.618,
                    current_price + (current_price - closest_support) * 1.0
                ],
                'confidence': min(0.9, len(valid_supports) / 3),
                'conditions': [
                    f"Volume acima da média no teste",
                    f"Confirmação de candle na área de valor",
                    f"EMA{levels['ema_window']} ascendente"
                ]
            }

        valid_resistances = [r for r in levels['resistances'] if r > current_price]
        if valid_resistances:
            closest_resistance = min(valid_resistances)
            signals['potential_put'] = {
                'entry': closest_resistance * (0.9997 if op_type == 'scalping' else 0.9992),
                'stop_loss': closest_resistance * 1.0005,
                'take_profit': [
                    current_price - (closest_resistance - current_price) * 0.618,
                    current_price - (current_price - current_price) * 1.0
                ],
                'confidence': min(0.9, len(valid_resistances) / 3),
                'conditions': [
                    f"Volume acima da média no teste",
                    f"Rejeição na área de valor",
                    f"EMA{levels['ema_window']} descendente"
                ]
            }

        # ---------------------------
        # 3. Confirmação extra via Volume Profile
        # ---------------------------
        if levels.get('clusters'):  # só existe no modo smooth (crypto)
            cluster_levels = levels['clusters']
            signals['volume_profile_clusters'] = {
                'clusters': cluster_levels,
                'note': "Níveis ocultos revelados via suavização (institucional clusters em crypto)"
            }

        # ---------------------------
        # 4. Pivot Points
        # ---------------------------
        if levels.get('pivot_points'):
            pp = levels['pivot_points']
            signals['pivot_rules'] = {
                'call_above_pivot': {
                    'level': pp['pivot'],
                    'entry': pp['pivot'] * (1.0002 if op_type == 'scalping' else 1.0005),
                    'stop_loss': pp['s1'],
                    'take_profit': [pp['r1'], pp['r2']],
                    'confidence': 0.7,
                    'conditions': [
                        "Preço acima do pivot com volume crescente",
                        "Confirmação de fechamento acima do pivot"
                    ]
                },
                'put_below_pivot': {
                    'level': pp['pivot'],
                    'entry': pp['pivot'] * (0.9998 if op_type == 'scalping' else 0.9995),
                    'stop_loss': pp['r1'],
                    'take_profit': [pp['s1'], pp['s2']],
                    'confidence': 0.7,
                    'conditions': [
                        "Preço abaixo do pivot com volume crescente",
                        "Confirmação de fechamento abaixo do pivot"
                    ]
                }
            }

        # ---------------------------
        # 5. VWAP Dinâmico
        # ---------------------------
        vwap_data = levels.get('vwap_dynamic')
        if vwap_data and 'trend' in vwap_data:
            if vwap_data['trend'] == "up":
                signals['vwap_trend'] = {
                    'direction': 'long',
                    'entry': vwap_data['bands']['lower1'],
                    'stop_loss': vwap_data['bands']['lower2'],
                    'confidence': 0.8 if vwap_data.get('slope', 0) > 0 else 0.6
                }
            elif vwap_data['trend'] == "down":
                signals['vwap_trend'] = {
                    'direction': 'short',
                    'entry': vwap_data['bands']['upper1'],
                    'stop_loss': vwap_data['bands']['upper2'],
                    'confidence': 0.8 if vwap_data.get('slope', 0) < 0 else 0.6
                }

        # ---------------------------
        # 6. VWAP Pivots (fallback)
        # ---------------------------
        vwap = levels.get('vwap_pivots') or levels.get('vwap_dynamic')
        if vwap and ('vwap_pivot' in vwap or 'bands' in vwap):
            try:
                pivot_val = vwap.get('vwap_pivot') or vwap['bands'].get('mid', vwap['vwap'])
                signals['vwap_rules'] = {
                    'call_above_vwap': {
                        'level': pivot_val,
                        'entry': pivot_val * 1.0002,
                        'stop_loss': vwap.get('vwap_s1') or vwap['bands']['lower1'],
                        'take_profit': [
                            vwap.get('vwap_r1') or vwap['bands']['upper1'],
                            vwap.get('vwap_r2') or vwap['bands']['upper2']
                        ],
                        'confidence': 0.8,
                        'conditions': [
                            "Volume acima da média no teste do VWAP",
                            "Preço sustentado acima do VWAP"
                        ]
                    },
                    'put_below_vwap': {
                        'level': pivot_val,
                        'entry': pivot_val * 0.9998,
                        'stop_loss': vwap.get('vwap_r1') or vwap['bands']['upper1'],
                        'take_profit': [
                            vwap.get('vwap_s1') or vwap['bands']['lower1'],
                            vwap.get('vwap_s2') or vwap['bands']['lower2']
                        ],
                        'confidence': 0.8,
                        'conditions': [
                            "Volume acima da média no teste do VWAP",
                            "Preço sustentado abaixo do VWAP"
                        ]
                    }
                }
            except Exception as e:
            logger.error(f"Erro no cálculo de regras VWAP: {e}")

        # ---------------------------
        # 7. Metadados
        # ---------------------------
        signals['metadata'] = {
            'poc': levels.get('poc'),
            'value_area': levels.get('value_area'),
            'clusters': levels.get('clusters', []),
            'liquidity_zones': levels.get('liquidity_zones'),
            'fibonacci_levels': levels.get('fibonacci'),
            'pivot_points': levels.get('pivot_points'),
            'vwap_pivots': levels.get('vwap_pivots'),
            'timeframe': levels['timeframe'],
            'analysis_time': levels['timestamp']
        }

        logger.info(f"Sinais gerados: {len(signals)} entradas potenciais")
        return signals
