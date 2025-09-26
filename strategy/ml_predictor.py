# strategy/ml_predictor.py
# Função: Previsão de direção com modelo de machine learning treinado.
# Enriquecido para usar força dos padrões de candlestick (pattern_strength).
# Predictor otimizado e compatível para modelos de ML de trading, com todos os principais indicadores e padrões.

import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Imports do seu projeto
from strategy.ml_utils import add_indicators
from strategy.candlestick_patterns import detect_candlestick_patterns, get_pattern_strength
from strategy.indicator_globe import TechnicalIndicators
from data.google_drive_client import download_file, get_folder_id_for_file

from data.fundamental_data import get_cot_feature, get_macro_feature, get_sentiment_feature
from utils.features_extra import calc_obv, calc_spread
from utils.aggregation import resample_candles

class MLPredictor:
    """Predictor otimizado para modelos de trading com cache, validação e download do Google Drive."""

    def __init__(self, model_dir: str = "models", min_candles: int = 100):
        self.model_dir = model_dir
        self.min_candles = max(min_candles, 50)
        self._init_cache()
        self.TF_MAPPING = {
            '1min': 'm1', '5min': 'm5', '15min': 'm15',
            '30min': 'm30', '1h': 'h1', '4h': 'h4',
            's1': 's1', '1m': 'm1', '5m': 'm5'
        }

    def _init_cache(self):
        self.model_cache = {}
        self.last_used = {}
        self.cache_expiry = timedelta(hours=1)

    def _normalize_timeframe(self, timeframe: str) -> str:
        tf = timeframe.lower().strip()
        return self.TF_MAPPING.get(tf, tf)

    def _ensure_model_local(self, symbol: str, timeframe: str) -> str:
        """Garante que o modelo está salvo localmente, baixando do Google Drive se necessário."""
        tf = self._normalize_timeframe(timeframe)
        sym = symbol.lower()
        filename = f"model_{sym}_{tf}.pkl"
        path = os.path.join(self.model_dir, filename)
        if not os.path.exists(path):
            try:
                logger.info(f"⬇️ Baixando modelo {filename} do Google Drive...")
                download_file(filename, path, drive_folder_id=get_folder_id_for_file(filename))
                logger.info(f"✅ Modelo {filename} baixado do Google Drive.")
            except Exception as e:
                logger.error(f"⚠️ Não foi possível baixar modelo {filename} do Google Drive: {e}")
        return path

    @lru_cache(maxsize=10)
    def _load_model(self, symbol: str, timeframe: str) -> Optional[object]:
        """Carrega modelo do disco (ou do Google Drive se necessário), com cache LRU."""
        try:
            sym = symbol.lower().strip()
            tf = self._normalize_timeframe(timeframe)
            model_key = (sym, tf)
            if model_key in self.model_cache:
                last_used = self.last_used.get(model_key)
                if last_used and (datetime.now() - last_used) < self.cache_expiry:
                    return self.model_cache[model_key]

            model_path = self._ensure_model_local(sym, tf)
            if not os.path.exists(model_path):
                logger.warning(f"Modelo não encontrado localmente nem no Drive: {model_path}")
                return None

            model_obj = joblib.load(model_path)
            if isinstance(model_obj, dict) and 'model' in model_obj:
                model = model_obj['model']
                self.features = model_obj.get('features', None)
            else:
                model = model_obj
                self.features = None

            if not hasattr(model, 'predict'):
                raise ValueError("Objeto carregado não é um modelo válido")

            self.model_cache[model_key] = model
            self.last_used[model_key] = datetime.now()
            logger.info(f"Modelo carregado: {sym.upper()} [{tf.upper()}]")
            return model
        except Exception as e:
            logger.error(f"Falha ao carregar modelo {symbol}/{timeframe}: {str(e)}")
            return None

    def _validate_candles(self, candles: List[Dict]) -> Optional[pd.DataFrame]:
        """Valida e converte lista de candles para DataFrame"""
        if not candles:
            logger.error("Lista de candles vazia")
            return None
        required_keys = {'open', 'high', 'low', 'close', 'volume', 'timestamp'}
        if not all(required_keys.issubset(c) for c in candles):
            logger.error("Candles com campos incompletos")
            return None
        try:
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Falha ao processar candles: {str(e)}")
            return None

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, timeframe: str = None, symbol: str = None) -> pd.DataFrame:
        if timeframe and timeframe.lower() in ['s1', '1s']:
            df = resample_candles(df, freq='10S')

        df = df.copy()
        closes = pd.Series(df["close"].values)
        highs = pd.Series(df["high"].values)
        lows = pd.Series(df["low"].values)
        volumes = pd.Series(df["volume"].values)

        # ==== INDICADORES ENRIQUECIDOS ====
        rsi = TechnicalIndicators.calc_rsi(closes)
        df["rsi_value"] = rsi["value"]
        df["rsi_zone"] = rsi["zone"]
        df["rsi_trend"] = rsi["trend"]

        macd = TechnicalIndicators.calc_macd(closes)
        df["macd_histogram"] = macd["histogram"]
        df["macd_line"] = macd["macd_line"]
        df["macd_signal_line"] = macd["signal_line"]
        df["macd_momentum"] = macd["momentum"]

        bb = TechnicalIndicators.calc_bollinger(closes)
        df["bb_upper"] = bb["upper"]
        df["bb_lower"] = bb["lower"]
        df["bb_width"] = bb["width"]
        df["bb_percent_b"] = bb["percent_b"]
        df["bb_position"] = bb["position"]

        atr = TechnicalIndicators.calc_atr(highs, lows, closes)
        df["atr_value"] = atr["value"]
        df["atr_ratio"] = atr["ratio"]
        df["atr_trend"] = atr["trend"]

        adx = TechnicalIndicators.calc_adx(highs, lows, closes)
        df["adx_value"] = adx["adx"]
        df["adx_di_plus"] = adx["di_plus"]
        df["adx_di_minus"] = adx["di_minus"]
        df["adx_strength"] = adx["strength"]

        ichimoku = TechnicalIndicators.calc_ichimoku(highs, lows, closes)
        df["ichimoku_conversion"] = ichimoku["conversion"]
        df["ichimoku_base"] = ichimoku["base"]
        df["ichimoku_leading_a"] = ichimoku["leading_a"]
        df["ichimoku_leading_b"] = ichimoku["leading_b"]
        df["ichimoku_cloud_position"] = ichimoku["cloud_position"]

        fibo = TechnicalIndicators.calc_fibonacci(highs, lows)
        df["fibo_23_6"] = fibo["23.6%"]
        df["fibo_38_2"] = fibo["38.2%"]
        df["fibo_50"] = fibo["50%"]
        df["fibo_61_8"] = fibo["61.8%"]

        supertrend = TechnicalIndicators.calc_supertrend(highs, lows, closes)
        df["supertrend_value"] = supertrend["value"]
        df["supertrend_direction"] = supertrend["direction"]
        df["supertrend_changed"] = supertrend["changed"]

        mprofile = TechnicalIndicators.get_market_profile(closes, volumes)
        df["market_poc"] = mprofile["poc"]
        df["market_va_low"] = mprofile["value_area"]["low"]
        df["market_va_high"] = mprofile["value_area"]["high"]

        stoch = TechnicalIndicators.calc_stochastic(highs, lows, closes)
        df["stoch_k"] = stoch["k_line"]
        df["stoch_d"] = stoch["d_line"]
        df["stoch_state"] = stoch["state"]
        df["stoch_cross"] = stoch["cross"]

        cci = TechnicalIndicators.calc_cci(highs, lows, closes)
        df["cci_value"] = cci["value"]
        df["cci_state"] = cci["state"]
        df["cci_momentum"] = cci["momentum"]
        df["cci_strength"] = cci["strength"]

        wr = TechnicalIndicators.calc_williams_r(highs, lows, closes)
        df["williamsr_value"] = wr["value"]
        df["williamsr_state"] = wr["state"]
        df["williamsr_trend"] = wr["trend"]

        psar = TechnicalIndicators.calc_parabolic_sar(highs, lows)
        df["psar_value"] = psar["value"]
        df["psar_trend"] = psar["trend"]
        df["psar_acceleration"] = psar["acceleration"]

        mom = TechnicalIndicators.calc_momentum(closes)
        df["momentum_value"] = mom["value"]
        df["momentum_trend"] = mom["trend"]
        df["momentum_acceleration"] = mom["acceleration"]
        df["momentum_strength"] = mom["strength"]

        roc = TechnicalIndicators.calc_roc(closes)
        df["roc_value"] = roc["value"]
        df["roc_trend"] = roc["trend"]
        df["roc_momentum"] = roc["momentum"]
        df["roc_extreme"] = roc["extreme"]

        dmi = TechnicalIndicators.calc_dmi(highs, lows, closes)
        df["dmi_adx"] = dmi["adx"]
        df["dmi_plus_di"] = dmi["plus_di"]
        df["dmi_minus_di"] = dmi["minus_di"]
        df["dmi_trend"] = dmi["trend"]
        df["dmi_crossover"] = dmi["crossover"]

        vwap = TechnicalIndicators.calc_vwap(highs, lows, closes, volumes)
        df["vwap_value"] = vwap["value"]
        df["vwap_relation"] = vwap["relation"]
        df["vwap_spread"] = vwap["spread"]
        df["vwap_trend"] = vwap["trend"]

        envelope = TechnicalIndicators.calc_envelope(closes)
        df["envelope_upper"] = envelope["upper"]
        df["envelope_lower"] = envelope["lower"]
        df["envelope_center"] = envelope["center"]
        df["envelope_position"] = envelope["position"]
        df["envelope_band_width"] = envelope["band_width"]
        df["envelope_percent_center"] = envelope["percent_from_center"]

        elliott = TechnicalIndicators.calc_elliott_wave(closes)
        df["elliott_peaks"] = str(elliott.get("peaks", []))
        df["elliott_troughs"] = str(elliott.get("troughs", []))
        df["elliott_phase"] = elliott.get("phase", "")
        df["elliott_wave_counts"] = str(elliott.get("wave_counts", {}))

        zz = TechnicalIndicators.calc_zigzag(closes)
        df["zigzag_peaks"] = str(zz.get("peaks", []))
        df["zigzag_troughs"] = str(zz.get("troughs", []))
        df["zigzag_trend"] = zz.get("trend", "")
        df["zigzag_pattern"] = zz.get("pattern", "")
        df["zigzag_retracements"] = str(zz.get("retracements", []))

        # ========= AUXILIARES CONTEXTUAIS =========
        ma_rating = TechnicalIndicators.calc_moving_averages(closes)
        osc_rating = TechnicalIndicators.calc_oscillators(rsi["value"], macd["histogram"])
        vol = TechnicalIndicators.calc_volatility(closes)
        volstat = TechnicalIndicators.calc_volume_status(volumes)
        sentiment = TechnicalIndicators.calc_sentiment(closes)
        trendctx = TechnicalIndicators.get_trend_context(closes)
        sr = TechnicalIndicators.get_support_resistance(closes)
        df["ma_rating"] = ma_rating["rating"]
        df["osc_rating"] = osc_rating["rating"]
        df["volatility_level"] = vol["level"]
        df["volume_status"] = volstat["status"]
        df["sentiment"] = sentiment["sentiment"]
        df["trend_score"] = trendctx["trend_score"]
        df["trend_strength"] = trendctx["trend_strength"]
        df["trend_suggestion"] = trendctx["suggestion"]
        df["support_lvls"] = str(sr.get("support", []))
        df["resistance_lvls"] = str(sr.get("resistance", []))
        df["price_position"] = sr.get("current_position", "")

        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        df["obv"] = calc_obv(df)
        df["spread"] = calc_spread(df)
        df["variation"] = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1)) * 100

        if timeframe and timeframe.lower() in ['h4', 'd1']:
            df["cot"] = get_cot_feature(symbol)
            df["macro"] = get_macro_feature(symbol)
            df["sentiment_news"] = get_sentiment_feature(symbol)
        else:
            for col in ["cot", "macro", "sentiment_news"]:
                df[col] = 0

        ma_mapping = {"buy": 1, "sell": -1, "neutral": 0}
        osc_mapping = {"buy": 1, "sell": -1, "neutral": 0}
        vol_mapping = {"High": 1, "Low": 0}
        volstat_mapping = {"Spiked": 2, "Normal": 1, "Low": 0}
        sentiment_mapping = {"Optimistic": 1, "Neutral": 0, "Pessimistic": -1}
        trend_mapping = {"strong": 2, "moderate": 1, "weak": 0, "bearish": -1}
        zone_mapping = {"overbought": 1, "neutral": 0, "oversold": -1}
        df["ma_rating"] = df["ma_rating"].map(ma_mapping)
        df["osc_rating"] = df["osc_rating"].map(osc_mapping)
        df["volatility_level"] = df["volatility_level"].map(vol_mapping)
        df["volume_status"] = df["volume_status"].map(volstat_mapping)
        df["sentiment"] = df["sentiment"].map(sentiment_mapping)
        df["rsi_zone"] = df["rsi_zone"].map(zone_mapping)
        df["trend_strength"] = df["trend_strength"].map(trend_mapping)
        
        # Diferença entre médias móveis (curta e longa)
        df['diff_sma_5_20'] = df['sma_5'] - df['sma_20']
        df['diff_ema_12_26'] = df['ema_12'] - df['ema_26']

        # Cruzamento de médias móveis (flag booleana)
        df['cross_sma_5_20'] = ((df['sma_5'] > df['sma_20']) & (df['sma_5'].shift(1) <= df['sma_20'].shift(1))).astype(int)
        df['cross_ema_12_26'] = ((df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))).astype(int)

        # Cruzamento MACD/Signal
        df['macd_cross'] = ((df['macd_line'] > df['macd_signal_line']) & (df['macd_line'].shift(1) <= df['macd_signal_line'].shift(1))).astype(int)

        # Supondo que você já tem uma lista de padrões em cada candle em df['patterns']
        df['num_patterns'] = df['patterns'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df['rare_pattern_event'] = (df['num_patterns'] >= 3).astype(int)

        # ATR para múltiplos períodos
        for period in [7, 14, 21, 28]:
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df[f'atr_{period}'] = true_range.rolling(period).mean()
            # Você pode criar também a razão ATR/close para cada período:
            df[f'atr_{period}_pct'] = df[f'atr_{period}'] / df['close']

    # Bandas de Bollinger para múltiplos períodos
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = sma + 2 * std
            df[f'bb_lower_{period}'] = sma - 2 * std
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_pct_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])

        df.ffill(inplace=True)
        df.dropna(inplace=True)
        return df

    def _add_candlestick_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pattern_list = [
            "bullish_engulfing", "bearish_engulfing", "hammer", "hanging_man", "inverted_hammer", "shooting_star",
            "morning_star", "evening_star", "piercing_line", "dark_cloud_cover","three_white_soldiers",
            "three_black_crows", "abandoned_baby_bullish", "abandoned_baby_bearish", "kicker_bullish", "kicker_bearish",
            "rising_three_methods", "falling_three_methods", "upside_tasuki_gap", "downside_tasuki_gap",
            "separating_lines", "doji", "dragonfly_doji", "gravestone_doji", "long_legged_doji", "spinning_top",
            "marubozu", "bullish_harami", "bearish_harami", "harami_cross", "tweezer_bottom", "tweezer_top",
            "three_inside_up", "three_inside_down", "three_outside_up", "three_outside_down", "gap_up",
            "gap_down", "on_neckline", "belt_hold_bullish", "belt_hold_bearish", "counterattack_bullish",
            "counterattack_bearish", "unique_three_river_bottom", "breakaway_bullish", "breakaway_bearish",
            "pattern_strength", "patterns",
        ]
        for pattern in pattern_list:
            df[pattern] = 0
        pattern_strengths = []
        patterns_col = []
        for i in range(len(df)):
            candles = df.iloc[max(i-5, 0):i+1][["open", "high", "low", "close", "volume"]].to_dict("records")
            patterns = detect_candlestick_patterns(candles)
            patterns_col.append(patterns)
            for pattern in pattern_list:
                if pattern in patterns:
                    df.at[df.index[i], pattern] = 1
            pattern_strengths.append(get_pattern_strength(patterns))
        df["pattern_strength"] = pattern_strengths
        df["patterns"] = patterns_col
        return df

    def _get_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        # Use a lista salva no modelo, se disponível, senão use o padrão
        features = getattr(self, "features", None)
        if not features:
            features = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility',
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_12', 'ema_26',
                'rsi_value', 'rsi_zone', 'rsi_trend',
                'macd_histogram', 'macd_line', 'macd_signal_line', 'macd_momentum',
                'bb_upper', 'bb_lower', 'bb_width', 'bb_percent_b', 'bb_position',
                'atr_value', 'atr_ratio', 'atr_trend',
                'adx_value', 'adx_di_plus', 'adx_di_minus', 'adx_strength',
                'ichimoku_conversion', 'ichimoku_base', 'ichimoku_leading_a', 'ichimoku_leading_b', 'ichimoku_cloud_position',
                'fibo_23_6', 'fibo_38_2', 'fibo_50', 'fibo_61_8',
                'supertrend_value', 'supertrend_direction', 'supertrend_changed',
                'market_poc', 'market_va_low', 'market_va_high',
                'stoch_k', 'stoch_d', 'stoch_state', 'stoch_cross',
                'cci_value', 'cci_state', 'cci_momentum', 'cci_strength',
                'williamsr_value', 'williamsr_state', 'williamsr_trend',
                'psar_value', 'psar_trend', 'psar_acceleration',
                'momentum_value', 'momentum_trend', 'momentum_acceleration', 'momentum_strength',
                'roc_value', 'roc_trend', 'roc_momentum', 'roc_extreme',
                'dmi_adx', 'dmi_plus_di', 'dmi_minus_di', 'dmi_trend', 'dmi_crossover',
                'vwap_value', 'vwap_relation', 'vwap_spread', 'vwap_trend',
                'envelope_upper', 'envelope_lower', 'envelope_center', 'envelope_position', 'envelope_band_width', 'envelope_percent_center',
                'elliott_peaks', 'elliott_troughs', 'elliott_phase', 'elliott_wave_counts',
                'zigzag_peaks', 'zigzag_troughs', 'zigzag_trend', 'zigzag_pattern', 'zigzag_retracements',
                'ma_rating', 'osc_rating', 'volatility_level', 'volume_status', 'sentiment',
                'trend_score', 'trend_strength', 'trend_suggestion', 'support_lvls', 'resistance_lvls', 'price_position',
                'obv', 'spread', 'variation',
                'cot', 'macro', 'sentiment_news',

                "bullish_engulfing", "bearish_engulfing", "hammer", "hanging_man", "inverted_hammer", "shooting_star",
                "morning_star", "evening_star", "piercing_line", "dark_cloud_cover","three_white_soldiers",
                "three_black_crows", "abandoned_baby_bullish", "abandoned_baby_bearish", "kicker_bullish", "kicker_bearish",
                "rising_three_methods", "falling_three_methods", "upside_tasuki_gap", "downside_tasuki_gap",
                "separating_lines", "doji", "dragonfly_doji", "gravestone_doji", "long_legged_doji", "spinning_top",
                "marubozu", "bullish_harami", "bearish_harami", "harami_cross", "tweezer_bottom", "tweezer_top",
                "three_inside_up", "three_inside_down", "three_outside_up", "three_outside_down", "gap_up",
                "gap_down", "on_neckline", "belt_hold_bullish", "belt_hold_bearish", "counterattack_bullish",
                "counterattack_bearish", "unique_three_river_bottom", "breakaway_bullish", "breakaway_bearish",
                "pattern_strength", "patterns",
                
                "diff_sma_5_20", "diff_ema_12_26", "cross_sma_5_20", "cross_ema_12_26", "macd_cross",
                "num_patterns", "rare_pattern_event",
                "atr_7", "atr_14", "atr_21", "atr_28", "atr_7_pct", "atr_14_pct", "atr_21_pct", "atr_28_pct",
                "bb_upper_10", "bb_lower_10", "bb_width_10", "bb_pct_10",
                "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_pct_20",
                "bb_upper_50", "bb_lower_50", "bb_width_50", "bb_pct_50"
            ]
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.error(f"Features obrigatórias ausentes: {missing}")
            return None
        return df[features].iloc[[-1]]
    
    def predict(self, symbol: str, timeframe: str, candles: List[Dict]) -> Optional[str]:
        """
        Faz previsão de direção usando modelo de ML ('up', 'down' ou None)
        """
        try:
            if not candles or len(candles) < 30:
                logger.warning(f"Dados insuficientes: fornecidos {len(candles) if candles else 0} candles")
                return None

            df = self._validate_candles(candles)
            if df is None:
                return None

            # Mantém últimos candles necessários
            candles_to_use = candles[-self.min_candles:] if len(candles) >= self.min_candles else candles
            df = self._validate_candles(candles_to_use)
            if df is None:
                return None

            model = self._load_model(symbol, timeframe)
            if model is None:
                return None

            df = MLPredictor.add_technical_indicators(df, timeframe, symbol)
            df = self._add_candlestick_features(df)
            df.dropna(inplace=True)

            features = self._get_features(df)
            if features is None:
                return None

            pred = model.predict(features)
            return 'up' if pred[0] == 1 else 'down'

        except Exception as e:
            logger.error(f"Erro durante previsão: {str(e)}", exc_info=True)
            return None

    def predict_with_confidence(self, symbol: str, timeframe: str, candles: List[Dict]) -> Optional[Dict]:
        """
        Faz previsão e retorna direção, confiança, features e timestamp
        """
        try:
            prediction = self.predict(symbol, timeframe, candles)
            if prediction is None:
                return None

            df = self._validate_candles(candles)
            candles_to_use = candles[-self.min_candles:] if len(candles) >= self.min_candles else candles
            df = self._validate_candles(candles_to_use)
            if df is None:
                return None

            df = MLPredictor.add_technical_indicators(df, timeframe, symbol)
            df = self._add_candlestick_features(df)
            df.dropna(inplace=True)

            model = self._load_model(symbol, timeframe)
            features = self._get_features(df)
            if features is None:
                return None

            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
            features_dict = features.iloc[0].to_dict()

            return {
                'direction': prediction,
                'confidence': confidence,
                'features': features_dict,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro em predict_with_confidence: {str(e)}", exc_info=True)
            return None
