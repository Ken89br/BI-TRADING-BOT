# strategy/train_model_historic.py
# Pipeline unificado: Treinamento de modelos com upload/download via Google Drive e máxima compatibilidade

import os
import glob
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List, Optional

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit

# Indicadores e padrões do seu projeto
from strategy.ml_utils import add_indicators
from strategy.candlestick_patterns import detect_candlestick_patterns, get_pattern_strength

# Google Drive utilities
from data.google_drive_client import upload_or_update_file as upload_file, download_file, find_file_id, get_folder_id_for_file

from data.fundamental_data import get_cot_feature, get_macro_feature, get_sentiment_feature
from utils.features_extra import calc_obv, calc_spread
from utils.aggregation import resample_candles

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RETRAIN_INTERVALS = {
    "s1": 30, "m1": 60, "m5": 300, "m15": 900, "m30": 1800, "h1": 3600, "h4": 14400
}
LAST_RETRAIN_TIMES = {}

MIN_CANDLES = 100  # Patch: mínimo de candles válidos para treinar

def get_symbol_and_timeframe_from_filename(filename):
    base = os.path.basename(filename).lower()
    if "_" in base and base.endswith(".csv"):
        symbol, tf_part = base.rsplit("_", 1)
        tf = tf_part.replace(".csv", "")
        return symbol, tf
    return None, None

def ensure_local_file(filename, folder="data"):
    local_path = os.path.join(folder, filename)
    if not os.path.exists(local_path):
        try:
            logger.info(f"⬇️ Baixando {filename} do Google Drive...")
            download_file(filename, local_path, drive_folder_id=get_folder_id_for_file(filename))
            logger.info(f"✅ Baixado {filename} do Google Drive.")
        except Exception as e:
            logger.error(f"⚠️ Não foi possível baixar {filename}: {e}")
    return local_path

class DataProcessor:
    """Processamento seguro de dados temporais para trading"""

    @staticmethod
    def load_and_validate_data(filepath: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(filepath)
            required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"Arquivo {filepath} não contém colunas necessárias")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.sort_values("timestamp", inplace=True)
            df = df.reset_index(drop=True)
            if df.isnull().values.any():
                logger.warning(f"Dados ausentes encontrados em {filepath}")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar {filepath}: {str(e)}", exc_info=True)
            return None

    @staticmethod
    def merge_and_save_candles(filepath, new_candles):
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

    @staticmethod
    def create_target_variable(df: pd.DataFrame, future_bars: int = 1) -> pd.DataFrame:
        """Cria variável target para previsão de tendência"""
        df = df.copy()
        future_price = df["close"].shift(-future_bars)
        df["target"] = (future_price > df["close"]).astype(int)
        df.dropna(subset=["target"], inplace=True)
        return df

    @staticmethod
    def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        split_idx = int(len(df) * (1 - test_size))
        return df.iloc[:split_idx], df.iloc[split_idx:]

class FeatureEngineer:
    """Engenharia de features para trading"""

    INDICATOR_CONFIG = {
        'sma_periods': [5, 10, 20, 50],
        'ema_periods': [12, 26],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bollinger_period': 20,
        'bollinger_std': 2,
        'atr_period': 14,
        'volatility_window': 20
    }

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, timeframe: str = None, symbol: str = None) -> pd.DataFrame:
        # --- Agrupamento de S1 em 10s ---
        if timeframe and timeframe.lower() in ['s1', '1s']:
            df = resample_candles(df, freq='10S')

        df = df.copy()
        closes = pd.Series(df["close"].values)
        highs = pd.Series(df["high"].values)
        lows = pd.Series(df["low"].values)
        volumes = pd.Series(df["volume"].values)

        # ==== INDICADORES ENRIQUECIDOS ====
        # RSI
        rsi = TechnicalIndicators.calc_rsi(closes)
        df["rsi_value"] = rsi["value"]
        df["rsi_zone"] = rsi["zone"]
        df["rsi_trend"] = rsi["trend"]

        # MACD
        macd = TechnicalIndicators.calc_macd(closes)
        df["macd_histogram"] = macd["histogram"]
        df["macd_line"] = macd["macd_line"]
        df["macd_signal_line"] = macd["signal_line"]
        df["macd_momentum"] = macd["momentum"]

        # Bollinger Bands
        bb = TechnicalIndicators.calc_bollinger(closes)
        df["bb_upper"] = bb["upper"]
        df["bb_lower"] = bb["lower"]
        df["bb_width"] = bb["width"]
        df["bb_percent_b"] = bb["percent_b"]
        df["bb_position"] = bb["position"]

        # ATR
        atr = TechnicalIndicators.calc_atr(highs, lows, closes)
        df["atr_value"] = atr["value"]
        df["atr_ratio"] = atr["ratio"]
        df["atr_trend"] = atr["trend"]

        # ADX
        adx = TechnicalIndicators.calc_adx(highs, lows, closes)
        df["adx_value"] = adx["adx"]
        df["adx_di_plus"] = adx["di_plus"]
        df["adx_di_minus"] = adx["di_minus"]
        df["adx_strength"] = adx["strength"]

        # Ichimoku
        ichimoku = TechnicalIndicators.calc_ichimoku(highs, lows, closes)
        df["ichimoku_conversion"] = ichimoku["conversion"]
        df["ichimoku_base"] = ichimoku["base"]
        df["ichimoku_leading_a"] = ichimoku["leading_a"]
        df["ichimoku_leading_b"] = ichimoku["leading_b"]
        df["ichimoku_cloud_position"] = ichimoku["cloud_position"]

        # Fibonacci
        fibo = TechnicalIndicators.calc_fibonacci(highs, lows)
        df["fibo_23_6"] = fibo["23.6%"]
        df["fibo_38_2"] = fibo["38.2%"]
        df["fibo_50"] = fibo["50%"]
        df["fibo_61_8"] = fibo["61.8%"]

        # Supertrend
        supertrend = TechnicalIndicators.calc_supertrend(highs, lows, closes)
        df["supertrend_value"] = supertrend["value"]
        df["supertrend_direction"] = supertrend["direction"]
        df["supertrend_changed"] = supertrend["changed"]

        # Market Profile
        mprofile = TechnicalIndicators.get_market_profile(closes, volumes)
        df["market_poc"] = mprofile["poc"]
        df["market_va_low"] = mprofile["value_area"]["low"]
        df["market_va_high"] = mprofile["value_area"]["high"]

        # Stochastic
        stoch = TechnicalIndicators.calc_stochastic(highs, lows, closes)
        df["stoch_k"] = stoch["k_line"]
        df["stoch_d"] = stoch["d_line"]
        df["stoch_state"] = stoch["state"]
        df["stoch_cross"] = stoch["cross"]

        # CCI
        cci = TechnicalIndicators.calc_cci(highs, lows, closes)
        df["cci_value"] = cci["value"]
        df["cci_state"] = cci["state"]
        df["cci_momentum"] = cci["momentum"]
        df["cci_strength"] = cci["strength"]

        # Williams %R
        wr = TechnicalIndicators.calc_williams_r(highs, lows, closes)
        df["williamsr_value"] = wr["value"]
        df["williamsr_state"] = wr["state"]
        df["williamsr_trend"] = wr["trend"]

        # Parabolic SAR
        psar = TechnicalIndicators.calc_parabolic_sar(highs, lows)
        df["psar_value"] = psar["value"]
        df["psar_trend"] = psar["trend"]
        df["psar_acceleration"] = psar["acceleration"]

        # Momentum
        mom = TechnicalIndicators.calc_momentum(closes)
        df["momentum_value"] = mom["value"]
        df["momentum_trend"] = mom["trend"]
        df["momentum_acceleration"] = mom["acceleration"]
        df["momentum_strength"] = mom["strength"]

        # ROC
        roc = TechnicalIndicators.calc_roc(closes)
        df["roc_value"] = roc["value"]
        df["roc_trend"] = roc["trend"]
        df["roc_momentum"] = roc["momentum"]
        df["roc_extreme"] = roc["extreme"]

        # DMI
        dmi = TechnicalIndicators.calc_dmi(highs, lows, closes)
        df["dmi_adx"] = dmi["adx"]
        df["dmi_plus_di"] = dmi["plus_di"]
        df["dmi_minus_di"] = dmi["minus_di"]
        df["dmi_trend"] = dmi["trend"]
        df["dmi_crossover"] = dmi["crossover"]

        # VWAP
        vwap = TechnicalIndicators.calc_vwap(highs, lows, closes, volumes)
        df["vwap_value"] = vwap["value"]
        df["vwap_relation"] = vwap["relation"]
        df["vwap_spread"] = vwap["spread"]
        df["vwap_trend"] = vwap["trend"]

        # Envelope
        envelope = TechnicalIndicators.calc_envelope(closes)
        df["envelope_upper"] = envelope["upper"]
        df["envelope_lower"] = envelope["lower"]
        df["envelope_center"] = envelope["center"]
        df["envelope_position"] = envelope["position"]
        df["envelope_band_width"] = envelope["band_width"]
        df["envelope_percent_center"] = envelope["percent_from_center"]

        # Elliott Wave
        elliott = TechnicalIndicators.calc_elliott_wave(closes)
        df["elliott_peaks"] = str(elliott.get("peaks", []))
        df["elliott_troughs"] = str(elliott.get("troughs", []))
        df["elliott_phase"] = elliott.get("phase", "")
        df["elliott_wave_counts"] = str(elliott.get("wave_counts", {}))

        # Zigzag
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

        # Rolling indicators
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
        for period in [12, 26]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # OBV, spread, variation, etc
        df["obv"] = calc_obv(df)
        df["spread"] = calc_spread(df)
        df["variation"] = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1)) * 100

        # Fundamentalistas para H4/D1
        if timeframe and timeframe.lower() in ['h4', 'd1']:
            df["cot"] = get_cot_feature(symbol)
            df["macro"] = get_macro_feature(symbol)
            df["sentiment_news"] = get_sentiment_feature(symbol)
        else:
            for col in ["cot", "macro", "sentiment_news"]:
                df[col] = 0
        
        # PADRÕES DE VELA:
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

        # Mapeamento para valores numéricos compatíveis com o restante do bot
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

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Lista de features para treino - ajuste para compatibilidade"""
        return [
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
            "diff_sma_5_20", "diff_ema_12_26", "cross_sma_5_20", "cross_ema_12_26", "macd_cross",
            "num_patterns", "rare_pattern_event",
            "atr_7", "atr_14", "atr_21", "atr_28", "atr_7_pct", "atr_14_pct", "atr_21_pct", "atr_28_pct",
            "bb_upper_10", "bb_lower_10", "bb_width_10", "bb_pct_10",
            "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_pct_20",
            "bb_upper_50", "bb_lower_50", "bb_width_50", "bb_pct_50",
        
            # PADRÕES DE VELA
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

    @staticmethod
    def create_feature_pipeline() -> Pipeline:
        """Pipeline de transformação de features"""
        return Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

class ModelTrainer:
    """Treinamento e avaliação de modelos"""

    MODEL_CONFIG = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50,
        'random_state': 42
    }

    def __init__(self):
        self.features = FeatureEngineer.get_feature_columns()
        self.feature_pipeline = FeatureEngineer.create_feature_pipeline()

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = self.feature_pipeline.fit_transform(df[self.features])
        y = df["target"].values
        return X, y

    def train_model(self, X: np.ndarray, y: np.ndarray) -> XGBClassifier:
        model = XGBClassifier(**self.MODEL_CONFIG)
        tscv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=10
            )
        return model

    def evaluate_model(self, model: XGBClassifier, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_pred = model.predict(X_test)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        return {"classification_report": clf_report}

    def save_model(self, model: XGBClassifier, symbol: str, tf: str) -> str:
        """Salva modelo SEM timestamp (sempre sobrescreve)"""
        model_name = f"model_{symbol}_{tf}.pkl"
        model_path = os.path.join(MODEL_DIR, model_name)
        joblib.dump({
            'model': model,
            'features': self.features,
            'pipeline': self.feature_pipeline,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }, model_path)
        return model_path

def ensure_latest_model(symbol, tf):
    """Baixa modelo do Drive se não existir localmente (usado só na predição!)"""
    filename = f"model_{symbol.lower()}_{tf}.pkl"
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        try:
            logger.info(f"⬇️ Baixando modelo {filename} do Google Drive...")
            download_file(filename, path, drive_folder_id=get_folder_id_for_file(filename))
            logger.info(f"✅ Modelo {filename} baixado do Google Drive.")
        except Exception as e:
            logger.error(f"⚠️ Não foi possível baixar {filename}: {e}")

def train_pipeline(filepath: str) -> Optional[Dict]:
    """Pipeline completo para um arquivo de dados"""
    from data.data_client import FallbackDataClient
    try:
        logger.info(f"Iniciando processamento para: {filepath}")
        df = DataProcessor.load_and_validate_data(filepath)
        symbol, tf = get_symbol_and_timeframe_from_filename(os.path.basename(filepath))
        if not symbol or not tf:
            logger.warning(f"Arquivo com nome inesperado: {filepath}")
            return None

        if df is None or len(df) < MIN_CANDLES:
            logger.warning(f"Arquivo {filepath} inválido ou com poucos dados ({0 if df is None else len(df)} linhas). Buscando candles frescos...")
            fallback_client = FallbackDataClient()
            interval = tf
            candles_result = fallback_client.fetch_candles(symbol, interval=interval, limit=500)
            candles = candles_result["history"] if candles_result and "history" in candles_result else None
            if not candles or len(candles) < MIN_CANDLES:
                logger.error(f"Não foi possível obter candles válidos para {symbol}/{tf}. Abortando.")
                return None
            DataProcessor.merge_and_save_candles(filepath, candles)
            df = DataProcessor.load_and_validate_data(filepath)

        logger.info(f"Processando dados para {symbol}/{tf} ({len(df)} registros)")
        df = FeatureEngineer.add_technical_indicators(df, timeframe=tf, symbol=symbol)
        df = DataProcessor.create_target_variable(df, future_bars=3)
        train_df, test_df = DataProcessor.temporal_split(df, test_size=0.2)
        trainer = ModelTrainer()
        X_train, y_train = trainer.prepare_data(train_df)
        model = trainer.train_model(X_train, y_train)
        X_test, y_test = trainer.prepare_data(test_df)
        evaluation = trainer.evaluate_model(model, X_test, y_test)
        model_path = trainer.save_model(model, symbol, tf)
        logger.info(f"Modelo treinado e salvo em: {model_path}")
        logger.info("Relatório de classificação:\n%s", evaluation["classification_report"])
        try:
            file_id = upload_file(model_path)
            logger.info(f"☁️ Arquivo {os.path.basename(model_path)} enviado ao Google Drive! ID: {file_id}")
        except Exception as e:
            logger.error(f"⚠️ Falha ao enviar {os.path.basename(model_path)} ao Google Drive: {e}")
        return {
            "symbol": symbol,
            "timeframe": tf,
            "model_path": model_path,
            "evaluation": evaluation,
            "train_samples": len(train_df),
            "test_samples": len(test_df)
        }
    except Exception as e:
        logger.error(f"Erro no pipeline para {filepath}: {str(e)}", exc_info=True)
        return None
        
def main():
    """Fluxo principal"""
    try:
        logger.info("Iniciando pipeline de treinamento de modelos")
        data_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
        if not data_files:
            logger.error(f"Nenhum arquivo CSV encontrado em {DATA_DIR}")
            return
        logger.info(f"Encontrados {len(data_files)} arquivos para processamento")
        results = []
        now = datetime.utcnow()
        for filepath in data_files:
            symbol, tf = get_symbol_and_timeframe_from_filename(os.path.basename(filepath))
            if not symbol or not tf:
                logger.warning(f"Pulo arquivo com nome inesperado: {filepath}")
                continue
            interval = RETRAIN_INTERVALS.get(tf, 60)
            last = LAST_RETRAIN_TIMES.get((symbol, tf))
            if not last or (now - last).total_seconds() >= interval:
                ensure_local_file(os.path.basename(filepath), folder=DATA_DIR)
                result = train_pipeline(filepath)
                if result:
                    results.append(result)
                LAST_RETRAIN_TIMES[(symbol, tf)] = now
            else:
                time_left = interval - (now - last).total_seconds()
                logger.info(f"⏳ Skipping {symbol.upper()} [{tf.upper()}] — next in {round(time_left)}s")
            ensure_latest_model(symbol, tf)
        logger.info("\n=== Resumo do Treinamento ===")
        for res in results:
            logger.info(
                f"{res['symbol']} [{res['timeframe']}]: "
                f"Train={res['train_samples']}, Test={res['test_samples']}, "
                f"Accuracy={res['evaluation']['classification_report']['accuracy']:.2f}"
            )
        logger.info("Pipeline concluído com sucesso!")
    except Exception as e:
        logger.error(f"Erro fatal no pipeline principal: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
