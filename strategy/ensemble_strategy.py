# Função: Estratégia “conselho de especialistas” (ensemble), versão aprimorada (lookahead preditivo + expiração dinâmica)
# O que faz:
# - Reúne várias estratégias e indicadores para votar no sinal (“up” ou “down”)
# - Decide a direção final pela maioria ou ML em caso de empate.
# - Calcula indicadores ricos (RSI, MACD, ATR, ADX...) e monta o dicionário de sinal.
# - Busca o melhor candle de entrada e expiração nos próximos N candles, não só no último.
# - Expiração é DINÂMICA, baseada nas condições do mercado (volatilidade, tendência, reversão, etc).
# - Aplica um filtro inteligente (SmartAIFilter) antes de retornar o sinal.

import time
from datetime import datetime, timedelta
from config import CONFIG
from strategy.feature_universal import prepare_universal_features
from strategy.candlestick_strategy import CandlestickStrategy
from strategy.rsi_ma import AggressiveRSIMA
from strategy.bollinger_breakout import BollingerBreakoutStrategy
from strategy.wick_reversal import WickReversalStrategy
from strategy.macd_reversal import MACDReversalStrategy
from strategy.rsi import RSIStrategy
from strategy.bbands import BollingerStrategy
from strategy.sma_cross import SMACrossStrategy
from strategy.ai_filter import SmartAIFilter
from strategy.ml_predictor import MLPredictor
from strategy.price_action import EnhancedPriceActionStrategy
from strategy.ema_strategy import EMAStrategy
from strategy.atr_strategy import ATRStrategy
from strategy.adx_strategy import ADXStrategy

from strategy.candlestick_patterns import detect_candlestick_patterns
from strategy.indicator_globe import TechnicalIndicators

from utils.cot_utils import get_latest_cot

class EnsembleStrategy:
    def __init__(self):
        self.strategies = [
            CandlestickStrategy(),
            AggressiveRSIMA(CONFIG["rsi_ma"]),
            BollingerBreakoutStrategy(CONFIG["bollinger_breakout"]),
            WickReversalStrategy(CONFIG["wick_reversal"]),
            MACDReversalStrategy(CONFIG["macd_reversal"]),
            RSIStrategy(CONFIG["rsi"]),
            SMACrossStrategy(),
            BollingerStrategy(CONFIG["bbands"]),
            EnhancedPriceActionStrategy(CONFIG["price_action"]),
            EMAStrategy(CONFIG["ema"]),
            ATRStrategy(CONFIG["atr"]),
            ADXStrategy(CONFIG["adx"]),
        ]
        self.filter = SmartAIFilter()
        self.ml = MLPredictor()

    def _score_entry(self, candles, idx):
        """
        Função customizada para pontuar cada candle futuro como possível entrada.
        Pode ser expandida para usar ML ou mais heurísticas.
        """
        c = candles[idx]
        prev = candles[idx-1] if idx > 0 else c
        score = 0
        if abs(c["close"] - c["open"]) > 0.5 * (max(c["high"], c["low"]) - min(c["high"], c["low"])):
            score += 1  # Candle forte
        if c["close"] > prev["close"]:
            score += 1  # Alta
        if c.get("volume", 0) > prev.get("volume", 0):
            score += 1  # Volume crescente
        # Adicione outros critérios: padrões, divergências, etc.
        return score

    def _dynamic_expiry(self, candles, entry_idx, indicators):
        """
        Calcula o número ideal de candles para expiração, baseado nas condições do mercado.
        Parâmetros:
            candles: lista de candles (dicts)
            entry_idx: índice do candle de entrada
            indicators: dicionário de indicadores do contexto (volatilidade, atr, adx, padrões, etc)
        Retorna:
            N_expire: int (quantos candles após a entrada deve ser a expiração)
        """
        # Parâmetros base (podem ser ajustados/configurados)
        min_expiry = CONFIG.get("min_expiry_candles", 1)
        max_expiry = CONFIG.get("max_expiry_candles", 5)

        # Critérios práticos:
        # - Se volatilidade muito alta: expiração curta (1-2 candles)
        # - Se volatilidade baixa e tendência forte (ADX alto): expiração longa (até 5)
        # - Se padrão de reversão: expiração curta
        # - Se ATR alto: expiração intermediária
        volatility = indicators.get("volatility", "Medium")
        adx = indicators.get("adx", 25)
        atr = indicators.get("atr", 1)
        patterns = indicators.get("patterns", [])
        trend_strong = adx > 30
        reversal_pattern = any(pat for pat in patterns if "reversal" in str(pat).lower() or "engulf" in str(pat).lower())

        if volatility == "High" and not trend_strong:
            N_expire = min_expiry  # Alta volatilidade e tendência fraca: expiração curta
        elif trend_strong and volatility != "High":
            N_expire = max_expiry  # Tendência forte e volatilidade controlada: expiração mais longa
        elif reversal_pattern:
            N_expire = min_expiry  # Reversão: saída rápida
        elif atr > 2:
            N_expire = max(min_expiry + 2, max_expiry - 1)  # ATR alto: intermediário
        else:
            N_expire = CONFIG.get("default_expiry_candles", 2)
        # Garante limites
        N_expire = max(min_expiry, min(max_expiry, N_expire))
        return N_expire

    def generate_signal(self, data, timeframe="1min"):
        symbol = data["symbol"]
        cot_info = get_latest_cot(symbol)

        candles = data["history"]
        # Use o DataFrame universal:
        features_df = prepare_universal_features(candles, symbol, timeframe)
        if features_df.empty or len(features_df) < 3:
            print("⚠️ Insufficient features, skip signal.")
            return None

        votes, details = [], []
        for strat in self.strategies:
            try:
                # Passe o DataFrame universal para todas as estratégias!
                result = strat.generate_signal(features_df)
                if result:
                    votes.append(result["signal"].lower())
                    details.append(result)
            except Exception as e:
                print(f"⚠️ {type(strat).__name__} failed: {e}")
        
        if not votes:
            print("⚠️ No strategies returned a signal.")
            return None

        up_votes = votes.count("up")
        down_votes = votes.count("down")

        ml_direction = None

        if up_votes > down_votes:
            direction = "up"
        elif down_votes > up_votes:
            direction = "down"
        else:
            print("⚠️ Equal votes — using ML to break tie.")
            try:
                ml_direction = self.ml.predict(data["symbol"], timeframe, data["history"])
                if ml_direction:
                    direction = ml_direction
                else:
                    print("⚠️ ML undecided — skipping signal.")
                    return None
            except Exception as e:
                print(f"⚠️ ML predictor failed during tiebreaker: {e}")
                return None

        confidence = round((max(up_votes, down_votes) / len(votes)) * 100)
        strength = "strong" if confidence >= 70 else "moderate"

        candles = data["history"]
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        volumes = [c.get("volume", 0) for c in candles]
        
        closes = pd.Series(closes)
        highs = pd.Series(highs)
        lows = pd.Series(lows)
        volumes = pd.Series(volumes)

# RSI
rsi_res = TechnicalIndicators.calc_rsi(closes)
rsi_value = rsi_res['value']
rsi_zone = rsi_res['zone']
rsi_trend = rsi_res['trend']

# MACD
macd_res = TechnicalIndicators.calc_macd(closes)
macd_hist = macd_res['histogram']
macd_line = macd_res['macd_line']
macd_signal = macd_res['signal_line']
macd_momentum = macd_res['momentum']
macd_cross = macd_res['cross']

# Bollinger Bands
bollinger_res = TechnicalIndicators.calc_bollinger(closes)
bb_upper = bollinger_res['upper']
bb_lower = bollinger_res['lower']
bb_width = bollinger_res['width']
bb_percent_b = bollinger_res['percent_b']
bb_position = bollinger_res['position']
bb_price_position = bollinger_res['price_position']

# ATR
atr_res = TechnicalIndicators.calc_atr(highs, lows, closes)
atr_value = atr_res['value']
atr_ratio = atr_res['ratio']
atr_trend = atr_res['trend']

# ADX
adx_res = TechnicalIndicators.calc_adx(highs, lows, closes)
adx_value = adx_res['adx']
adx_strength = adx_res['strength']
adx_trend = adx_res['trend']
di_plus = adx_res['di_plus']
di_minus = adx_res['di_minus']

# Ichimoku
ichimoku_res = TechnicalIndicators.calc_ichimoku(highs, lows, closes)
ichimoku_conversion = ichimoku_res['conversion']
ichimoku_base = ichimoku_res['base']
ichimoku_a = ichimoku_res['leading_a']
ichimoku_b = ichimoku_res['leading_b']
ichimoku_cloud_position = ichimoku_res['cloud_position']

# Fibonacci
fibo_res = TechnicalIndicators.calc_fibonacci(highs, lows)
fibo_236 = fibo_res['23.6%']
fibo_382 = fibo_res['38.2%']
fibo_50 = fibo_res['50%']
fibo_618 = fibo_res['61.8%']

# Supertrend
supertrend_res = TechnicalIndicators.calc_supertrend(highs, lows, closes)
supertrend_value = supertrend_res['value']
supertrend_direction = supertrend_res['direction']
supertrend_changed = supertrend_res['changed']

# Market Profile
market_profile_res = TechnicalIndicators.get_market_profile(closes, volumes)
mp_poc = market_profile_res['poc']
mp_val_area_low = market_profile_res['value_area']['low']
mp_val_area_high = market_profile_res['value_area']['high']

# Stochastic
stoch_res = TechnicalIndicators.calc_stochastic(highs, lows, closes)
stoch_k = stoch_res['k']
stoch_d = stoch_res['d']
stoch_state = stoch_res['state']
stoch_trend = stoch_res['trend']

# CCI
cci_res = TechnicalIndicators.calc_cci(highs, lows, closes)
cci_value = cci_res['value']
cci_state = cci_res['state']

# Williams %R
williams_res = TechnicalIndicators.calc_williams_r(highs, lows, closes)
williams_value = williams_res['value']
williams_state = williams_res['state']

# Parabolic SAR
psar_res = TechnicalIndicators.calc_parabolic_sar(highs, lows)
psar_value = psar_res['value']
psar_trend = psar_res['trend']

# Momentum
mom_res = TechnicalIndicators.calc_momentum(closes)
mom_value = mom_res['value']
mom_trend = mom_res['trend']

# ROC
roc_res = TechnicalIndicators.calc_roc(closes)
roc_value = roc_res['value']
roc_trend = roc_res['trend']

# DMI
dmi_res = TechnicalIndicators.calc_dmi(highs, lows, closes)
dmi_plus = dmi_res['plus_di']
dmi_minus = dmi_res['minus_di']
dmi_adx = dmi_res['adx']
dmi_trend = dmi_res['trend']

# VWAP
vwap_res = TechnicalIndicators.calc_vwap(highs, lows, closes, volumes)
vwap_value = vwap_res['value']
vwap_relation = vwap_res['relation']

# Envelope
envelope_res = TechnicalIndicators.calc_envelope(closes)
envelope_upper = envelope_res['upper']
envelope_lower = envelope_res['lower']
envelope_center = envelope_res['center']
envelope_position = envelope_res['position']

# Zigzag
zigzag_res = TechnicalIndicators.calc_zigzag(closes)
zigzag_peaks = zigzag_res['peaks']
zigzag_troughs = zigzag_res['troughs']
zigzag_trend = zigzag_res['trend']

# Elliott Wave
elliott_res = TechnicalIndicators.calc_elliott_wave(closes)
elliott_peaks = elliott_res.get('peaks', [])
elliott_troughs = elliott_res.get('troughs', [])
elliott_phase = elliott_res.get('phase', '')

# ========== AUXILIARES (sempre dict!) ==========

# Médias móveis rating
ma_res = TechnicalIndicators.calc_moving_averages(closes)
ma_rating = ma_res['rating']

# Osciladores rating
osc_res = TechnicalIndicators.calc_oscillators(rsi_value, macd_hist)
osc_rating = osc_res['rating']

# Volatilidade
vol_res = TechnicalIndicators.calc_volatility(closes)
volatility = vol_res['level']

# Volume status
volstat_res = TechnicalIndicators.calc_volume_status(volumes)
volume_status = volstat_res['status']

# Sentimento
sentiment_res = TechnicalIndicators.calc_sentiment(closes)
sentiment = sentiment_res['sentiment']

# Força da tendência (score + strength)
trend_strength_res = TechnicalIndicators.get_trend_strength(closes)
trend_score = trend_strength_res['score']
trend_strength = trend_strength_res['strength']

# Suportes e resistências (últimos 3)
sr_res = TechnicalIndicators.get_support_resistance(closes)
support_lvls = sr_res.get('support', [])
resistance_lvls = sr_res.get('resistance', [])

        # --- BUSCA DO MELHOR CANDLE DE ENTRADA/EXPIRAÇÃO (LOOKAHEAD) ---
        LOOKAHEAD = CONFIG.get("max_lookahead_candles", 5)
        if len(candles) < LOOKAHEAD + 2:
            print("⚠️ Histórico insuficiente para lookahead, usando último candle como fallback.")
            best_entry_idx = -1
        else:
            best_score = float('-inf')
            best_entry_idx = -1
            for i in range(-LOOKAHEAD, -1):
                score = self._score_entry(candles, i)
                if score > best_score:
                    best_score = score
                    best_entry_idx = i
            if best_entry_idx == -1:
                best_entry_idx = -2  # Se tudo igual, pega penúltimo

        # Candle recomendado para entrada
        entry_candle = candles[best_entry_idx]
        entry_ts = entry_candle.get("timestamp") or entry_candle.get("t")
        entry_ts = int(entry_ts) / 1000 if entry_ts else time.time()
        entry_dt = datetime.utcfromtimestamp(entry_ts)
        entry_price = entry_candle["close"]

        # --- COLETA DOS INDICADORES DO CONTEXTO PARA EXPIRAÇÃO DINÂMICA ---
        rsi = calc_rsi(closes)
        macd_hist, macd_val, macd_signal = calc_macd(closes)
        bollinger_str, bb_width, bb_pos = calc_bollinger(closes)
        atr = calc_atr(highs, lows, closes)
        adx = calc_adx(highs, lows, closes)
        ma_rating = calc_moving_averages(closes)
        osc_rating = calc_oscillators(rsi, macd_hist)
        volatility = calc_volatility(closes)
        volume_status = calc_volume_status(volumes)
        sentiment = calc_sentiment(closes)
        patterns = detect_candlestick_patterns(candles)

        support = min(lows[-10:])
        resistance = max(highs[-10:])

        variation = f"{((closes[-1] - closes[-2]) / closes[-2]) * 100:.2f}%"

        rsi_str = f"{rsi:.1f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})"
        macd_str = f"{macd_val:.4f} (Hist: {macd_hist:.4f})"
        atr_str = f"{atr:.5f}"
        adx_str = f"{adx:.2f}"

        context_indicators = {
            "volatility": volatility,
            "adx": adx,
            "atr": atr,
            "patterns": patterns
        }

        # --- EXPIRAÇÃO DINÂMICA ---
        N_expire = self._dynamic_expiry(candles, best_entry_idx, context_indicators)
        expire_idx = best_entry_idx + N_expire
        if -len(candles) <= expire_idx < 0:
            expire_candle = candles[expire_idx]
            expire_ts = expire_candle.get("timestamp") or expire_candle.get("t")
            expire_ts = int(expire_ts) / 1000 if expire_ts else time.time()
            expire_dt = datetime.utcfromtimestamp(expire_ts)
            expire_price = expire_candle["close"]
        else:
            expire_dt = entry_dt + timedelta(minutes=N_expire)
            expire_price = entry_price

        # --- ENTRADA/EXPIRAÇÃO REALISTAS E DINÂMICAS ---

        signal_data = {
            "signal": direction,
            "strength": strength,
            "confidence": confidence,
            "price": entry_price,
            "recommended_entry_time": entry_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "recommended_entry_price": entry_price,
            "expire_entry_time": expire_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "expire_entry_price": expire_price,
            "high": max(c["high"] for c in data["history"]),
            "low": min(c["low"] for c in data["history"]),
            "volume": sum(c["volume"] for c in data["history"]),

            "variation": variation,
            "risk": "Low" if volatility == "Low" and adx < 25 else "High",
            "volatility": volatility,
            "sentiment": sentiment,
            "volume_status": volume_status,
            "support": support,
            "resistance": resistance,
            "summary": "strong buy" if direction == "up" else "strong sell" if direction == "down" else "neutral",
            "moving_averages": ma_rating,
            "oscillators": osc_rating,
            "rsi": rsi_str,
            "macd": macd_str,
            "bollinger": bollinger_str,
            "atr": atr_str,
            "adx": adx_str,
            "patterns": patterns  # <-- padrões já vão para o filtro
        }
        
    # Integração COT
    original_confidence = signal_data.get("confidence", 50)
    if cot_info:
    # 1. Armazenamento dos dados brutos
        signal_data.update({
            "cot_net_position": cot_info["net_position"],
            "cot_pct_long": cot_info["pct_long"],
            "cot_open_interest": cot_info["open_interest"],
            "cot_date": cot_info["date"],
            "cot_52w_high": cot_info.get("52w_high", None),  # Novo: Máximo histórico
            "cot_52w_low": cot_info.get("52w_low", None),    # Novo: Mínimo histórico
            "cot_4w_avg": cot_info.get("4w_avg", None)       # Novo: Média móvel
    })

    # 2. Cálculo de métricas derivadas
    cot_strength = (cot_info["pct_long"] - 0.5) * 2  # Normalizado entre -1 e 1
    signal_data["cot_strength"] = cot_strength

    # 3. Influência na confiança com sistema hierárquico
    base_confidence = original_confidence
    
    # Regra 1: Alinhamento direto
    if (signal_data["signal"] == "up" and cot_strength > 0.1) or \
       (signal_data["signal"] == "down" and cot_strength < -0.1):
        base_confidence += 15 * cot_strength  # Impacto proporcional
        
    # Regra 2: Extremos históricos
    elif cot_info.get("52w_high") and cot_info["pct_long"] > cot_info["52w_high"] * 0.9:
        base_confidence += 20 if signal_data["signal"] == "up" else -20
    elif cot_info.get("52w_low") and cot_info["pct_long"] < cot_info["52w_low"] * 1.1:
        base_confidence += 20 if signal_data["signal"] == "down" else -20
        
    # Regra 3: Tendência persistente
    elif cot_info.get("4w_avg"):
        if (cot_info["pct_long"] - cot_info["4w_avg"]) > 0.05:
            base_confidence += 10 if signal_data["signal"] == "up" else -10
        elif (cot_info["pct_long"] - cot_info["4w_avg"]) < -0.05:
            base_confidence += 10 if signal_data["signal"] == "down" else -10

    # 4. Ajuste final com limites e suavização
    signal_data["confidence"] = min(95, max(5, base_confidence))  # Limites 5-95%
    signal_data["cot_confidence_impact"] = signal_data["confidence"] - original_confidence  # Para análise/debug
        
        try:
            ml_prediction = self.ml.predict(data["symbol"], timeframe, data["history"])
            if ml_prediction and ml_prediction != signal_data["signal"]:
                print("⚠️ ML disagrees — downgrading confidence")
                signal_data["confidence"] = max(signal_data["confidence"] - 20, 10)
                signal_data["strength"] = "weak"
        except Exception as e:
            print(f"⚠️ ML predictor failed: {e}")

        return self.filter.apply(signal_data, data["history"])
