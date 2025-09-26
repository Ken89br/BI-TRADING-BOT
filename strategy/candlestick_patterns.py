# strategy/candlestick_patterns.py
# Padrões de candlestick completos com todas as velas da lista fornecida
import numpy as np
import pandas as pd

# Dicionário de força/confiança dos padrões atualizado
PATTERN_STRENGTH = {
    # Padrões de reversão
    "bullish_engulfing": 1.0,
    "bearish_engulfing": 1.0,
    "hammer": 0.8,
    "hanging_man": 0.8,
    "inverted_hammer": 0.8,
    "shooting_star": 0.8,
    "morning_star": 1.0,
    "evening_star": 1.0,
    "piercing_line": 0.7,
    "dark_cloud_cover": 0.7,
    "three_white_soldiers": 1.0,
    "three_black_crows": 1.0,
    "abandoned_baby_bullish": 1.0,
    "abandoned_baby_bearish": 1.0,
    "kicker_bullish": 0.8,
    "kicker_bearish": 0.8,
    
    # Padrões de continuação
    "rising_three_methods": 0.9,
    "falling_three_methods": 0.9,
    "upside_tasuki_gap": 0.6,
    "downside_tasuki_gap": 0.6,
    "separating_lines": 0.5,
    
    # Padrões neutros/indecisão
    "doji": 0.3,
    "dragonfly_doji": 0.3,
    "gravestone_doji": 0.3,
    "long_legged_doji": 0.3,
    "spinning_top": 0.3,
    "marubozu": 0.7,
    
    # Padrões híbridos
    "bullish_harami": 0.7,
    "bearish_harami": 0.7,
    "harami_cross": 0.6,
    "tweezer_bottom": 0.5,
    "tweezer_top": 0.5,
    "three_inside_up": 0.7,
    "three_inside_down": 0.7,
    "three_outside_up": 0.7,
    "three_outside_down": 0.7,
    
    # Gap patterns
    "gap_up": 0.4,
    "gap_down": 0.4,
    "on_neckline": 0.4,
    
    # Novos padrões adicionados
    "belt_hold_bullish": 0.7,
    "belt_hold_bearish": 0.7,
    "counterattack_bullish": 0.6,
    "counterattack_bearish": 0.6,
    "unique_three_river_bottom": 0.8,
    "breakaway_bullish": 0.7,
    "breakaway_bearish": 0.7
} 

def get_pattern_strength(patterns):
    """Retorna a soma das forças dos padrões detectados."""
    return sum(PATTERN_STRENGTH.get(p, 0.2) for p in patterns)

# ============== FUNÇÕES DE DETECÇÃO DE PADRÕES ==============

# --- Padrões de um candle ---
def is_doji(candle, threshold=0.1):
    body = abs(candle["close"] - candle["open"])
    full = candle["high"] - candle["low"] or 1e-8
    return body / full < threshold

def is_dragonfly_doji(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    full = candle["high"] - candle["low"] or 1e-8
    return is_doji(candle) and lower_wick > 2 * body and upper_wick < body

def is_gravestone_doji(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    full = candle["high"] - candle["low"] or 1e-8
    return is_doji(candle) and upper_wick > 2 * body and lower_wick < body

def is_long_legged_doji(candle):
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    return is_doji(candle) and upper_wick > 0 and lower_wick > 0

def is_spinning_top(candle):
    body = abs(candle["close"] - candle["open"])
    full = candle["high"] - candle["low"] or 1e-8
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    return 0.2 < body / full < 0.5 and upper_wick > 0 and lower_wick > 0

def is_hammer(candle):
    body = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    full = candle["high"] - candle["low"] or 1e-8
    return body / full < 0.3 and lower_wick > 2 * body and upper_wick < body

def is_hanging_man(candle):
    return is_hammer(candle)  # Formato igual, contexto diferente

def is_inverted_hammer(candle):
    body = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    full = candle["high"] - candle["low"] or 1e-8
    return body / full < 0.3 and upper_wick > 2 * body and lower_wick < body

def is_shooting_star(candle):
    return is_inverted_hammer(candle)  # Formato igual, contexto diferente

def is_marubozu(candle, threshold=0.02):
    body = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["close"], candle["open"])
    lower_wick = min(candle["close"], candle["open"]) - candle["low"]
    full = candle["high"] - candle["low"] or 1e-8
    return upper_wick / full < threshold and lower_wick / full < threshold

# --- Padrões de dois candles ---
def is_bullish_engulfing(candle, prev):
    return (
        candle["close"] > candle["open"] and
        prev["close"] < prev["open"] and
        candle["open"] < prev["close"] and
        candle["close"] > prev["open"]
    )

def is_bearish_engulfing(candle, prev):
    return (
        candle["close"] < candle["open"] and
        prev["close"] > prev["open"] and
        candle["open"] > prev["close"] and
        candle["close"] < prev["open"]
    )

def is_bullish_harami(candle, prev):
    return (
        prev["close"] < prev["open"] and
        candle["close"] > candle["open"] and
        candle["open"] > prev["close"] and
        candle["close"] < prev["open"]
    )

def is_bearish_harami(candle, prev):
    return (
        prev["close"] > prev["open"] and
        candle["close"] < candle["open"] and
        candle["open"] < prev["close"] and
        candle["close"] > prev["open"]
    )

def is_harami_cross(candle, prev):
    return is_doji(candle) and (is_bullish_harami(candle, prev) or is_bearish_harami(candle, prev))

def is_piercing_line(candle, prev):
    mid_prev = (prev["open"] + prev["close"]) / 2
    return (
        prev["close"] < prev["open"] and
        candle["open"] < prev["close"] and
        candle["close"] > mid_prev and
        candle["close"] < prev["open"]
    )

def is_dark_cloud_cover(candle, prev):
    mid_prev = (prev["open"] + prev["close"]) / 2
    return (
        prev["close"] > prev["open"] and
        candle["open"] > prev["close"] and
        candle["close"] < mid_prev and
        candle["close"] > prev["open"]
    )

def is_tweezer_bottom(candle, prev, threshold=0.1):
    return (
        prev["close"] < prev["open"] and
        candle["close"] > candle["open"] and
        abs(prev["low"] - candle["low"]) / (abs(prev["low"]) + 1e-8) < threshold
    )

def is_tweezer_top(candle, prev, threshold=0.1):
    return (
        prev["close"] > prev["open"] and
        candle["close"] < candle["open"] and
        abs(prev["high"] - candle["high"]) / (abs(prev["high"]) + 1e-8) < threshold
    )

def is_kicker_bullish(prev, candle):
    return (
        prev["close"] < prev["open"] and
        candle["open"] > prev["close"] and
        candle["close"] > candle["open"]
    )

def is_kicker_bearish(prev, candle):
    return (
        prev["close"] > prev["open"] and
        candle["open"] < prev["close"] and
        candle["close"] < candle["open"]
    )

def is_gap_up(prev, candle):
    return candle["low"] > prev["high"]

def is_gap_down(prev, candle):
    return candle["high"] < prev["low"]

def is_on_neckline(candle, prev, threshold=0.05):
    return (
        prev["close"] < prev["open"] and
        candle["open"] < prev["close"] and
        abs(candle["close"] - prev["low"]) / prev["low"] < threshold
    )

def is_separating_lines(prev, candle):
    return (
        prev["close"] < prev["open"] and
        candle["open"] == prev["open"] and
        candle["close"] > candle["open"]
    ) or (
        prev["close"] > prev["open"] and
        candle["open"] == prev["open"] and
        candle["close"] < candle["open"]
    )

# --- Padrões de três candles ---
def is_morning_star(prev2, prev1, candle):
    return (
        prev2["close"] < prev2["open"] and
        abs(prev1["close"] - prev1["open"]) < (prev2["open"] - prev2["close"]) * 0.5 and
        candle["close"] > candle["open"] and
        candle["close"] > ((prev2["close"] + prev2["open"]) / 2)
    )

def is_evening_star(prev2, prev1, candle):
    return (
        prev2["close"] > prev2["open"] and
        abs(prev1["close"] - prev1["open"]) < (prev2["close"] - prev2["open"]) * 0.5 and
        candle["close"] < candle["open"] and
        candle["close"] < ((prev2["close"] + prev2["open"]) / 2)
    )

def is_three_white_soldiers(candles):
    if len(candles) < 3:
        return False
    last = candles[-1]
    prev1 = candles[-2]
    prev2 = candles[-3]
    return (
        all(c["close"] > c["open"] for c in [prev2, prev1, last]) and
        prev2["close"] < prev1["open"] and
        prev1["close"] < last["open"]
    )

def is_three_black_crows(candles):
    if len(candles) < 3:
        return False
    last = candles[-1]
    prev1 = candles[-2]
    prev2 = candles[-3]
    return (
        all(c["close"] < c["open"] for c in [prev2, prev1, last]) and
        prev2["close"] > prev1["open"] and
        prev1["close"] > last["open"]
    )

def is_three_inside_up(candles):
    if len(candles) < 3:
        return False
    prev2 = candles[-3]
    prev1 = candles[-2]
    last = candles[-1]
    return (
        is_bearish_engulfing(prev1, prev2) and
        last["close"] > prev1["close"]
    )

def is_three_inside_down(candles):
    if len(candles) < 3:
        return False
    prev2 = candles[-3]
    prev1 = candles[-2]
    last = candles[-1]
    return (
        is_bullish_engulfing(prev1, prev2) and
        last["close"] < prev1["close"]
    )

def is_three_outside_up(candles):
    if len(candles) < 3:
        return False
    prev2 = candles[-3]
    prev1 = candles[-2]
    last = candles[-1]
    return (
        is_bullish_engulfing(prev1, prev2) and
        last["close"] > prev1["close"]
    )

def is_three_outside_down(candles):
    if len(candles) < 3:
        return False
    prev2 = candles[-3]
    prev1 = candles[-2]
    last = candles[-1]
    return (
        is_bearish_engulfing(prev1, prev2) and
        last["close"] < prev1["close"]
    )

def is_abandoned_baby_bullish(prev2, prev1, candle, threshold=0.001):
    return (
        prev2["close"] < prev2["open"] and
        is_doji(prev1) and
        prev1["low"] > prev2["high"] + threshold and
        candle["open"] > prev1["high"] + threshold and
        candle["close"] > candle["open"]
    )

def is_abandoned_baby_bearish(prev2, prev1, candle, threshold=0.001):
    return (
        prev2["close"] > prev2["open"] and
        is_doji(prev1) and
        prev1["high"] < prev2["low"] - threshold and
        candle["open"] < prev1["low"] - threshold and
        candle["close"] < candle["open"]
    )

def is_upside_tasuki_gap(candles):
    if len(candles) < 3:
        return False
    prev2, prev1, last = candles[-3], candles[-2], candles[-1]
    return (
        prev2["close"] > prev2["open"] and
        prev1["close"] > prev1["open"] and
        is_gap_up(prev2, prev1) and
        last["close"] < last["open"] and
        last["open"] > prev1["close"] and
        last["close"] > prev1["open"]
    )

def is_downside_tasuki_gap(candles):
    if len(candles) < 3:
        return False
    prev2, prev1, last = candles[-3], candles[-2], candles[-1]
    return (
        prev2["close"] < prev2["open"] and
        prev1["close"] < prev1["open"] and
        is_gap_down(prev2, prev1) and
        last["close"] > last["open"] and
        last["open"] < prev1["close"] and
        last["close"] < prev1["open"]
    )

# --- Padrões de cinco candles ---
def is_rising_three_methods(candles):
    if len(candles) < 5:
        return False
    a, b, c, d, e = candles[-5:]
    return (
        a["close"] > a["open"] and
        all(x["close"] < x["open"] for x in [b, c, d]) and
        e["close"] > e["open"] and
        e["close"] > a["close"]
    )

def is_falling_three_methods(candles):
    if len(candles) < 5:
        return False
    a, b, c, d, e = candles[-5:]
    return (
        a["close"] < a["open"] and
        all(x["close"] > x["open"] for x in [b, c, d]) and
        e["close"] < e["open"] and
        e["close"] < a["close"]
    )

def is_belt_hold_bullish(candle):
    """Belt Hold (Alakozakura) - Bullish"""
    body = candle["close"] - candle["open"]
    lower_wick = candle["open"] - candle["low"]
    return (
        body > 0 and
        lower_wick <= body * 0.1 and
        (candle["high"] - candle["close"]) <= body * 0.1
    )

def is_belt_hold_bearish(candle):
    """Belt Hold (Alakozakura) - Bearish"""
    body = candle["open"] - candle["close"]
    upper_wick = candle["high"] - candle["open"]
    return (
        body > 0 and
        upper_wick <= body * 0.1 and
        (candle["close"] - candle["low"]) <= body * 0.1
    )

def is_counterattack_bullish(prev, candle):
    """Counterattack - Bullish"""
    return (
        prev["close"] < prev["open"] and
        candle["open"] < prev["close"] and
        abs(candle["close"] - prev["open"]) < (prev["open"] - prev["close"]) * 0.1
    )

def is_counterattack_bearish(prev, candle):
    """Counterattack - Bearish"""
    return (
        prev["close"] > prev["open"] and
        candle["open"] > prev["close"] and
        abs(candle["close"] - prev["open"]) < (prev["close"] - prev["open"]) * 0.1
    )

def is_unique_three_river_bottom(candles):
    """Unique Three River Bottom"""
    if len(candles) < 3:
        return False
    prev2, prev1, last = candles[-3], candles[-2], candles[-1]
    return (
        prev2["close"] < prev2["open"] and
        is_hammer(prev1) and
        prev1["close"] < prev2["close"] and
        last["open"] > last["close"] and
        last["open"] < prev1["close"] and
        last["close"] > prev2["low"]
    )

def is_breakaway_bullish(candles):
    """Breakaway - Bullish"""
    if len(candles) < 5:
        return False
    a, b, c, d, e = candles[-5:]
    return (
        a["close"] < a["open"] and
        b["close"] < b["open"] and
        c["close"] < c["open"] and
        d["close"] > d["open"] and
        e["close"] > e["open"] and
        e["close"] > a["open"]
    )

def is_breakaway_bearish(candles):
    """Breakaway - Bearish"""
    if len(candles) < 5:
        return False
    a, b, c, d, e = candles[-5:]
    return (
        a["close"] > a["open"] and
        b["close"] > b["open"] and
        c["close"] > c["open"] and
        d["close"] < d["open"] and
        e["close"] < e["open"] and
        e["close"] < a["open"]
    )

# ============== DETECÇÃO PRINCIPAL ==============

def detect_candlestick_patterns(candles):
    """Detecta todos os padrões de candlestick na série de candles fornecida"""
    patterns = []
    l = len(candles)
    if l < 1:
        return patterns

    last = candles[-1]

    # Padrões de um candle
    if is_doji(last):
        patterns.append("doji")
    if is_dragonfly_doji(last):
        patterns.append("dragonfly_doji")
    if is_gravestone_doji(last):
        patterns.append("gravestone_doji")
    if is_long_legged_doji(last):
        patterns.append("long_legged_doji")
    if is_spinning_top(last):
        patterns.append("spinning_top")
    if is_hammer(last):
        patterns.append("hammer")
    if is_hanging_man(last):
        patterns.append("hanging_man")
    if is_inverted_hammer(last):
        patterns.append("inverted_hammer")
    if is_shooting_star(last):
        patterns.append("shooting_star")
    if is_marubozu(last):
        patterns.append("marubozu")
    if is_belt_hold_bullish(last):
        patterns.append("belt_hold_bullish")
    if is_belt_hold_bearish(last):
        patterns.append("belt_hold_bearish")

    # Padrões de dois candles
    if l >= 2:
        prev = candles[-2]
        if is_bullish_engulfing(last, prev):
            patterns.append("bullish_engulfing")
        if is_bearish_engulfing(last, prev):
            patterns.append("bearish_engulfing")
        if is_piercing_line(last, prev):
            patterns.append("piercing_line")
        if is_dark_cloud_cover(last, prev):
            patterns.append("dark_cloud_cover")
        if is_tweezer_bottom(last, prev):
            patterns.append("tweezer_bottom")
        if is_tweezer_top(last, prev):
            patterns.append("tweezer_top")
        if is_bullish_harami(last, prev):
            patterns.append("bullish_harami")
        if is_bearish_harami(last, prev):
            patterns.append("bearish_harami")
        if is_harami_cross(last, prev):
            patterns.append("harami_cross")
        if is_kicker_bullish(prev, last):
            patterns.append("kicker_bullish")
        if is_kicker_bearish(prev, last):
            patterns.append("kicker_bearish")
        if is_gap_up(prev, last):
            patterns.append("gap_up")
        if is_gap_down(prev, last):
            patterns.append("gap_down")
        if is_on_neckline(last, prev):
            patterns.append("on_neckline")
        if is_separating_lines(prev, last):
            patterns.append("separating_lines")
        if is_counterattack_bullish(prev, last):
            patterns.append("counterattack_bullish")
        if is_counterattack_bearish(prev, last):
            patterns.append("counterattack_bearish")

    # Padrões de três candles
    if l >= 3:
        prev2 = candles[-3]
        prev1 = candles[-2]
        if is_morning_star(prev2, prev1, last):
            patterns.append("morning_star")
        if is_evening_star(prev2, prev1, last):
            patterns.append("evening_star")
        if is_three_white_soldiers(candles[-3:]):
            patterns.append("three_white_soldiers")
        if is_three_black_crows(candles[-3:]):
            patterns.append("three_black_crows")
        if is_three_inside_up(candles[-3:]):
            patterns.append("three_inside_up")
        if is_three_inside_down(candles[-3:]):
            patterns.append("three_inside_down")
        if is_three_outside_up(candles[-3:]):
            patterns.append("three_outside_up")
        if is_three_outside_down(candles[-3:]):
            patterns.append("three_outside_down")
        if is_abandoned_baby_bullish(prev2, prev1, last):
            patterns.append("abandoned_baby_bullish")
        if is_abandoned_baby_bearish(prev2, prev1, last):
            patterns.append("abandoned_baby_bearish")
        if is_upside_tasuki_gap(candles[-3:]):
            patterns.append("upside_tasuki_gap")
        if is_downside_tasuki_gap(candles[-3:]):
            patterns.append("downside_tasuki_gap")
        if is_unique_three_river_bottom(candles[-3:]):
            patterns.append("unique_three_river_bottom")

    # Padrões de cinco candles
    if l >= 5:
        if is_rising_three_methods(candles[-5:]):
            patterns.append("rising_three_methods")
        if is_falling_three_methods(candles[-5:]):
            patterns.append("falling_three_methods")
        if is_breakaway_bullish(candles[-5:]):
            patterns.append("breakaway_bullish")
        if is_breakaway_bearish(candles[-5:]):
            patterns.append("breakaway_bearish")

    return patterns

REVERSAL_UP = [
    "hammer", "bullish_engulfing", "piercing_line", "morning_star", "tweezer_bottom",
    "bullish_harami", "kicker_bullish", "three_inside_up", "three_outside_up", "gap_up",
    "dragonfly_doji", "three_white_soldiers", "inverted_hammer", "belt_hold_bullish",
    "breakaway_bullish", "counterattack_bullish", "unique_three_river_bottom"
]
REVERSAL_DOWN = [
    "hanging_man", "bearish_engulfing", "dark_cloud_cover", "evening_star", "tweezer_top",
    "bearish_harami", "kicker_bearish", "three_inside_down", "three_outside_down", "gap_down",
    "gravestone_doji", "three_black_crows", "shooting_star", "belt_hold_bearish",
    "breakaway_bearish", "counterattack_bearish", "abandoned_baby_bearish"
]
TREND_UP = ["three_white_soldiers", "rising_three_methods", "upside_tasuki_gap", "separating_lines"]
TREND_DOWN = ["three_black_crows", "falling_three_methods", "downside_tasuki_gap", "separating_lines"]
NEUTRAL = ["doji", "dragonfly_doji", "gravestone_doji", "long_legged_doji", "spinning_top", "marubozu"]
HYBRID = ["harami_cross"

def get_dynamic_pattern_strength(patterns, candle, context=None):
    """
    Calcula o score dinâmico dos padrões com pesos institucionais avançados.
    Retorna: tuple (score, confidence) onde:
        - score: float (0-100) representando força do sinal
        - confidence: float (0-1) representando confiança no sinal
    """
    if context is None:
        context = {}
    
    score = 0
    confidence = 0.5  # base confidence
    
    for p in patterns:
        base = PATTERN_STRENGTH.get(p, 0.2)
        pattern_confidence = 1.0
        
        # 1. Fatores de Volume e Liquidez
        volume_factor = 1.0
        if candle.get("volume") and context.get("avg_volume"):
            volume_ratio = candle["volume"] / context["avg_volume"]
            if volume_ratio > 2.0:
                volume_factor = 1.3
                pattern_confidence *= 1.1
            elif volume_ratio > 1.5:
                volume_factor = 1.2
            elif volume_ratio < 0.7:
                volume_factor = 0.8
                pattern_confidence *= 0.9
        
        # 2. Alinhamento com Tendência
        trend_factor = 1.0
        if context.get("trend_strength"):
            strength, direction = context["trend_strength"]
            if p in REVERSAL_UP and direction == "up":
                trend_factor = 0.7
                pattern_confidence *= 0.8
            elif p in REVERSAL_DOWN and direction == "down":
                trend_factor = 0.7
                pattern_confidence *= 0.8
            elif p in CONTINUATION and direction == "up":
                trend_factor = 1.2
                pattern_confidence *= 1.1
        
        # 3. Níveis Estruturais
        level_factor = 1.0
        if context.get("price_position"):
            if (p in REVERSAL_UP and context["price_position"] == "strong_support") or \
               (p in REVERSAL_DOWN and context["price_position"] == "strong_resistance"):
                level_factor = 1.4
                pattern_confidence *= 1.3
        
        # 4. Volatilidade e Sessão
        volatility_factor = 1.0
        if context.get("volatility") == "high" and context.get("session") in ["London", "NY"]:
            if p in BIG_RANGE_PATTERNS:
                volatility_factor = 1.25
        elif context.get("volatility") == "low":
            volatility_factor = 0.85
        
        # 5. Fator de Confirmação
        confirmation_factor = 1.0
        if p in NEEDS_CONFIRMATION:
            if _has_confirmation(candle, context):
                confirmation_factor = 1.3
            else:
                confirmation_factor = 0.6
        
        # 6. Raridade do Padrão
        rarity_factor = 1.0
        if p in RARE_PATTERNS:
            rarity_factor = 1.35
            pattern_confidence *= 1.15
        
        # Cálculo final do score para este padrão
        pattern_score = base * volume_factor * trend_factor * level_factor * volatility_factor * confirmation_factor * rarity_factor
        score += pattern_score
        
        # Atualiza confiança geral (média ponderada)
        confidence = (confidence + pattern_confidence) / 2
    
    # Normaliza o score para 0-100
    score = min(max(score * 20, 0), 100)
    
    return score, confidence

def _calculate_trend_strength(window):
    """
    Retorna (força, direção) da tendência.
    força: "weak", "moderate", "strong"
    direção: "up", "down", "side"
    """
    closes = window['close']
    if len(closes) < 5:
        return ("weak", "side")
    ret = closes.iloc[-1] - closes.iloc[0]
    abs_ret = abs(ret)
    std = closes.std()
    if abs_ret > 2*std:
        strength = "strong"
    elif abs_ret > std:
        strength = "moderate"
    else:
        strength = "weak"
    if ret > 0:
        direction = "up"
    elif ret < 0:
        direction = "down"
    else:
        direction = "side"
    return (strength, direction)

def _get_price_position(df, idx):
    """
    Retorna a posição do preço em relação a suportes/resistências institucionais
    ("strong_support", "support", "neutral", "resistance", "strong_resistance").
    """
    price = df.iloc[idx]['close']
    # Exemplo simples: percentil em relação ao range da janela longa
    window = df.iloc[max(idx-20, 0):idx+1]
    low = window['low'].min()
    high = window['high'].max()
    pct = (price - low) / (high - low + 1e-8)
    if pct < 0.1:
        return "strong_support"
    elif pct < 0.3:
        return "support"
    elif pct > 0.9:
        return "strong_resistance"
    elif pct > 0.7:
        return "resistance"
    else:
        return "neutral"

def _detect_liquidity_zones(window):
    """
    Detecta zonas de liquidez baseadas em volume acima da média.
    """
    volumes = window['volume']
    threshold = volumes.mean() * 1.5
    zones = []
    for i in range(1, len(window)):
        if volumes.iloc[i] > threshold:
            zones.append({'idx': window.index[i], 'price': window['close'].iloc[i]})
    return zones

def _check_economic_calendar(ts):
    """
    Placeholder para calendário econômico: retorne eventos importantes aqui.
    """
    # Você pode plugar seu calendário, aqui é só exemplo:
    return []

def _calculate_rsi(series, period=14):
    """
    Calcula RSI simples institucional.
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50

def _analyze_order_flow(window):
    """
    Análise de fluxo de ordens institucional (simples: saldo de candles de volume).
    """
    # Exemplo: saldo de candles de alta/baixa no volume da janela
    vol_up = window[window['close'] > window['open']]['volume'].sum()
    vol_down = window[window['close'] < window['open']]['volume'].sum()
    if vol_up > vol_down * 1.3:
        return "buy_pressure"
    elif vol_down > vol_up * 1.3:
        return "sell_pressure"
    else:
        return "neutral"

def _has_confirmation(candle, context):
    """
    Exemplo simples: confirmação por volume acima do normal ou fechamento forte.
    """
    if candle.get("volume") and context.get("avg_volume"):
        if candle["volume"] > context["avg_volume"] * 1.5:
            return True
    if abs(candle.get("close", 0) - candle.get("open", 0)) > 0.7 * (candle.get("high", 0) - candle.get("low", 0) + 1e-8):
        return True
    return False

CONTINUATION = [
    "rising_three_methods", "falling_three_methods", "upside_tasuki_gap", "downside_tasuki_gap",
    "separating_lines", "three_inside_up", "three_inside_down", "three_outside_up", "three_outside_down"
]

BIG_RANGE_PATTERNS = [
    "marubozu", "three_white_soldiers", "three_black_crows", "belt_hold_bullish", "belt_hold_bearish",
    "breakaway_bullish", "breakaway_bearish"
]

NEEDS_CONFIRMATION = [
    "hammer", "hanging_man", "inverted_hammer", "shooting_star", "harami_cross", "doji",
    "spinning_top", "long_legged_doji"
]

RARE_PATTERNS = [
    "unique_three_river_bottom", "abandoned_baby_bullish", "abandoned_baby_bearish",
    "breakaway_bullish", "breakaway_bearish", "counterattack_bullish", "counterattack_bearish"
]

# Alias para compatibilidade
detect_patterns = detect_candlestick_patterns
