from strategy.feature_universal import LOOKBACK_UNIVERSAL

CONFIG = {
    "ken_config": {
        "max_lookahead_candles": 20,    # Quantos candles procurar para prever o melhor ponto de entrada (lookahead)
        "min_expiry_candles": 1,       # Mínimo de candles para expiração dinâmica
        "max_expiry_candles": 20,       # Máximo de candles para expiração dinâmica
        "default_expiry_candles": 20,   # Valor padrão se não houver critério claro
        #Padrões e contextos dinâmicos
        "CANDLE_PATTERN_LOOKBACK_SMALL": 5,
        "CANDLE_PATTERN_LOOKBACK_MEDIUM": 15,
        "CANDLE_PATTERN_LOOKBACK_LARGE": 50,
    },

    "min_train_rows": 50,
    
    # WICK REVERSAL STRATEGY CONFIGURATION
    "wick_reversal": {
        "lookback": LOOKBACK_UNIVERSAL,
        "wick_ratio": 2.0,
        "min_body_ratio": 0.1,
        "volume_multiplier": 1.5,
        "trend_confirmation": True
    },
    
    # MACD REVERSAL STRATEGY CONFIGURATION
    "macd_reversal": {
        "lookback": LOOKBACK_UNIVERSAL,
        "fast": 10,
        "slow": 21,
        "signal": 7,
        "threshold": 0.15
    },
    
    # RSI MA STRATEGY CONFIGURATION
    "rsi_ma": {
        "lookback": LOOKBACK_UNIVERSAL,
        "rsi_period": 10,
        "ma_period": 3,
        "overbought": 70,
        "oversold": 30,
        "confirmation": True,
        "volume_threshold": 1.5
    },
    
    # EMA STRATEGY CONFIGURATION
    "ema": {
        "lookback": LOOKBACK_UNIVERSAL,
        "short_period": 9,
        "long_period": 21,
        "candle_lookback": 3,
        "pattern_boost": 0.2,
        "min_confidence": 70
    },
    
    # PRICE ACTION STRATEGY CONFIGURATION
    "price_action": {
        "lookback": LOOKBACK_UNIVERSAL,
        "min_wick_ratio": 2.5,
        "volume_multiplier": 2.0,
        "confirmation": True,
        "trend_lookback": 5
    },

    # ADX STRATEGY CONFIGURATION
    "adx": {
        "lookback": LOOKBACK_UNIVERSAL,
        "adx_period": 14,
        "di_period": 14,
        "adx_threshold": 20,
        "min_history": 20,
        "candle_lookback": 3,
        "pattern_boost": 0.2,
        "min_confidence": 70,
        "trend_confirmation": True,
        "volume_threshold": 1.3,
    },
    # ATR INDICATOR CONFIGURATION
    "atr_config": {
        "lookback": LOOKBACK_UNIVERSAL,
        "period_min": 5,
        "period_max": 300,
        "volume_threshold": 1.5,
        "min_confidence": 70,
        "position_risk_pct": 0.01,
        "risk_reward_ratio": 1.5,
        # Parâmetros dinâmicos por timeframe
        "timeframe_adjustments": {
            "1m":  {"atr_multiplier": 0.9, "rsi_period": 10, "ema_short": 15,  "ema_long": 50},
            "5m":  {"atr_multiplier": 1.0, "rsi_period": 12, "ema_short": 20,  "ema_long": 70},
            "15m": {"atr_multiplier": 1.1, "rsi_period": 14, "ema_short": 25,  "ema_long": 100},
            "1h":  {"atr_multiplier": 1.3, "rsi_period": 16, "ema_short": 30,  "ema_long": 150},
            "4h":  {"atr_multiplier": 1.5, "rsi_period": 18, "ema_short": 40,  "ema_long": 200},
            "1d":  {"atr_multiplier": 2.0, "rsi_period": 20, "ema_short": 50,  "ema_long": 300}
        }
    },
    
    # BOLLINGER STRATEGY CONFIGURATION
    "bbands": {
        "lookback": LOOKBACK_UNIVERSAL,
        "period": 20,   
        "std_dev": 2.0,
        "candle_lookback": 3,
        "min_confidence": 75,
        "pattern_boost": 0.22,
        "min_history": 25,
        "volume_threshold": 1.5,         # Opcional, se usado no seu código
        "signal_direction_filter": "both", # 'up', 'down' ou 'both'
        "allow_neutral_signals": False,
    },
    # BOLLINGER BREAKOUT STRATEGY
    "bollinger_breakout": {
        "period": 20,                # Período da média móvel (padrão clássico)
        "std_dev": 2.0,              # Desvio padrão para bandas (2.0 é o padrão técnico)
        "candle_lookback": 3,        # Candles para buscar padrões de vela
        "pattern_boost": 0.2,        # Boost de confiança por padrão detectado
        "min_confidence": 70,        # Confiança mínima para sinal
        "min_history": 23,           # Histórico mínimo (período + lookback)
        "volume_threshold": 1.5,     # Opcional: volume acima da média para validar sinal
        "signal_direction_filter": "both", # 'call', 'put' ou 'both' (filtra por direção se desejar)
        "allow_neutral_signals": False,    # Permite sinais neutros? (normalmente False)
    },
    
    # RSI STRATEGY CONFIGURATION
    "rsi": {
        "lookback": LOOKBACK_UNIVERSAL,
        "overbought": 70,
        "oversold": 30,
        "window": 14,
        "confirmation": True,
        "trend_filter": True,
        "volume_threshold": 1.3,
        "candle_lookback": 25,
        "min_confidence": 70,
        "enable_pattern_boost": True

        #não configurado ainda (preciso confirmar com o copilot.
        "rsi_period", 14,
        "smooth_period", 3,
        "atr_period", 14,
        "vol_window", 20,
        "min_confidence", 65,
        "pattern_boost", 0.2),
        "volume_multiplier", 1.5),
        "divergence_lookback", 14)
    },
    
