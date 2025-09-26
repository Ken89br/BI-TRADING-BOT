#serve para importar os dados do candlestick_patterns.py


# Estratégia universalizada para detecção de padrões de vela via DataFrame universal de features
from config_strategy import CONFIG

class CandlestickStrategy:
    def generate_signal(self, features_df):
        # Use os padrões do contexto small (reversão), medium (tendência), ou combine
        last = features_df.iloc[-1]
        # Combine padrões de todos contextos se quiser maior robustez
        patterns = set(last.get("patterns_small", []) + last.get("patterns_medium", []) + last.get("patterns_large", []))
        for pattern in patterns:
            if pattern in CONFIG["candlestick_patterns"]["reversal_up"]:
                return {"signal": "up", "pattern": pattern}
            if pattern in CONFIG["candlestick_patterns"]["reversal_down"]:
                return {"signal": "down", "pattern": pattern}
            if pattern in CONFIG["candlestick_patterns"]["neutral"]:
                return {"signal": "neutral", "pattern": pattern}
        return None
