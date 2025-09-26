# config.py
import os
import logging

def get_env(key, default=None, required=False):
    val = os.getenv(key, default)
    if required and val is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val

CONFIG = {
    "telegram": {
        "enabled": True,
        "bot_token": get_env("TELEGRAM_BOT_TOKEN", required=True),
        "chat_id": get_env("TELEGRAM_CHAT_ID"),
        "admin_id": get_env("TELEGRAM_ADMIN_ID"),
    },

    "support": {
        "username": "@kenbreu"
    },

    "webhook": {
        "url": get_env("WEBHOOK_URL")
    },

    # ✅ Regular Forex Pairs
    "symbols": [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD",
        "USDCAD", "EURJPY", "EURNZD", "AEDCNY", "AUDCAD", "AUDCHF",
        "AUDNZD", "AUDUSD", "CADJPY", "CHFJPY", "EURGBP", "EURJPY"
    ],

    # ✅ OTC Pairs
    "otc_symbols": [
        "EURUSD OTC", "GBPUSD OTC", "USDJPY OTC", "AUDUSD OTC", "EURJPY OTC",
        "NZDUSD OTC", "AUDCAD OTC", "AUDCHF OTC", "GBPJPY OTC", "CADJPY OTC"
    ],

    "timeframes": ["S30", "M1", "M2", "M3", "M5", "M10", "M15", "M30", "H1", "H2", "H4", "D1"],

    "log_level": "INFO",
    
    "languages": {
        "en": {
            "start": "Welcome! Tap 📈 Start to generate a signal.",
            "choose_mode": "🧭 Choose trading mode:",
            "choose_timeframe": "⏱ Choose a timeframe:",
            "choose_symbol": "💱 Choose a currency pair:",
            "generating": "📡 Generating signal...",
            "progress_generating": "⏳ Please wait, analyzing the market...",
            "no_signal": "⚠️ No signal at this moment.",
            "signal_title": "📡 New Forex Signal!",
            "pair": "Pair",
            "direction": "Direction",
            "strength": "Strength",
            "confidence": "Confidence",
            "entry": "Entry Price",
            "recommend_entry": "Recommended Entry",
            "expire_entry": "Expires At",
            "high": "High",
            "low": "Low",
            "lot_size": "Order size",
            "volume": "Volume",
            "payout": "Simulated Payout (92%)",
            "timer": "",
            "refresh": "Refresh",
            "main_menu": "Main menu. Tap 📈 Start to generate a signal.",
            "back": "Back",
            "more": "More",
            "failed_price_data": "⚠️ Failed to retrieve price data.",
            "error": "Error",
            "no_previous_signal": "⚠️ No previous signal to refresh.",
            "bot_running": "✅ Bot is running.\n\n🕐 Timeframe: `{timeframe}`\n💱 Symbol: `{symbol}`",
            "bot_running_no_ctx": "✅ Bot is running.\nℹ️ No signal context found. Use 📈 Start to begin.",
            "force_retraining": "🔁 Force retraining initiated (manual override).",
            "language_set": "🌐 Language set to English ✅",
            "support_contact": "Contact support:",
            # Directions
            "up": "HIGHER",
            "down": "LOWER",
            "neutral": "NEUTRAL",
            # Strengths
            "strong": "Strong",
            "weak": "Weak",

            # ====== NEW FOR SIGNAL RICH MESSAGE ======
            "forecast": "Forecast",
            "variation": "Variation",
            "risk": "Risk",
            "low_risk": "Low risk",
            "market_overview": "Market Overview",
            "volatility": "Volatility",
            "sentiment": "Sentiment",
            "market_snapshot": "Market Snapshot",
            "current_value": "Current Value",
            "support": "Support (S1)",
            "resistance": "Resistance (R1)",
            "tradingview_rating": "TradingView Rating",
            "summary": "Summary",
            "moving_averages": "Moving Averages",
            "oscillators": "Oscillators",
            "technical_analysis": "Technical Analysis",
            "bollinger_bands": "Bollinger Bands",
            "atr": "ATR",
            "adx": "ADX",
            "patterns": "Patterns",
            "volume_status": "Volume Status",
        },
        
        "pt": {
            "start": "Bem-vindo! Toque 📈 Start para gerar um sinal.",
            "choose_mode": "🧭 Escolha o modo de negociação:",
            "choose_timeframe": "⏱ Escolha o timeframe:",
            "choose_symbol": "💱 Escolha o par de moedas:",
            "generating": "📡 Gerando sinal...",
            "progress_generating": "⏳ Aguarde, analisando o mercado...",
            "no_signal": "⚠️ Nenhum sinal neste momento.",
            "signal_title": "📡 Novo Sinal Forex!",
            "pair": "Par",
            "direction": "Direção",
            "strength": "Força",
            "confidence": "Confiança",
            "entry": "Preço de Entrada",
            "recommend_entry": "Entrada Recomendada",
            "expire_entry": "Expira em",
            "high": "Alta",
            "low": "Baixa",
            "lot_size": "Ordem (lote)",
            "volume": "Volume",
            "payout": "Lucro Simulado (92%)",
            "timer": "",
            "refresh": "Atualizar",
            "main_menu": "Menu principal. Toque 📈 Start para gerar um sinal.",
            "back": "Voltar",
            "more": "Mais",
            "failed_price_data": "⚠️ Falha ao obter dados de preço.",
            "error": "Erro",
            "no_previous_signal": "⚠️ Nenhum sinal anterior para atualizar.",
            "bot_running": "✅ Bot em execução.\n\n🕐 Timeframe: `{timeframe}`\n💱 Par: `{symbol}`",
            "bot_running_no_ctx": "✅ Bot em execução.\nℹ️ Nenhum contexto de sinal encontrado. Use 📈 Iniciar para começar.",
            "force_retraining": "🔁 Retreinamento forçado iniciado (sob demanda).",
            "language_set": "🌐 Idioma definido para Português ✅",
            "support_contact": "Contato do suporte:",
            # Direções
            "up": "ALTA",
            "down": "BAIXA",
            "neutral": "NEUTRO",
            # Força do sinal
            "strong": "Forte",
            "weak": "Fraco",

            # ====== NOVOS CAMPOS PARA MENSAGEM RICA ======
            "forecast": "Previsão",
            "variation": "Variação",
            "risk": "Risco",
            "low_risk": "Baixo risco",
            "market_overview": "Visão de Mercado",
            "volatility": "Volatilidade",
            "sentiment": "Sentimento",
            "market_snapshot": "Resumo de Mercado",
            "current_value": "Valor Atual",
            "support": "Suporte (S1)",
            "resistance": "Resistência (R1)",
            "tradingview_rating": "Rating TradingView",
            "summary": "Resumo",
            "moving_averages": "Médias Móveis",
            "oscillators": "Osciladores",
            "technical_analysis": "Análise Técnica",
            "bollinger_bands": "Bandas de Bollinger",
            "atr": "ATR",
            "adx": "ADX",
            "patterns": "Padrões",
            "volume_status": "Status do Volume",
            # Padrões (adicione todos os do seu sistema de detecção)
            "bullish_engulfing": "Engolfo de Alta",
            "bearish_engulfing": "Engolfo de Baixa",
            "piercing_line": "Linha de Penetração",
            "dark_cloud_cover": "Nuvem Negra",
            "tweezer_bottom": "Pinça de Fundo",
            "tweezer_top": "Pinça de Topo",
            "bullish_harami": "Harami de Alta",
            "bearish_harami": "Harami de Baixa",
            "kicker_bullish": "Kicker de Alta",
            "kicker_bearish": "Kicker de Baixa",
            "on_neckline": "No Pescoço",
            "separating_lines": "Linhas Separadoras",
            "gap_up": "Gap de Alta",
            "gap_down": "Gap de Baixa",
            "doji": "Doji",
            "dragonfly_doji": "Doji Libélula",
            "gravestone_doji": "Doji Lápide",
            "long_legged_doji": "Doji Pernas Longas",
            "spinning_top": "Pião",
            "hammer": "Martelo",
            "inverted_hammer": "Martelo Invertido",
            "shooting_star": "Estrela Cadente",
            "hanging_man": "Homem Enforcado",
            "marubozu": "Marubozu",
            "morning_star": "Estrela da Manhã",
            "evening_star": "Estrela da Noite",
            "three_inside_up": "Três Dentro de Alta",
            "three_inside_down": "Três Dentro de Baixa",
            "three_outside_up": "Três Fora de Alta",
            "three_outside_down": "Três Fora de Baixa",
            "upside_tasuki_gap": "Tasuki Gap de Alta",
            "downside_tasuki_gap": "Tasuki Gap de Baixa",
            "three_white_soldiers": "Três Soldados Brancos",
            "three_black_crows": "Três Corvos Negros",
        }
    }
}

logging.basicConfig(level=getattr(logging, CONFIG["log_level"].upper(), logging.INFO))
