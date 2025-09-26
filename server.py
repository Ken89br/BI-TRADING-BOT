# server.py
import os
import logging
import asyncio
from aiohttp import web
from data.data_client import FallbackDataClient
from strategy.ensemble_strategy import EnsembleStrategy
from messaging.telegram_bot import TelegramNotifier
from config import CONFIG

# Logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def get_env_or_config(key, default=None):
    # Busca em env, depois em CONFIG (dict), depois default
    return os.environ.get(key, CONFIG.get(key, default))

async def healthcheck(request):
    return web.Response(text="ok", status=200)

async def init_app():
    try:
        data_client = FallbackDataClient()
        strategy = EnsembleStrategy()
        notifier = TelegramNotifier(CONFIG["telegram"]["bot_token"], strategy, data_client)

        app = web.Application()
        app.router.add_post(f"/webhook/{notifier.token}", notifier.webhook_handler)
        app.router.add_get("/health", healthcheck)

        # Seta webhook só se necessário
        await notifier.set_webhook()

        logger.info(f"Webhook endpoint ativo em: /webhook/{notifier.token}")
        logger.info("Bot inicializado com sucesso.")
        return app
    except Exception as exc:
        logger.error(f"Erro ao iniciar o Bot: {exc}", exc_info=True)
        raise

if __name__ == "__main__":
    host = get_env_or_config("HOST", "0.0.0.0")
    port = int(get_env_or_config("PORT", 10000))
    logger.info(f"Iniciando servidor em http://{host}:{port}")
    try:
        web.run_app(init_app(), host=host, port=port)
    except Exception as exc:
        logger.critical(f"Servidor encerrado por erro fatal: {exc}", exc_info=True)
