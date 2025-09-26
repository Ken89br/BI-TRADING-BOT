# messaging/telegram_bot.py
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from aiohttp import web

from config import CONFIG
from utils.signal_logger import log_signal
from utils.telegram_safe import safe_send
from strategy.train_model_historic import main as run_training

import pandas as pd
import os
from dotenv import load_dotenv

from datetime import datetime
import pytz

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração de fuso horário
TZ_MAPUTO = pytz.timezone("Africa/Maputo")

def to_maputo_time(dt_utc):
    """Converte datetime UTC para o fuso horário de Maputo"""
    if isinstance(dt_utc, pd.Timestamp):
        dt_utc = dt_utc.to_pydatetime()
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=pytz.UTC)
    return dt_utc.astimezone(TZ_MAPUTO)

load_dotenv()

# Estados do bot
class SignalState(StatesGroup):
    choosing_mode = State()
    choosing_timeframe = State()
    choosing_symbol = State()

# Variáveis globais
REGISTERED_USERS = set()
signal_context = {}
user_languages = {}

# Paginação de símbolos
SYMBOL_PAGES = [CONFIG["symbols"][:8], CONFIG["symbols"][8:]]
SYMBOL_PAGES_OTC = [CONFIG["otc_symbols"][:8], CONFIG["otc_symbols"][8:]]

def get_text(key, lang=None, chat_id=None):
    """Obtém texto traduzido com fallback para inglês"""
    if chat_id:
        lang = user_languages.get(chat_id, "en")
    lang = lang or "en"
    return CONFIG["languages"].get(lang, CONFIG["languages"]["en"]).get(key, key)

def menu_main(chat_id):
    """Menu principal com base no idioma do usuário"""
    lang = user_languages.get(chat_id, "en")
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton("📈 Start" if lang == "en" else "📈 Iniciar"))
    markup.add(KeyboardButton("/status"), KeyboardButton("/retrain"), KeyboardButton("/stop"))
    markup.add(KeyboardButton("/help"), KeyboardButton("/support"))
    markup.add(KeyboardButton("🌐 Language" if lang == "en" else "🌐 Idioma"))
    return markup

def menu_cancel(chat_id):
    """Menu de cancelamento com texto localizado"""
    lang = user_languages.get(chat_id, "en")
    markup = ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    markup.add(KeyboardButton("❌ Cancel" if lang == "en" else "❌ Cancelar"))
    return markup

class TelegramNotifier:
    def __init__(self, token, strategy, data_client):
        self.bot = Bot(token=token)
        self.dp = Dispatcher(self.bot, storage=MemoryStorage())
        self.strategy = strategy
        self.data_client = data_client
        self.mode_map = {}

        # Handler para comando /start
        @self.dp.message_handler(commands=["start"])
        async def start_cmd(msg: types.Message):
            try:
                chat_id = msg.chat.id
                REGISTERED_USERS.add(chat_id)
                await safe_send(
                    self.bot, 
                    chat_id, 
                    get_text("start", chat_id=chat_id), 
                    reply_markup=menu_main(chat_id)
                )
            except Exception as e:
                logger.exception(f"Error in /start handler: {e}")

        # Handler para comando /help
        @self.dp.message_handler(lambda msg: msg.text.lower() in ["/help", "help"])
        async def help_cmd(msg: types.Message):
            try:
                await safe_send(self.bot, msg.chat.id, get_text("help", chat_id=msg.chat.id))
            except Exception as e:
                logger.exception(f"Error in /help handler: {e}")

        # Handler para comando /support
        @self.dp.message_handler(lambda msg: msg.text.lower() in ["/support", "support"])
        async def support_cmd(msg: types.Message):
            try:
                await safe_send(
                    self.bot, 
                    msg.chat.id, 
                    f"🛟 {get_text('support_contact', chat_id=msg.chat.id)} {CONFIG['support']['username']}"
                )
            except Exception as e:
                logger.exception(f"Error in /support handler: {e}")

        # Handler para comando /stop
        @self.dp.message_handler(lambda msg: msg.text.lower() in ["/stop", "stop"])
        async def stop_cmd(msg: types.Message, state: FSMContext):
            try:
                await state.finish()
                await safe_send(
                    self.bot, 
                    msg.chat.id, 
                    get_text("stopped", chat_id=msg.chat.id), 
                    reply_markup=menu_main(msg.chat.id)
                )
            except Exception as e:
                logger.exception(f"Error in /stop handler: {e}")

        # Handler para comando /status
        @self.dp.message_handler(lambda msg: msg.text.lower() in ["/status", "status"])
        async def status_cmd(msg: types.Message):
            try:
                user_id = msg.chat.id
                sym_info = signal_context.get(user_id)
                if sym_info:
                    response = get_text("bot_running", chat_id=user_id).format(
                        timeframe=sym_info['timeframe'], 
                        symbol=sym_info['symbol']
                    )
                else:
                    response = get_text("bot_running_no_ctx", chat_id=user_id)
                await safe_send(self.bot, user_id, response, parse_mode="Markdown")
            except Exception as e:
                logger.exception(f"Error in /status handler: {e}")

        # Handler para comando /retrain
        @self.dp.message_handler(lambda msg: msg.text.lower() in ["/retrain", "retrain"])
        async def retrain_force(msg: types.Message):
            try:
                await safe_send(
                    self.bot, 
                    msg.chat.id, 
                    get_text("force_retraining", chat_id=msg.chat.id)
                )
                run_training()
            except Exception as e:
                logger.exception(f"Error in /retrain handler: {e}")

        # Handler para alternar idioma
        @self.dp.message_handler(lambda msg: msg.text in ["🌐 Language", "🌐 Idioma"])
        async def toggle_lang(msg: types.Message):
            try:
                chat_id = msg.chat.id
                current_lang = user_languages.get(chat_id, "en")
                new_lang = "pt" if current_lang == "en" else "en"
                user_languages[chat_id] = new_lang
                await safe_send(
                    self.bot, 
                    chat_id, 
                    get_text("language_set", chat_id=chat_id), 
                    reply_markup=menu_main(chat_id)
                )
            except Exception as e:
                logger.exception(f"Error in language toggle handler: {e}")

        # Handler para iniciar sinal
        @self.dp.message_handler(lambda msg: msg.text in ["📈 Start", "📈 Iniciar"], state="*")
        async def start_signal(msg: types.Message):
            try:
                await SignalState.choosing_mode.set()
                kb = InlineKeyboardMarkup(row_width=2)
                kb.add(
                    InlineKeyboardButton("🌍 CoreFX", callback_data="mode:normal"),
                    InlineKeyboardButton("🕒 OTC", callback_data="mode:otc")
                )
                await safe_send(
                    self.bot, 
                    msg.chat.id, 
                    get_text("choose_mode", chat_id=msg.chat.id), 
                    reply_markup=kb
                )
                # Envia menu de cancelamento separadamente
                await safe_send(
                    self.bot,
                    msg.chat.id,
                    " ",
                    reply_markup=menu_cancel(msg.chat.id)
                )
            except Exception as e:
                logger.exception(f"Error in start_signal handler: {e}")

        # Handler para cancelamento
        @self.dp.message_handler(lambda msg: msg.text in ["❌ Cancel", "❌ Cancelar"], state="*")
        async def cancel_any(msg: types.Message, state: FSMContext):
            try:
                await state.finish()
                await safe_send(
                    self.bot, 
                    msg.chat.id, 
                    get_text("cancelled", chat_id=msg.chat.id),
                    reply_markup=menu_main(msg.chat.id)
                )
            except Exception as e:
                logger.exception(f"Error in cancel_any handler: {e}")

        # Handler para callback de escolha de modo
        @self.dp.callback_query_handler(lambda c: c.data.startswith("mode:"), state=SignalState.choosing_mode)
        async def select_mode(callback: types.CallbackQuery, state: FSMContext):
            try:
                mode = callback.data.split(":")[1]
                self.mode_map[callback.from_user.id] = mode
                await state.set_state(SignalState.choosing_timeframe.state)
                kb = InlineKeyboardMarkup(row_width=3)
                buttons = [InlineKeyboardButton(tf, callback_data=f"timeframe:{tf}") for tf in CONFIG["timeframes"]]
                kb.add(*buttons)
                kb.add(InlineKeyboardButton(get_text("back", chat_id=callback.from_user.id), callback_data="back_choose_mode"))
                await callback.message.edit_text(get_text("choose_timeframe", chat_id=callback.from_user.id), reply_markup=kb)
                await callback.answer()
                await safe_send(self.bot, callback.from_user.id, " ", chat_id=callback.from_user.id, reply_markup=menu_cancel(callback.from_user.id))
            except Exception as e:
                logger.exception(f"Error in select_mode handler: {e}")

        # Handler para callback de retorno ao menu principal
        @self.dp.callback_query_handler(lambda c: c.data == "back_mainmenu", state="*")
        async def back_main_menu(callback: types.CallbackQuery, state: FSMContext):
            try:
                await state.finish()
                chat_id = callback.from_user.id
                await callback.message.edit_text(get_text("main_menu", chat_id=chat_id))
                await safe_send(self.bot, chat_id, get_text("start", chat_id=chat_id), reply_markup=menu_main(chat_id))
            except Exception as e:
                logger.exception(f"Error in back_main_menu handler: {e}")

        # Handler para callback de retorno à escolha de modo
        @self.dp.callback_query_handler(lambda c: c.data == "back_choose_mode", state=SignalState.choosing_timeframe)
        async def back_choose_mode(callback: types.CallbackQuery, state: FSMContext):
            try:
                await state.set_state(SignalState.choosing_mode.state)
                kb = InlineKeyboardMarkup(row_width=2)
                kb.add(
                    InlineKeyboardButton("🌍 CoreFX", callback_data="mode:normal"),
                    InlineKeyboardButton("🕒 OTC", callback_data="mode:otc")
                )
                await callback.message.edit_text(get_text("choose_mode", chat_id=callback.from_user.id), reply_markup=kb)
                await callback.answer()
                await safe_send(self.bot, callback.from_user.id, " ", reply_markup=menu_cancel(callback.from_user.id))
            except Exception as e:
                logger.exception(f"Error in back_choose_mode handler: {e}")

        # Handler para callback de escolha de timeframe
        @self.dp.callback_query_handler(lambda c: c.data.startswith("timeframe:"), state=SignalState.choosing_timeframe)
        async def select_timeframe(callback: types.CallbackQuery, state: FSMContext):
            try:
                tf = callback.data.split(":")[1]
                await state.update_data(timeframe=tf)
                await state.set_state(SignalState.choosing_symbol.state)
                await self.send_symbol_buttons(callback.message, callback.from_user.id, page=0)
                await callback.answer()
                await safe_send(self.bot, callback.from_user.id, " ", chat_id=callback.from_user.id, reply_markup=menu_cancel(callback.from_user.id))
            except Exception as e:
                logger.exception(f"Error in select_timeframe handler: {e}")

        # Handler para callback de próxima página de símbolos
        @self.dp.callback_query_handler(lambda c: c.data == "more_symbols", state=SignalState.choosing_symbol)
        async def next_symbols(callback: types.CallbackQuery, state: FSMContext):
            try:
                await self.send_symbol_buttons(callback.message, callback.from_user.id, page=1)
                await callback.answer()
            except Exception as e:
                logger.exception(f"Error in next_symbols handler: {e}")

        # Handler para callback de retorno à seleção de timeframe
        @self.dp.callback_query_handler(lambda c: c.data == "back_symbols", state=SignalState.choosing_symbol)
        async def back_symbols(callback: types.CallbackQuery, state: FSMContext):
            try:
                await state.set_state(SignalState.choosing_timeframe.state)
                kb = InlineKeyboardMarkup(row_width=3)
                buttons = [InlineKeyboardButton(tf, callback_data=f"timeframe:{tf}") for tf in CONFIG["timeframes"]]
                kb.add(*buttons)
                kb.add(InlineKeyboardButton(get_text("back", chat_id=callback.from_user.id), callback_data="back_choose_mode"))
                await callback.message.edit_text(get_text("choose_timeframe", chat_id=callback.from_user.id), reply_markup=kb)
                await callback.answer()
                await safe_send(self.bot, callback.from_user.id, " ", chat_id=callback.from_user.id, reply_markup=menu_cancel(callback.from_user.id))
            except Exception as e:
                logger.exception(f"Error in back_symbols handler: {e}")

        # Handler para callback de seleção de símbolo
        @self.dp.callback_query_handler(lambda c: c.data.startswith("symbol:"), state=SignalState.choosing_symbol)
        async def select_symbol(callback: types.CallbackQuery, state: FSMContext):
            try:
                await callback.answer()
                symbol = callback.data.split(":")[1].replace(" OTC", "")
                await state.update_data(symbol=symbol)
                user_data = await state.get_data()
                timeframe = user_data["timeframe"]
                kb = InlineKeyboardMarkup()
                kb.add(InlineKeyboardButton(get_text("back", chat_id=callback.from_user.id), callback_data="back_symbols"))
                progress_bar = get_text("progress_generating", chat_id=callback.from_user.id)
                await callback.message.edit_text(
                    f"⏱ {get_text('timeframe', chat_id=callback.from_user.id)}: `{timeframe}`\n"
                    f"💱 {get_text('pair', chat_id=callback.from_user.id)}: `{symbol}`\n\n"
                    f"{get_text('generating', chat_id=callback.from_user.id)}\n{progress_bar}",
                    parse_mode="Markdown",
                    reply_markup=kb
                )
                candles = self.data_client.fetch_candles(symbol, interval=self._map_timeframe(timeframe))
                if not candles or "history" not in candles:
                    await safe_send(self.bot, callback.from_user.id, get_text("failed_price_data", chat_id=callback.from_user.id), reply_markup=menu_main(callback.from_user.id))
                    return

                # Usa sempre os campos dinâmicos vindos do ensemble
                signal_data = self.strategy.generate_signal(candles, timeframe=self._map_timeframe(timeframe))
                if not signal_data:
                    await safe_send(self.bot, callback.from_user.id, get_text("no_signal", chat_id=callback.from_user.id), reply_markup=menu_main(callback.from_user.id))
                else:
                    signal_context[callback.from_user.id] = {"symbol": symbol, "timeframe": timeframe}
                    await self.send_trade_signal(callback.from_user.id, symbol, signal_data)
                await state.finish()
                await safe_send(self.bot, callback.from_user.id, get_text("start", chat_id=callback.from_user.id), reply_markup=menu_main(callback.from_user.id))
            except Exception as e:
                logger.exception(f"Error in select_symbol handler: {e}")

        # Handler para callback de refresh de sinal
        @self.dp.callback_query_handler(lambda c: c.data == "refresh_signal")
        async def refresh(callback: types.CallbackQuery):
            try:
                uid = callback.from_user.id
                await callback.answer()
                if uid not in signal_context:
                    await callback.answer(get_text("no_previous_signal", chat_id=uid), show_alert=True)
                    return
                ctx = signal_context[uid]
                candles = self.data_client.fetch_candles(ctx["symbol"], interval=self._map_timeframe(ctx["timeframe"]))
                if not candles or "history" not in candles:
                    await safe_send(self.bot, uid, get_text("no_signal", chat_id=uid))
                    return
                signal_data = self.strategy.generate_signal(candles, timeframe=self._map_timeframe(ctx["timeframe"]))
                if signal_data:
                    await self.send_trade_signal(uid, ctx["symbol"], signal_data)
            except Exception as e:
                logger.exception(f"Error in refresh handler: {e}")

    async def send_symbol_buttons(self, message, user_id, page=0):
        """Mostra os símbolos disponíveis com paginação"""
        kb = InlineKeyboardMarkup(row_width=2)
        symbols = SYMBOL_PAGES_OTC[page] if self.mode_map.get(user_id) == "otc" else SYMBOL_PAGES[page]
        buttons = [InlineKeyboardButton(sym, callback_data=f"symbol:{sym}") for sym in symbols]
        kb.add(*buttons)
        if page == 0:
            kb.add(InlineKeyboardButton("➡️ " + get_text("more", chat_id=user_id), callback_data="more_symbols"))
        kb.add(InlineKeyboardButton(get_text("back", chat_id=user_id), callback_data="back_symbols"))
        await message.edit_text(get_text("choose_symbol", chat_id=message.chat.id), reply_markup=kb)

    def _map_timeframe(self, tf):
        """Mapeia o timeframe do usuário para o formato da API"""
        return {
            "S1": "s1", "M1": "1min", "M5": "5min", "M15": "15min",
            "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"
        }.get(tf, "1min")

    async def send_trade_signal(self, chat_id, asset, signal_data):
        """Envia mensagem rica de sinal no Telegram"""
        signal_data["symbol"] = asset
        signal_data["user"] = chat_id
        signal_data["timestamp"] = to_maputo_time(pd.Timestamp.utcnow()).strftime("%Y-%m-%d %H:%M:%S")
        signal_data["recommend_entry"] = signal_data.get("recommended_entry_time")

        # ======== LOTE BASEADO NA CONFIANÇA ==========
        try:
            confidence = float(signal_data.get("confidence", 0))
            confidence = max(0, min(confidence, 100))
        except Exception:
            confidence = 0

        min_lot = 1
        max_lot = 100
        lot_size = min_lot + (max_lot - min_lot) * (confidence / 100)
        lot_size = round(lot_size)
        lot_display = f"${lot_size}"

        signal_data["lot_size"] = lot_size
        signal_data["lot_display"] = lot_display
        # =============================================

        log_signal(chat_id, asset, signal_data.get("timeframe"), signal_data)

        SIGNAL_CSV_PATH = "signals.csv"
        df = pd.DataFrame([signal_data])
        if os.path.exists(SIGNAL_CSV_PATH):
            df.to_csv(SIGNAL_CSV_PATH, mode="a", header=False, index=False)
        else:
            df.to_csv(SIGNAL_CSV_PATH, index=False)

        payout = round(signal_data.get('price', 0) * 0.92, 5)
        patterns_list = signal_data.get("patterns", [])
        patterns_str = ", ".join([get_text(p, chat_id=chat_id) for p in patterns_list]) if patterns_list else "-"

        par = asset
        timer = signal_data.get('timer', get_text('timer', chat_id=chat_id))
        direction = get_text(signal_data.get('signal', '').lower(), chat_id=chat_id)
        strength = get_text(signal_data.get('strength', '-').lower(), chat_id=chat_id)
        confidence_disp = signal_data.get('confidence', '-')
        entry = signal_data.get('recommended_entry_price', signal_data.get('price', '-'))
        recommend_entry = signal_data.get('recommended_entry_time', '-')
        expire_entry = signal_data.get('expire_entry_time', '-')
        expire_entry_price = signal_data.get('expire_entry_price', entry)
        high = signal_data.get('high', '-')
        low = signal_data.get('low', '-')
        volume = signal_data.get('volume', '-')
        volatility = signal_data.get('volatility', '-')
        sentiment = signal_data.get('sentiment', '-')
        variation = signal_data.get('variation', '-')
        risk = signal_data.get('risk', get_text('low_risk', chat_id=chat_id))
        support = signal_data.get('support', '-')
        resistance = signal_data.get('resistance', '-')
        summary = get_text(signal_data.get('summary', '-').lower(), chat_id=chat_id)
        ma = get_text(signal_data.get('moving_averages', '-').lower(), chat_id=chat_id)
        osc = get_text(signal_data.get('oscillators', '-').lower(), chat_id=chat_id)
        rsi = signal_data.get('rsi', '-')
        macd = signal_data.get('macd', '-')
        bollinger = signal_data.get('bollinger', '-')
        atr = signal_data.get('atr', '-')
        adx = signal_data.get('adx', '-')
        volume_status = get_text(signal_data.get('volume_status', '-').lower(), chat_id=chat_id)

        # Mensagem rica: hora/expiração/preço sempre dinâmicos vindos do ensemble
        msg = (
            f"📡 *{get_text('signal_title', chat_id=chat_id)}*\n\n"
            f"📌 *{get_text('pair', chat_id=chat_id)}:* `{par}`\n"
            f"⏱ *{get_text('timer', chat_id=chat_id)}:* {timer}\n"
            f"🕒 *{get_text('recommend_entry', chat_id=chat_id)}:* `{recommend_entry} (entry price: {entry})`\n"
            f"⏳ *{get_text('expire_entry', chat_id=chat_id)}:* `{expire_entry} (entry price: {expire_entry_price})`\n"
            f"💵 *{get_text('lot_size', chat_id=chat_id)}:* `{lot_display}`\n\n"
            f"📈 *{get_text('direction', chat_id=chat_id)}:* `{direction}`\n"
            f"💪 *{get_text('strength', chat_id=chat_id)}:* `{strength}`\n"
            f"🎯 *{get_text('confidence', chat_id=chat_id)}:* `{confidence_disp}%`\n"
            f"💰 *{get_text('entry', chat_id=chat_id)}:* `{entry}`\n"
            f"📈 *{get_text('high', chat_id=chat_id)}:* `{high}`\n"
            f"📉 *{get_text('low', chat_id=chat_id)}:* `{low}`\n"
            f"📦 *{get_text('volume', chat_id=chat_id)}:* `{volume}`\n"
            f"💸 *{get_text('payout', chat_id=chat_id)}:* `{payout}`\n\n"
            f"__*{get_text('market_overview', chat_id=chat_id)}*__\n"
            f"• {get_text('volatility', chat_id=chat_id)}: *{volatility}*\n"
            f"• {get_text('sentiment', chat_id=chat_id)}: *{sentiment}*\n"
            f"• {get_text('variation', chat_id=chat_id)}: *{variation}*\n"
            f"• {get_text('risk', chat_id=chat_id)}: *{risk}*\n"
            f"• {get_text('support', chat_id=chat_id)}: `{support}`\n"
            f"• {get_text('resistance', chat_id=chat_id)}: `{resistance}`\n\n"
            f"__*{get_text('tradingview_rating', chat_id=chat_id)}*__\n"
            f"• {get_text('summary', chat_id=chat_id)}: *{summary}*\n"
            f"• {get_text('moving_averages', chat_id=chat_id)}: *{ma}*\n"
            f"• {get_text('oscillators', chat_id=chat_id)}: *{osc}*\n\n"
            f"__*{get_text('technical_analysis', chat_id=chat_id)}*__\n"
            f"• RSI: *{rsi}*\n"
            f"• MACD: *{macd}*\n"
            f"• {get_text('bollinger_bands', chat_id=chat_id)}: *{bollinger}*\n"
            f"• {get_text('atr', chat_id=chat_id)}: *{atr}*\n"
            f"• {get_text('adx', chat_id=chat_id)}: *{adx}*\n"
            f"• {get_text('patterns', chat_id=chat_id)}: *{patterns_str}*\n"
            f"• {get_text('volume_status', chat_id=chat_id)}: *{volume_status}*\n"
        )

        keyboard = InlineKeyboardMarkup()
        keyboard.add(InlineKeyboardButton(
            "🔁 " + get_text("refresh", chat_id=chat_id), 
            callback_data="refresh_signal"
        ))
        await safe_send(
            self.bot, 
            chat_id, 
            msg, 
            parse_mode="Markdown", 
            reply_markup=keyboard
        )

    @property
    def token(self):
        return CONFIG["telegram"]["bot_token"]

    async def set_webhook(self):
        """Configura webhook para integração com Telegram"""
        await self.bot.set_webhook(f"{CONFIG['webhook']['url']}/webhook/{self.token}")

    async def webhook_handler(self, request: web.Request):
        """Handler para receber updates do Telegram via webhook"""
        update = types.Update(**(await request.json()))
        Bot.set_current(self.bot)
        Dispatcher.set_current(self.dp)
        await self.dp.process_update(update)
        return web.Response()
