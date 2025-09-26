# utils/telegram_safe.py
from aiogram.utils.exceptions import BotBlocked

def remove_user(chat_id):
    from messaging.telegram_bot import REGISTERED_USERS, signal_context, user_languages
    REGISTERED_USERS.discard(chat_id)
    signal_context.pop(chat_id, None)
    user_languages.pop(chat_id, None)
    print(f"🗑️ Removed blocked user {chat_id}")

async def safe_send(bot, chat_id, text, **kwargs):
    try:
        return await bot.send_message(chat_id, text, **kwargs)
    except BotBlocked:
        print(f"❌ BotBlocked error for user {chat_id}")
        remove_user(chat_id)
        return None
