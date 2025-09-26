from typing import Dict, Optional
from datetime import datetime
import requests

class TelegramSender:
    # Configurações padrão (sem vírgula no final)
    bot_token = "123456789:AAFm2WbJxXyYzZzZzZzZzZzZzZzZzZzZzZzZ"
    chat_id = "-100987654321"
    
    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        # Sobrescreve os defaults se novos valores forem fornecidos
        self.bot_token = bot_token or self.bot_token
        self.chat_id = chat_id or self.chat_id
        
        # Verificação flexível das credenciais
        if not self.bot_token and not self.chat_id:
            raise ValueError("❌ Erro: Nenhuma credencial fornecida. Defina pelo menos bot_token ou chat_id")
        elif not self.bot_token:
            print("⚠️ Aviso: bot_token não definido - Mensagens não serão enviadas")
        elif not self.chat_id:
            print("⚠️ Aviso: chat_id não definido - Mensagens não serão enviadas")

    def enviar_sinal(self, sinal: Dict[str, str]) -> bool:
        """
        :param sinal: Dicionário com chaves:
            - 'resposta': Texto da análise
            - 'par': Par negociado
            - 'timeframe': Timeframe
            - 'volatilidade': Nível de volatilidade
            - 'estrategia': Tipo de estratégia usada
            - 'timestamp': Horário da análise
        :return: True se o envio foi bem-sucedido
        """
        # Verifica se tem credenciais mínimas para enviar
        if not self.bot_token or not self.chat_id:
            print("⏭️ Mensagem não enviada - Credenciais incompletas")
            return False
            
        if not all(key in sinal for key in ['resposta', 'par', 'timeframe']):
            print("[ERRO] Dicionário de sinal inválido.")
            return False

        try:
            # Formatação melhorada da mensagem
            texto = (
                f"🚨 **SINAL CONFIRMADO** 🚨\n"
                f"🔄 **Par**: {sinal['par']}\n"
                f"⏳ **Timeframe**: {sinal['timeframe']}\n"
                f"📊 **Volatilidade**: {sinal.get('volatilidade', 'N/A')}\n"
                f"🎯 **Estratégia**: {sinal.get('estrategia', 'Padrão')}\n"
                f"🕒 **Horário**: {datetime.utcnow().strftime('%H:%M')} UTC\n\n"
                f"📌 **Análise Técnica**\n"
                f"{sinal['resposta']}\n\n"
                f"⚠️ **Gerenciamento de Risco**\n"
                f"- Opere com no máximo 2% do capital por sinal\n"
                f"- Confirme sempre no timeframe superior\n"
                f"- Valide spreads antes de entrar\n\n"
                f"#Forex #Binárias #Trading"
            )
            
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": texto,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            # Feedback detalhado no console
            if response.status_code == 200:
                print(f"✅ Sinal enviado com sucesso para o chat {self.chat_id}")
                return True
            else:
                error_msg = response.json().get('description', 'Erro desconhecido')
                print(f"❌ Falha ao enviar sinal (Status {response.status_code}): {error_msg}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"🔴 Erro de conexão com o Telegram: {str(e)}")
            return False
        except Exception as e:
            print(f"⚠️ Erro inesperado: {str(e)}")
            return False
