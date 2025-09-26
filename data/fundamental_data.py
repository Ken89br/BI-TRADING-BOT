#data/fundamental_data.py
import pandas as pd

def get_cot_feature(symbol, dt=None):
    """
    Placeholder: Retorna o posicionamento dos grandes players para um símbolo.
    Você precisa baixar/processar o COT da CFTC (csv) e mapear para seu símbolo.
    """
    # Exemplo de retorno fictício
    return 0  # 0 = neutro, 1 = net long, -1 = net short

def get_macro_feature(symbol, dt=None):
    """
    Placeholder: Integração com calendário econômico, retorna 1 (evento de alto impacto), 0 (neutro), -1 (evento negativo)
    """
    return 0

def get_sentiment_feature(symbol, dt=None):
    """
    Placeholder: Análise de sentimento de notícias (NLP).
    """
    return 0
