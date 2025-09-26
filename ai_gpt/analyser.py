import pandas as pd
from datetime import datetime
import openai
import time
import os
import numpy as np
import random
from typing import Dict, List, Optional, Tuple
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import requests
from bs4 import BeautifulSoup
from xgboost import XGBClassifier
from data.data_client import FallbackDataClient
from messager.gpt_signal import TelegramSender
from support_resistance import AdvancedSRAnalyzer as SupportResistanceAnalyzer
from sklearn.metrics import accuracy_score

# === CONFIGURAÇÕES ===
OPENAI_KEYS = [
    "sk-COLE_SUA_CHAVE_AQUI_1",
    "sk-COLE_SUA_CHAVE_AQUI_2",
    "sk-COLE_SUA_CHAVE_AQUI_3"
]

openai.api_key = OPENAI_KEYS[0]
MODEL = "gpt-4o"
WINDOW_SIZE_DYNAMIC = {
    "30s": 20,     # Apenas para OTC
    "1min": 30,
    "2min": 50,
    "5min": 75,
    "10min": 90,
    "15min": 100,
    "30min": 150,
    "1H": 200,
    "4H": 300
}
MAX_TOKENS = 500
TEMPERATURE = 0.18  # Menor = mais determinístico

telegram = TelegramSender()

# ✅ Pares regulares e OTC
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD",
    "USDCAD", "EURJPY", "EURNZD", "AEDCNY", "AUDCAD", "AUDCHF",
    "AUDNZD", "CADJPY", "CHFJPY", "EURGBP"
]

OTC_SYMBOLS = [
    "EURUSD OTC", "GBPUSD OTC", "USDJPY OTC", "AUDUSD OTC", "EURJPY OTC",
    "NZDUSD OTC", "AUDCAD OTC", "AUDCHF OTC", "GBPJPY OTC", "CADJPY OTC"
]

# ✅ Timeframes dinâmicos (ordenados por agressividade)
TIMEFRAMES = ["30s", "1min", "2min", "5min", "10min", "15min", "30min", "1H", "4H"]

# ✅ Horários de operação em UTC
HORARIOS_UTC = ["08:30", "10:30", "12:30", "15:30", "20:30"]

# Estratégias disponíveis
ESTRATEGIAS = {
    'CONSERVADORA': {
        'MM': [5, 21],
        'RSI': 14,
        'STOCH': (14, 3, 3),
        'CCI': 20,
        'ADX_THRESHOLD': 25,
        'RISCO': 'baixo'
    },
    'AGGRESSIVA': {
        'MM': [3, 9],
        'RSI': 9,
        'STOCH': (5, 3, 3),
        'CCI': 14,
        'ADX_THRESHOLD': 20,
        'RISCO': 'alto'
    },
    'HIBRIDA': {
        'MM': [5, 14],
        'RSI': 11,
        'STOCH': (8, 3, 3),
        'CCI': 18,
        'ADX_THRESHOLD': 22,
        'RISCO': 'medio'
    }
}

class MLModel:
    def __init__(self):
        self.model = None
        self.features = [
            'MM5', 'MM14', 'MM21', 'ATR', 'BB_width', 
            'RSI', 'MACD', 'Volume_Change', 'ADX', 'OBV'
        ]
        self.target = 'Signal'
        self.model_path = 'model/ml_signal_v2.joblib'
        os.makedirs('model', exist_ok=True)
        
    def carregar_modelo(self):
        """Carrega modelo treinado ou cria novo"""
        try:
            self.model = joblib.load(self.model_path)
            print("✅ Modelo ML carregado")
        except:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                eval_metric='logloss'
            )
            print("⚠️ Novo modelo XGBoost criado")

    def preparar_dados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados históricos para treino com mais indicadores"""
        df['Volume_Change'] = df['Volume'].pct_change() * 100
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'] = talib.MACD(df['Close'])[0]
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['Signal'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        return df.dropna()

    def treinar(self, dados_historicos: pd.DataFrame):
        """Treina o modelo com novos dados"""
        df = self.preparar_dados(dados_historicos)
        X = df[self.features]
        y = df[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Avaliação
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        joblib.dump(self.model, self.model_path)
        print(f"🔁 Modelo retreinado | Acurácia: {accuracy:.2%}")

    def prever(self, dados_atual: dict) -> Tuple[float, dict]:
        """Retorna probabilidade e metadados"""
        if not self.model:
            return 0.5, {'model': 'none', 'confidence': 0}
        
        input_data = pd.DataFrame([dados_atual])
        proba = self.model.predict_proba(input_data)[0][1]
        confidence = 1 - abs(proba - 0.5) * 2  # Mapeia 0.5→0, 0→1, 1→1
        
        return proba, {
            'model': 'xgboost',
            'confidence': confidence,
            'features': {k: v for k, v in dados_atual.items() if k in self.features}
        }

ml_model = MLModel()
ml_model.carregar_modelo()
data_client = FallbackDataClient()
sr_analyzer = AdvancedSRAnalyzer(data_client)

# === Fallback de chaves OpenAI ===
def set_next_openai_key():
    current = OPENAI_KEYS.index(openai.api_key)
    next_index = (current + 1) % len(OPENAI_KEYS)
    openai.api_key = OPENAI_KEYS[next_index]
    print(f"🔄 Alternando para próxima OpenAI key: {next_index + 1}/{len(OPENAI_KEYS)}")

def calcular_medias_moveis(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula apenas as MMs mais relevantes"""
    return {
        "MM5": df['Close'].rolling(window=5).mean().iloc[-1],
        "MM14": df['Close'].rolling(window=14).mean().iloc[-1],
        "MM21": df['Close'].rolling(window=21).mean().iloc[-1],
        "MM200": df['Close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
    }

def calcular_bollinger(df: pd.DataFrame, window: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """Calcula Bandas de Bollinger com parâmetros dinâmicos"""
    rolling_mean = df['Close'].rolling(window=window).mean()
    rolling_std = df['Close'].rolling(window=window).std()
    return {
        "BB_upper": (rolling_mean + (rolling_std * std_dev)).iloc[-1],
        "BB_middle": rolling_mean.iloc[-1],
        "BB_lower": (rolling_mean - (rolling_std * std_dev)).iloc[-1],
        "BB_width": ((rolling_mean + (rolling_std * std_dev)) - (rolling_mean - (rolling_std * std_dev))).iloc[-1],
        "BB_std_dev": std_dev
    }

def calcular_atr(df: pd.DataFrame, window: int = 14) -> float:
    """Calcula o Average True Range (ATR)"""
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean().iloc[-1]

# Dicionário de padrões
PATTERNS = {
    # Reversão de Baixa
    -100: "Engolfo Baixa",
    -200: "Três Corvos",
    -300: "Estrela Cadente",
    -400: "Nuvem Negra",
    # Reversão de Alta
    100: "Engolfo Alta",
    200: "Três Soldados",
    300: "Martelo",
    400: "Estrela da Manhã",
    # Indecisão
    0: "Doji",
    50: "Spinning Top",
    150: "Harami"
}

def detectar_padroes(df: pd.DataFrame) -> Tuple[str, float]:
    """Identifica padrões de candles com todos os recursos do TA-Lib"""
    padroes_detectados = []
    
    # Mapeamento de funções TA-Lib para padrões
    padroes_config = {
        # Reversão de Baixa
        talib.CDLENGULFING: ("Engolfo Baixa", -1),
        talib.CDL3BLACKCROWS: ("Três Corvos", -1),
        talib.CDLDARKCLOUDCOVER: ("Nuvem Negra", -1),
        talib.CDLSHOOTINGSTAR: ("Estrela Cadente", -1),
        # Reversão de Alta
        talib.CDLENGULFING: ("Engolfo Alta", 1),
        talib.CDL3WHITESOLDIERS: ("Três Soldados", 1),
        talib.CDLHAMMER: ("Martelo", 1),
        talib.CDLMORNINGSTAR: ("Estrela da Manhã", 1),
        # Indecisão
        talib.CDLDOJI: ("Doji", 0),
        talib.CDLSPINNINGTOP: ("Spinning Top", 0),
        talib.CDLHARAMI: ("Harami", 0)
    }
    
    for func, (nome, direcao) in padroes_config.items():
        resultado = func(df['Open'], df['High'], df['Low'], df['Close'])
        padroes = [(i, nome) for i, val in enumerate(resultado) if (
            (direcao == 1 and val > 0) or 
            (direcao == -1 and val < 0) or
            (direcao == 0 and val != 0)
        )]
        
        for idx, padrao in padroes[-5:]:
            forca = abs(resultado[idx])
            if forca > 50:  # Filtro de qualidade
                padroes_detectados.append((nome, forca, df.iloc[idx]['Date']))
    
    # Ordena por força e remove duplicados
    padroes_detectados.sort(key=lambda x: x[1], reverse=True)
    padroes_unicos = []
    nomes_vistos = set()
    
    for padrao in padroes_detectados:
        if padrao[0] not in nomes_vistos:
            padroes_unicos.append(f"{padrao[0]} (força: {padrao[1]}, horário: {padrao[2]})")
            nomes_vistos.add(padrao[0])
    
    return " | ".join(padroes_unicos) if padroes_unicos else "Nenhum padrão significativo"

# GAPS
def detectar_gaps(df: pd.DataFrame) -> str:
    """Detecta gaps de preço (Superman Pattern)"""
    gaps = []
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i-1]
        current_open = df['Open'].iloc[i]
        
        # Gap de Alta (Superman Compra)
        if current_open > prev_close * 1.002:  # 0.2% acima
            gaps.append(f"GAP_ALTA: {prev_close:.5f} → {current_open:.5f}")
        
        # Gap de Baixa (Superman Venda)
        elif current_open < prev_close * 0.998:  # 0.2% abaixo
            gaps.append(f"GAP_BAIXA: {prev_close:.5f} → {current_open:.5f}")
    
    return " | ".join(gaps[-3:]) if gaps else "Sem gaps relevantes"

def calcular_indicadores_avancados(df: pd.DataFrame, estrategia: str) -> Dict[str, float]:
    """Calcula indicadores com parâmetros dinâmicos"""
    # Validação de dados
    if df.isnull().values.any():
        df = df.dropna()
        if len(df) == 0:
            raise ValueError("Dados insuficientes após limpeza")
                    
    config = ESTRATEGIAS[estrategia]
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    return {
        # Médias Móveis
        f"MM{config['MM'][0]}": talib.SMA(close, timeperiod=config['MM'][0])[-1],
        f"MM{config['MM'][1]}": talib.SMA(close, timeperiod=config['MM'][1])[-1],
        
        # Osciladores
        'RSI': talib.RSI(close, timeperiod=config['RSI'])[-1],
        'STOCH_K': talib.STOCH(high, low, close, *config['STOCH'])[0][-1],
        'STOCH_D': talib.STOCH(high, low, close, *config['STOCH'])[1][-1],
        'CCI': talib.CCI(high, low, close, timeperiod=config['CCI'])[-1],
        
        # Outros
        'ADX': talib.ADX(high, low, close, timeperiod=14)[-1],
        'ATR': talib.ATR(high, low, close, timeperiod=14)[-1],
        'VOLUME': volume[-1] / np.mean(volume[-20:])
}
    
def verificar_noticias() -> bool:
    """Verifica se há notícias importantes no calendário econômico"""
    try:
        agora = datetime.utcnow()
        # Filtra apenas horários de trading intenso
        if 12 <= agora.hour < 16:  # Horário de Londres/NY
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get('https://www.forexfactory.com/calendar', headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            eventos = soup.find_all(class_='calendar__row--highlight')
            return len(eventos) > 0
        return False
    except Exception as e:
        print(f"⚠️ Erro ao verificar notícias: {e}")
        return False

def analisar_condicoes_mercado(df: pd.DataFrame) -> dict:
    """Analisa múltiplos fatores para determinar a estratégia ideal"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Calcula indicadores chave
    atr = talib.ATR(high, low, close, timeperiod=14)[-1]
    adx = talib.ADX(high, low, close, timeperiod=14)[-1]
    rsi = talib.RSI(close, timeperiod=14)[-1]
    volatility = talib.STDDEV(close, timeperiod=20)[-1] / close[-1]
    
    # Calcula volume relativo
    volume_ratio = volume[-1] / np.mean(volume[-20:])
    
    # Determina perfil do mercado
    conditions = {
        'volatilidade': 'alta' if volatility > 0.003 else 'baixa' if volatility < 0.001 else 'moderada',
        'tendencia': 'forte' if adx > 25 else 'fraca',
        'volume': 'alto' if volume_ratio > 1.3 else 'baixo' if volume_ratio < 0.7 else 'normal',
        'hora': datetime.utcnow().hour
    }
    
    return conditions
    
def selecionar_estrategia(conditions: dict) -> str:
    """Seleciona a estratégia baseado nas condições de mercado"""
    hora = conditions['hora']
    
    # Horário noturno (mais conservador)
    if 0 <= hora < 6:
        return 'CONSERVADORA'
    
    # Horário Londres/NY (mais agressivo)
    if 8 <= hora < 16:
        if conditions['volatilidade'] == 'alta' and conditions['volume'] == 'alto':
            return 'AGGRESSIVA' if conditions['tendencia'] == 'forte' else 'HIBRIDA'
        return 'HIBRIDA'
    
    # Horário de transição
    return 'CONSERVADORA' if conditions['volatilidade'] == 'baixa' else 'HIBRIDA'

# === Função para formatar candles ===
def formatar_candles(df_block: pd.DataFrame, interval: str) -> str:
    linhas = ["Data\tOpen\tHigh\tLow\tClose\tVolume"]
    for _, row in df_block.iterrows():
        linhas.append(f"{row['Date']}\t{row['Open']}\t{row['High']}\t{row['Low']}\t{row['Close']}\t{row['Volume']}")
    
    # Cálculo de volatilidade
    atr = calcular_atr(df_block)
    atr_pct = (atr / df_block['Close'].iloc[-1]) * 100
    
    # Ajuste dinâmico dos parâmetros
    if atr_pct < 0.1:  # Mercado muito parado
        bb_std_dev = 1.5
        mm_filter = "MM5/MM14 (modo conservador)"
    elif atr_pct > 0.3:  # Mercado muito volátil
        bb_std_dev = 2.5
        mm_filter = "MM9/MM21 (modo agressivo)"
    else:  # Mercado normal
        bb_std_dev = 2.0
        mm_filter = "MM5/MM14 + MM9/MM21"
    
    mms = calcular_medias_moveis(df_block)
    bb = calcular_bollinger(df_block, std_dev=bb_std_dev)
    
    linhas.append(f"\n### VOLATILIDADE (ATR 14): {atr:.5f} ({atr_pct:.2f}%)")
    linhas.append(f"\n### CONFIGURAÇÃO DINÂMICA:")
    linhas.append(f"Modo: {'Conservador' if atr_pct < 0.1 else 'Agressivo' if atr_pct > 0.3 else 'Normal'}")
    linhas.append(f"Bollinger STD: {bb_std_dev}")
    linhas.append(f"Filtro MMs: {mm_filter}")
    
    linhas.append(f"\n### MÉDIAS MÓVEIS:")
    linhas.append(f"MM5: {mms['MM5']:.5f} | MM14: {mms['MM14']:.5f}")
    linhas.append(f"MM9: {mms['MM9']:.5f} | MM21: {mms['MM21']:.5f}")
    linhas.append(f"MM50: {mms['MM50']:.5f} | MM200: {mms['MM200']:.5f if mms['MM200'] else 'N/A'}")
    
    linhas.append(f"\n### BANDAS BOLLINGER (20,{bb_std_dev}):")
    linhas.append(f"Superior: {bb['BB_upper']:.5f} | Média: {bb['BB_middle']:.5f}")
    linhas.append(f"Inferior: {bb['BB_lower']:.5f} | Largura: {bb['BB_width']:.5f}")
    
    padrões = detectar_padroes(df_block)
    linhas.append(f"\n### PADRÕES DE CANDLES (TA-Lib):\n{padrões}")

    gaps = detectar_gaps(df_block)
    linhas.append(f"\n### GAPS (SUPERMAN):\n{gaps}")

    return "\n".join(linhas)

# === Seleção dinâmica de par e timeframe ===
def selecionar_par_tempo() -> Dict[str, str]:
    agora = datetime.utcnow().hour
    par = random.choice(OTC_SYMBOLS if random.random() < 0.3 else SYMBOLS)
    
    # Lógica para OTC (permite 30s)
    if "OTC" in par:
        if 7 <= agora < 11:  # Horário mais volátil
            timeframe = random.choice(["30s", "1min", "2min"])
        else:
            timeframe = random.choice(["1min", "2min", "5min"])
    else:
        if 7 <= agora < 11:
            timeframe = random.choice(["1min", "2min", "5min"])
        else:
            timeframe = random.choice(["5min", "15min", "30min"])
    
    return {"par": par, "timeframe": timeframe}

# === Prompt base ===
def montar_prompt(candles_text: str, symbol: str, interval: str, atr_pct: float, estrategia: str) -> str:
    window = WINDOW_SIZE_DYNAMIC.get(interval, 50)
    
    # Validade dinâmica baseada no timeframe e volatilidade
    base_validity = {
        "30s": "1-3min",
        "1min": "2-5min",
        "2min": "3-7min",
        "5min": "5-15min",
        "10min": "10-20min",
        "15min": "15-30min",
        "30min": "30-60min",
        "1H": "1-2H",
        "4H": "4-8H"
    }.get(interval, "15-30min")
    
    # Ajuste de validade pela volatilidade
    if atr_pct < 0.1:
        validity = f"{base_validity.split('-')[0]}-{int(base_validity.split('-')[1])*2}"  # Dobra a validade
    elif atr_pct > 0.3:
        validity = f"{int(base_validity.split('-')[0])//2}-{base_validity.split('-')[1]}"  # Metade do tempo
    else:
        validity = base_validity
    
    # Definir estratégia principal baseada na volatilidade
    if atr_pct < 0.1:
        main_strategy = "MM5/MM14 + Confirmação Bollinger Apertado"
    elif atr_pct > 0.3:
        main_strategy = "MM9/MM21 + Filtro MM200"
    else:
        main_strategy = "MM5/MM14 + MM9/MM21"
    
    return f"""
Você é um analista especialista em **opções binárias e Forex**, focado em scalping/day trade.
Sua tarefa é analisar os últimos {window} candles do par {symbol} (timeframe: {interval}) e gerar um sinal **COM CONFIRMAÇÃO TÉCNICA** de alta precisão.

### 📊 DADOS PARA ANÁLISE (ÚLTIMOS {window} CANDLES):
{candles_text}

### 🔍 ESTRATÉGIA ATIVA: {main_strategy}
1. **Tendência com Médias Móveis**:
   - Timeframe Atual ({interval}):
     → COMPRA CALL: Preço > MM5 > MM14 {'> MM9 > MM21' if atr_pct >= 0.1 else ''}
     → COMPRA PUT: Preço < MM5 < MM14 {'< MM9 < MM21' if atr_pct >= 0.1 else ''}
     → MM200 como filtro absoluto

2. **Bandas de Bollinger**:
   - {f"STD={1.5 if atr_pct < 0.1 else 2.5 if atr_pct > 0.3 else 2.0}"}
   - Compressão (BB_width < {'0.001' if interval in ['1min','2min'] else '0.002'}):
     → Possível rompimento iminente
   - Preço fora das bandas → Potencial reversão

3. **Indicadores Chave**:
   - RSI(14): 
     → CALL: Ideal 40-60 (evitar >65)
     → PUT: Ideal 40-60 (evitar <35)
   - ATR(14): 
     → Mínimo 0.002 para operar
     → SL = {'3xATR' if atr_pct > 0.3 else '1.5xATR'}

4. **Volume**:
   - Acima da média: +20% para confirmação
   - Volume reduzido: Aguardar confirmação

5. **Hierarquia de Padrões** (ordem de prioridade):
   1. Engolfing (força > 90)
   2. Estrela da Manhã/Estrela Cadente
   3. Três Corvos/Soldados
   4. Nuvem Negra
   5. Martelos/Doji com volume +30%
   
   Regras:
   - Padrões precisam estar alinhados com:
     → Tendência MMs (5/14/21)
     → Bandas Bollinger
   - Volume mínimo: +20% da média
   - Horário: Priorizar padrões nas últimas 2 velas

6. **Padrão Superman (Gaps)**:
   - Gap > 0.2%: Operar apenas na direção do gap
   - Gap < -0.2%: Cancelar operações contra

7. **Confirmação por IA**:
   - Probabilidade ML > 70% → Alta confiança
   - Probabilidade ML < 50% → Ignorar sinal

    ### 🔍 ESTRATÉGIA SELECIONADA: {estrategia}
    Parâmetros ativos:
    - Médias Móveis: {ESTRATEGIAS[estrategia]['MM']}
    - RSI({ESTRATEGIAS[estrategia]['RSI']})
    - Stoch({','.join(map(str, ESTRATEGIAS[estrategia]['STOCH']))})
    - CCI({ESTRATEGIAS[estrategia]['CCI']})
    - ADX Threshold: {ESTRATEGIAS[estrategia]['ADX_THRESHOLD']

### 📉 RESPOSTA EXIGIDA (FORMATO RÍGIDO):
Sinal: [COMPRA CALL | COMPRA PUT | AGUARDAR]
Horário: {datetime.utcnow().strftime('%H:%M')} UTC
Timeframe: {interval}
Validade: {validity}
Confiança: [1-5]  # 5 = Máxima confirmação
Stop Loss: [Preço]  # {'3x' if atr_pct > 0.3 else '1.5x'} ATR(14)
Gatilho: [Condição clara de entrada]
Justificativa: [Análise técnica concisa]

### ❗ REGRAS ABSOLUTAS:
- SEM alinhamento multi-timeframe → "AGUARDAR"
- RSI em extremos (≤35 ou ≥65) → Não operar contra
- Spread > 3 pips (OTC) ou > 1.5 pips (regular) → Cancelar
"""

# === Consulta o GPT com fallback de API ===
def consultar_gpt(prompt: str) -> Optional[str]:
    for attempt in range(len(OPENAI_KEYS)):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Você é um analista técnico especialista em Forex e opções binárias."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print("⚠️ Limite atingido para esta chave. Tentando próxima...")
            set_next_openai_key()
        except Exception as e:
            print(f"[ERRO GPT] {e}")
            set_next_openai_key()
    return None

def gerar_sinal_adaptativo(df: pd.DataFrame, indicadores: dict, estrategia: str) -> dict:
    """Gera sinal baseado na estratégia selecionada"""
    config = ESTRATEGIAS[estrategia]
    close = df['Close'].values
    
    # Lógica para cada estratégia
    if estrategia == 'CONSERVADORA':
        sinal = (
            'CALL' if (close[-1] > indicadores[f"MM{config['MM'][0]}"]) and 
                     (indicadores['RSI'] < 60) and
                     (indicadores['STOCH_K'] < 80) else
            'PUT' if (close[-1] < indicadores[f"MM{config['MM'][0]}"]) and 
                    (indicadores['RSI'] > 40) and
                    (indicadores['STOCH_K'] > 20) else
            'NEUTRO'
        )
    
    elif estrategia == 'AGGRESSIVA':
        sinal = (
            'CALL' if (indicadores['STOCH_K'] > indicadores['STOCH_D']) and 
                     (indicadores['CCI'] > 0) else
            'PUT' if (indicadores['STOCH_K'] < indicadores['STOCH_D']) and 
                    (indicadores['CCI'] < 0) else
            'NEUTRO'
        )
    
    else:  # HIBRIDA
        sinal = (
            'CALL' if (close[-1] > indicadores[f"MM{config['MM'][1]}"]) and 
                     (indicadores['ADX'] > config['ADX_THRESHOLD']) and
                     (indicadores['VOLUME'] > 1.2) else
            'PUT' if (close[-1] < indicadores[f"MM{config['MM'][1]}"]) and 
                    (indicadores['ADX'] > config['ADX_THRESHOLD']) and
                    (indicadores['VOLUME'] > 1.2) else
            'NEUTRO'
        )
    
    return {
        'sinal': sinal,
        'confianca': calcular_confianca_sinal(indicadores, estrategia),
        'timestamp': datetime.utcnow().isoformat()
        }

def calcular_confianca_sinal(indicadores: dict, estrategia: str) -> float:
    """Calcula nível de confiança do sinal (0-1)"""
    if estrategia == 'CONSERVADORA':
        return min(
            0.9,
            0.3 + (indicadores['ADX']/100 * 0.5) + 
            (1 - abs(indicadores['RSI']-50)/50 * 0.2
        )
    
    elif estrategia == 'AGGRESSIVA':
        return min(
            0.85,
            0.4 + abs(indicadores['CCI'])/100 * 0.4 +
            (indicadores['VOLUME']-1) * 0.2
        )
    
    else:  # HIBRIDA
        return min(
            0.95,
            0.5 + (indicadores['ADX']/100 * 0.3) +
            (1 - abs(indicadores['RSI']-50)/50 * 0.2 +
            (indicadores['VOLUME']-1) * 0.1
        )

def analisar_par() -> Optional[Dict[str, str]]:
    selecao = selecionar_par_tempo()
    symbol, interval = selecao["par"], selecao["timeframe"]
    window = WINDOW_SIZE_DYNAMIC.get(interval, 50)
    
    # Verificar notícias antes de continuar
    if verificar_noticias():
        print(f"⚠️ Notícias importantes - evitando operar {symbol}")
        return None

    base_symbol = symbol.replace(" OTC", "").replace(" ", "")
    pair = f"{base_symbol[:3]}/{base_symbol[3:]}"

    result = data_client.fetch_candles(pair, interval=interval, limit=window)
    if not result or "history" not in result:
        print(f"⚠️ Nenhum provedor retornou dados para {symbol}")
        return None

    df = pd.DataFrame(result["history"])
    if df.empty:
        print(f"⚠️ DataFrame vazio para {symbol}")
        return None
        
    df["Date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    }, inplace=True)
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    if len(df) < window:
        print(f"⚠️ Dados insuficientes para {symbol} (window={window}, obtidos={len(df)})")
        return None

    # Passo 1: Analisar condições do mercado
    conditions = analisar_condicoes_mercado(df)
    estrategia = selecionar_estrategia(conditions)

    # Passo 2: Calcular indicadores
    mms = calcular_medias_moveis(df)
    bb = calcular_bollinger(df)
    atr = calcular_atr(df)
    atr_pct = (atr / df['Close'].iloc[-1]) * 100
    indicadores = calcular_indicadores_avancados(df, estrategia)
    
    # Passo 3: Gerar sinal adaptativo
    sinal = gerar_sinal_adaptativo(df, indicadores, estrategia)
    
    levels = data_client.fetch_support_resistance(pair, period="1D") if hasattr(data_client, 'fetch_support_resistance') else None

    levels = sr_analyzer.analyze(pair, operation_type)

    # Verificar se temos dados de fractais avançados
    if levels and 'fractal_details' in levels:
        fractal_info = levels['fractal_details']
        
    candles_text = formatar_candles(df, interval)
    if levels:
        candles_text += f"\n\n### 🎯 NÍVEIS (24h):\nSuporte: {levels.get('support', 'N/A')}\nResistência: {levels.get('resistance', 'N/A')}"
    
    prompt = montar_prompt(candles_text, symbol, interval, atr_pct, estrategia)
    resposta = consultar_gpt(prompt)

    if not resposta or resposta.startswith("[ERRO GPT]"):
        return None

    print(f"\n📈 [{symbol} | {interval}] - {datetime.utcnow().isoformat()} UTC\n{resposta}")

    # Dados para ML
    dados_ml = {
        'MM5': mms['MM5'],
        'MM14': mms['MM14'],
        'MM21': mms['MM21'],
        'ATR': atr,
        'BB_width': bb['BB_width'],
        'RSI': indicadores['RSI'],
        'STOCH_K': indicadores['STOCH_K'],
        'STOCH_D': indicadores['STOCH_D'],
        'CCI': indicadores['CCI'],
        'ADX': indicadores['ADX'],
        'VOLUME': indicadores['VOLUME'],
        'Volume_Change': (df['Volume'].iloc[-1] / df['Volume'].mean() - 1) * 100
    }
    
    prob_ml, meta_ml = ml_model.prever(dados_ml)
    
    # Ajuste dinâmico
    if prob_ml < 0.65 and "COMPRA" in resposta:
        resposta = resposta.replace("COMPRA", "AGUARDAR")
        resposta += f"\n⚠️ ML: Confiança insuficiente ({prob_ml:.0%} | Modelo: {meta_ml['model']})"
    
    resultado = {
        "par": symbol,
        "timeframe": interval,
        "timestamp": datetime.utcnow().isoformat(),
        "volatilidade": f"{atr_pct:.2f}%",
        "estrategia": estrategia,
        "condicoes_mercado": conditions,
        "resposta": resposta,
        "sinal_ml": sinal,
        "prob_ml": f"{prob_ml:.0%}",
        "modelo_ml": meta_ml['model'],
        "confianca_ml": f"{meta_ml['confidence']:.0%}"
    }

    # Upload do arquivo ml
    from ai_gpt.save_upload_ml.ml_data_uploader import upload_gpt_ml_model
    success, _ = upload_gpt_ml_model(data=dados_ml)
    
    if not success:
        print("⚠️ Atenção: Dados ML não foram enviados ao Drive (mas análise continua)")
    
    telegram.enviar_sinal(resultado)
    return resultado
            
# === Loop principal ===
def main_loop():
    ultimo_horario_rodado = None

    while True:
        agora = datetime.utcnow().strftime("%H:%M")
        if agora in HORARIOS_UTC and agora != ultimo_horario_rodado:
            print(f"\n🕒 Executando análise às {agora} UTC...\n")
            resultado = analisar_par()

            if resultado:
                df_result = pd.DataFrame([resultado])
                os.makedirs("output", exist_ok=True)
                df_result.to_csv(
                    f"output/sinais_{datetime.utcnow().date()}.csv",
                    mode='a', index=False, header=False
                )

            ultimo_horario_rodado = agora
            print(f"\n✅ Análise finalizada para {agora} UTC.\n")

        time.sleep(10)

# Instância global do cliente de dados
data_client = FallbackDataClient()

if __name__ == "__main__":
    main_loop()
