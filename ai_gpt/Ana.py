import pandas as pd
from datetime import datetime
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
from support_resistance import AdvancedSRAnalyzer

# ✅ Configurações simplificadas
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD",
    "USDCAD", "EURJPY", "EURNZD", "AEDCNY", "AUDCAD", "AUDCHF",
    "AUDNZD", "CADJPY", "CHFJPY", "EURGBP"
]

OTC_SYMBOLS = [
    "EURUSD OTC", "GBPUSD OTC", "USDJPY OTC", "AUDUSD OTC", "EURJPY OTC",
    "NZDUSD OTC", "AUDCAD OTC", "AUDCHF OTC", "GBPJPY OTC", "CADJPY OTC"
]

TIMEFRAMES = ["1min", "5min", "15min", "30min", "1H", "4H"]

# ✅ Estratégias otimizadas para ML
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
            'RSI', 'MACD', 'Volume_Change', 'ADX', 'OBV',
            'STOCH_K', 'STOCH_D', 'CCI'
        ]
        self.target = 'Signal'
        self.model_path = 'model/ml_signal_xgboost.joblib'
        os.makedirs('model', exist_ok=True)
        
    def carregar_modelo(self):
        """Carrega modelo treinado ou cria novo"""
        try:
            self.model = joblib.load(self.model_path)
            print("✅ Modelo XGBoost carregado")
        except:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=42
            )
            print("🆕 Novo modelo XGBoost criado")

    def preparar_dados_treino(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara dados históricos para treino"""
        df = df.copy()
        
        # Indicadores técnicos
        df['MM5'] = talib.SMA(df['Close'], timeperiod=5)
        df['MM14'] = talib.SMA(df['Close'], timeperiod=14)
        df['MM21'] = talib.SMA(df['Close'], timeperiod=21)
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'] = talib.MACD(df['Close'])[0]
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_width'] = bb_upper - bb_lower
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        
        # Volume
        df['Volume_Change'] = df['Volume'].pct_change() * 100
        
        # Target (sinal para próximo candle)
        df['Signal'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return df.dropna()

    def treinar_modelo(self, dados_historicos: pd.DataFrame):
        """Treina o modelo com novos dados"""
        df_preparado = self.preparar_dados_treino(dados_historicos)
        
        if len(df_preparado) < 100:
            print("⚠️ Dados insuficientes para treino")
            return
            
        X = df_preparado[self.features]
        y = df_preparado[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        
        # Avaliação
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        joblib.dump(self.model, self.model_path)
        print(f"🔁 Modelo treinado | Treino: {train_score:.2%} | Teste: {test_score:.2%}")

    def extrair_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extrai features para previsão em tempo real"""
        if len(df) < 50:
            return {}
            
        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        volume = df['Volume'].values
        
        features = {
            'MM5': talib.SMA(close, timeperiod=5)[-1],
            'MM14': talib.SMA(close, timeperiod=14)[-1],
            'MM21': talib.SMA(close, timeperiod=21)[-1],
            'ATR': talib.ATR(high, low, close, timeperiod=14)[-1],
            'RSI': talib.RSI(close, timeperiod=14)[-1],
            'MACD': talib.MACD(close)[0][-1],
            'ADX': talib.ADX(high, low, close, timeperiod=14)[-1],
            'OBV': talib.OBV(close, volume)[-1],
            'STOCH_K': talib.STOCH(high, low, close)[0][-1],
            'STOCH_D': talib.STOCH(high, low, close)[1][-1],
            'CCI': talib.CCI(high, low, close, timeperiod=14)[-1],
        }
        
        # Bandas de Bollinger
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        features['BB_width'] = bb_upper[-1] - bb_lower[-1] if all([bb_upper[-1], bb_lower[-1]]) else 0
        
        # Volume
        features['Volume_Change'] = (volume[-1] / np.mean(volume[-20:]) - 1) * 100 if len(volume) >= 20 else 0
        
        return features

    def prever_sinal(self, df: pd.DataFrame) -> Dict:
        """Faz previsão completa baseada nos dados atuais"""
        if not self.model:
            return {'sinal': 'NEUTRO', 'confianca': 0.5, 'probabilidade': 0.5}
        
        features = self.extrair_features(df)
        if not features:
            return {'sinal': 'NEUTRO', 'confianca': 0.5, 'probabilidade': 0.5}
        
        # Converter para DataFrame
        input_data = pd.DataFrame([features])
        
        try:
            # Probabilidade de alta
            proba = self.model.predict_proba(input_data)[0][1]
            
            # Determinar sinal baseado na probabilidade
            if proba > 0.6:
                sinal = 'CALL'
                confianca = (proba - 0.5) * 2  # Normaliza para 0-1
            elif proba < 0.4:
                sinal = 'PUT'
                confianca = (0.5 - proba) * 2
            else:
                sinal = 'NEUTRO'
                confianca = 1 - abs(proba - 0.5) * 2
            
            return {
                'sinal': sinal,
                'confianca': min(confianca, 0.95),  # Limitar a 95%
                'probabilidade': proba,
                'features': features
            }
            
        except Exception as e:
            print(f"❌ Erro na previsão: {e}")
            return {'sinal': 'NEUTRO', 'confianca': 0.5, 'probabilidade': 0.5}

# ✅ Instâncias globais
ml_model = MLModel()
ml_model.carregar_modelo()
data_client = FallbackDataClient()
sr_analyzer = AdvancedSRAnalyzer(data_client)

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
    """Analisa condições do mercado para seleção de estratégia"""
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    
    # Volatilidade
    atr = talib.ATR(high, low, close, timeperiod=14)[-1]
    volatility = (atr / close[-1]) * 100
    
    # Tendência
    adx = talib.ADX(high, low, close, timeperiod=14)[-1]
    
    # Volume
    volume_ratio = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1.0
    
    hora = datetime.utcnow().hour
    
    return {
        'volatilidade': 'alta' if volatility > 0.3 else 'baixa' if volatility < 0.1 else 'moderada',
        'tendencia': 'forte' if adx > 25 else 'fraca',
        'volume': 'alto' if volume_ratio > 1.3 else 'baixo' if volume_ratio < 0.7 else 'normal',
        'hora': hora
    }

def selecionar_estrategia(conditions: dict) -> str:
    """Seleciona estratégia baseada nas condições"""
    hora = conditions['hora']
    
    if 0 <= hora < 6:  # Horário noturno
        return 'CONSERVADORA'
    
    if 8 <= hora < 16:  # Horário Londres/NY
        if conditions['volatilidade'] == 'alta' and conditions['volume'] == 'alto':
            return 'AGGRESSIVA'
        return 'HIBRIDA'
    
    return 'CONSERVADORA'

def gerar_sinal_tecnico(df: pd.DataFrame, estrategia: str) -> Dict:
    """Gera sinal técnico baseado na estratégia selecionada"""
    config = ESTRATEGIAS[estrategia]
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Calcular indicadores da estratégia
    mm_fast = talib.SMA(close, timeperiod=config['MM'][0])[-1]
    mm_slow = talib.SMA(close, timeperiod=config['MM'][1])[-1]
    rsi = talib.RSI(close, timeperiod=config['RSI'])[-1]
    stoch_k, stoch_d = talib.STOCH(high, low, close, *config['STOCH'])
    cci = talib.CCI(high, low, close, timeperiod=config['CCI'])[-1]
    adx = talib.ADX(high, low, close, timeperiod=14)[-1]
    
    # Lógica por estratégia
    if estrategia == 'CONSERVADORA':
        if close[-1] > mm_fast > mm_slow and rsi < 60 and stoch_k[-1] < 80:
            sinal = 'CALL'
        elif close[-1] < mm_fast < mm_slow and rsi > 40 and stoch_k[-1] > 20:
            sinal = 'PUT'
        else:
            sinal = 'NEUTRO'
            
    elif estrategia == 'AGGRESSIVA':
        if stoch_k[-1] > stoch_d[-1] and cci > 0:
            sinal = 'CALL'
        elif stoch_k[-1] < stoch_d[-1] and cci < 0:
            sinal = 'PUT'
        else:
            sinal = 'NEUTRO'
            
    else:  # HIBRIDA
        volume_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
        if close[-1] > mm_slow and adx > config['ADX_THRESHOLD'] and volume_ratio > 1.2:
            sinal = 'CALL'
        elif close[-1] < mm_slow and adx > config['ADX_THRESHOLD'] and volume_ratio > 1.2:
            sinal = 'PUT'
        else:
            sinal = 'NEUTRO'
    
    return {
        'sinal_tecnico': sinal,
        'estrategia': estrategia,
        'timestamp': datetime.utcnow().isoformat()
    }

def analisar_par(symbol: str, timeframe: str) -> Optional[Dict]:
    """Análise completa do par usando apenas ML e análise técnica"""
    print(f"📊 Analisando {symbol} ({timeframe})...")
    
    # Buscar dados
    result = data_client.fetch_candles(symbol, interval=timeframe, limit=100)
    if not result or "history" not in result:
        print(f"❌ Dados não encontrados para {symbol}")
        return None

    df = pd.DataFrame(result["history"])
    if df.empty or len(df) < 50:
        print(f"❌ Dados insuficientes para {symbol}")
        return None
        
    # Preparar DataFrame
    df["Date"] = pd.to_datetime(df["timestamp"], unit="s")
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    }, inplace=True)
    
    # Análise de condições
    conditions = analisar_condicoes_mercado(df)
    estrategia = selecionar_estrategia(conditions)
    
    # Sinal técnico
    sinal_tecnico = gerar_sinal_tecnico(df, estrategia)
    
    # Previsão ML
    sinal_ml = ml_model.prever_sinal(df)
    
    # Decisão final (combina técnico + ML)
    decisao_final = tomar_decisao_final(sinal_tecnico, sinal_ml)
    
    resultado = {
        "par": symbol,
        "timeframe": timeframe,
        "timestamp": datetime.utcnow().isoformat(),
        "condicoes_mercado": conditions,
        "estrategia": estrategia,
        "sinal_tecnico": sinal_tecnico['sinal_tecnico'],
        "sinal_ml": sinal_ml['sinal'],
        "confianca_ml": f"{sinal_ml['confianca']:.2%}",
        "probabilidade_ml": f"{sinal_ml['probabilidade']:.2%}",
        "decisao_final": decisao_final,
        "features_ml": sinal_ml.get('features', {})
    }
    
    print(f"✅ Análise concluída: {decisao_final} (ML: {sinal_ml['confianca']:.2%})")
    return resultado

def tomar_decisao_final(sinal_tecnico: Dict, sinal_ml: Dict) -> str:
    """Combina sinal técnico com ML para decisão final"""
    # Se ML tem alta confiança, prioriza
    if sinal_ml['confianca'] > 0.7:
        return sinal_ml['sinal']
    
    # Se técnico e ML concordam
    if sinal_tecnico['sinal_tecnico'] == sinal_ml['sinal'] and sinal_ml['confianca'] > 0.5:
        return sinal_ml['sinal']
    
    # Caso contrário, neutro ou técnico
    if sinal_tecnico['sinal_tecnico'] != 'NEUTRO' and sinal_ml['confianca'] < 0.4:
        return sinal_tecnico['sinal_tecnico']
    
    return 'NEUTRO'

def treinar_modelo_com_dados_historicos():
    """Função para treinar modelo com dados históricos"""
    print("🔄 Coletando dados para treino...")
    
    # Exemplo: coletar dados de vários pares e timeframes
    dados_treino = []
    
    for symbol in SYMBOLS[:3]:  # Limitar para teste


uu




SYMBOLS[:3]:  # Limitar para teste

        for timeframe in ['15min', '1H']:
            result = data_client.fetch_candles(symbol, interval=timeframe, limit=500)
            if result and "history" in result:

                df = pd.DataFrame(result["history"])

                if not df.empty:
                    df["symbol"] = symbol
                    df["timeframe"] = timeframe
                    dados_treino.append(df)

    if dados_treino:

        df_completo = pd.concat(dados_treino, ignore_index=True)
        ml_model.treinar_modelo(df_completo)

        print("✅ Treino concluído")

    else:

        print("❌ Dados insuficientes para treino")

def main_loop():

    """Loop principal sem GPT"""

    print("🚀 Iniciando analisador XGBoost...")    

    # Treinar modelo inicial se necessário
    if not os.path.exists(ml_model.model_path):

        treinar_modelo_com_dados_historicos()

    while True:

        try:

            # Selecionar par e timeframe aleatórios

            symbol = random.choice(SYMBOLS + OTC_SYMBOLS)

            timeframe = random.choice(TIMEFRAMES)  

            # Fazer análise
            resultado = analisar_par(symbol, timeframe)            

            if resultado:

                # Salvar resultado
                df_result = pd.DataFrame([resultado])
                os.makedirs("output", exist_ok=True)
                df_result.to_csv(
                    f"output/sinais_ml_{datetime.utcnow().date()}.csv",
                    mode='a', index=False, header=not os.path.exists(f"output/sinais_ml_{datetime.utcnow().date()}.csv")

                )
       
                print(f"💾 Resultado salvo: {resultado['decisao_final']}")
            
            # Esperar antes da próxima análise
            import time
            time.sleep(60)  # Analisar a cada 1 minuto
            
        except Exception as e:
            print(f"❌ Erro no loop principal: {e}")
            import time
            time.sleep(30)

if __name__ == "__main__":
    main_loop()
