import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from Utility.logging_config import get_trading_logger
import re
from enum import Enum
from datetime import datetime, time

logger = logging.getLogger("market_aware_processor")

try:
    from cache_manager import global_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning("Cache manager não disponível - usando fallback")

class MarketType(Enum):
    FOREX = "forex"
    CRYPTO = "crypto" 
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"

class MarketRegime(Enum):
    FOREX_DRIVEN = "forex_driven"
    COMMODITY_DRIVEN = "commodity_driven"
    RISK_OFF = "risk_off"
    RATES_DRIVEN = "rates_driven"
    HYBRID = "hybrid"

class MarketAwareDataProcessor:
    """
    Processador de dados inteligente que adapta dinamicamente o tratamento por tipo de mercado
    """
    
    # Padrões de identificação de mercados (AGORA COM CACHE)
    FOREX_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
        "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "EURNZD"
    ]
    
    CRYPTO_SYMBOLS = [
        "BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XRP", "EOS", "XTZ",
        "ATOM", "SOL", "AVAX", "MATIC", "DOT", "UNI", "LINK"
    ]
    
    # Commodities que são negociadas como Forex (pares)
    COMMODITY_FOREX_SYMBOLS = [
        "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "OILUSD"
    ]
    
    # Commodities puras (futuros, CFDs)
    PURE_COMMODITY_SYMBOLS = [
        "GOLD", "XAU", "GC", "SI", "XAG", "CL", "BRENT",
        "OIL", "WTI", "NG", "NATGAS", "COPPER", "HG", "PLATINUM", "PALLADIUM"
    ]
    
    # CACHE DE SAZONALIDADE (estático)
    SEASONALITY_MAP = {
        1: 1.05,   # Janeiro - forte
        2: 1.02,   # Fevereiro - moderado
        3: 0.98,   # Março - fraco
        4: 0.95,   # Abril - fraco
        5: 0.96,   # Maio - neutro
        6: 0.98,   # Junho - neutro
        7: 1.00,   # Julho - neutro
        8: 1.03,   # Agosto - fortalecimento
        9: 1.08,   # Setembro - forte
        10: 1.10,  # Outubro - muito forte
        11: 1.12,  # Novembro - pico
        12: 1.08   # Dezembro - forte
    }
    
    # CACHE DE PARÂMETROS POR TIPO DE OPERAÇÃO
    EMA_SPANS_CACHE = {
        'forex': {
            'scalping': 12, 'daytrade': 21, 'swing': 50, 'position': 100
        },
        'crypto': {
            'scalping': 9, 'daytrade': 14, 'swing': 25, 'position': 50
        },
        'commodity': {
            'precious_metal': {
                'scalping': (9, 21), 'daytrade': (14, 50), 'swing': (21, 100), 'position': (50, 200)
            },
            'energy': {
                'scalping': (7, 14), 'daytrade': (12, 26), 'swing': (20, 50), 'position': (40, 100)
            },
            'other': {
                'scalping': (10, 20), 'daytrade': (15, 30), 'swing': (25, 60), 'position': (50, 150)
            }
        }
    }

    @staticmethod
    def detect_market_type_cached(symbol: str) -> MarketType:
        """Versão cacheada da detecção de tipo de mercado"""
        return MarketAwareDataProcessor._detect_market_type_original(symbol)

    @staticmethod
    def _detect_market_type_original(symbol: str) -> MarketType:
        """Implementação original (privada)"""
        symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        
        # 1. PRIMEIRO: Commodities que são negociadas como pares Forex
        if any(commodity_forex in symbol_clean for commodity_forex in MarketAwareDataProcessor.COMMODITY_FOREX_SYMBOLS):
            return MarketType.FOREX
            
        # 2. Forex tradicional
        if any(forex_pair in symbol_clean for forex_pair in MarketAwareDataProcessor.FOREX_PAIRS):
            return MarketType.FOREX
            
        # 3. Padrões Forex genéricos
        if re.match(r'^[A-Z]{6}$', symbol_clean):
            return MarketType.FOREX
        elif re.match(r'^[A-Z]{3,}USD$|^USD[A-Z]{3}$', symbol_clean):
            return MarketType.FOREX
            
        # 4. Commodities puras
        if any(commodity in symbol_clean for commodity in MarketAwareDataProcessor.PURE_COMMODITY_SYMBOLS):
            return MarketType.COMMODITY
            
        # 5. Padrões específicos para commodities
        if re.match(r'^XAU|^XAG|^GC|^SI|^CL|^NG', symbol_clean):
            return MarketType.COMMODITY
        elif re.match(r'^GOLD|^SILVER|^OIL|^BRENT|^WTI|^COPPER', symbol_clean):
            return MarketType.COMMODITY
            
        # 6. Crypto
        if any(crypto in symbol_clean for crypto in MarketAwareDataProcessor.CRYPTO_SYMBOLS):
            return MarketType.CRYPTO
        elif re.match(r'^[A-Z]{3,}USDT$|^USDT[A-Z]{3}$', symbol_clean):
            return MarketType.CRYPTO
            
        # Fallback
        if len(symbol_clean) <= 6 and symbol_clean.isalpha():
            return MarketType.FOREX
        else:
            return MarketType.CRYPTO

    @staticmethod
    def detect_market_type(symbol: str) -> MarketType:
        """Interface pública com cache"""
        if CACHE_AVAILABLE:
            # Usar cache manager se disponível
            return global_cache.get_for_or_refresh(
                symbol, "market_type", {},
                refresh_fn=lambda: MarketAwareDataProcessor._detect_market_type_original(symbol),
                expiry=3600,  # 1 hora de cache
                cache_namespace="market_analysis"
            )
        else:
            # Fallback para lru_cache
            return MarketAwareDataProcessor.detect_market_type_cached(symbol)

    @staticmethod
    def _get_commodity_params_cached(symbol: str) -> Dict:
        """Cache de parâmetros de commodity"""
        symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        
        if 'XAU' in symbol_clean or 'GOLD' in symbol_clean:
            return {'type': 'precious_metal', 'multiplier': 100, 'volatility_factor': 1.2}
        elif 'XAG' in symbol_clean or 'SILVER' in symbol_clean:
            return {'type': 'precious_metal', 'multiplier': 5000, 'volatility_factor': 1.5}
        elif 'OIL' in symbol_clean:
            return {'type': 'energy', 'multiplier': 1000, 'volatility_factor': 1.8}
        else:
            return {'type': 'other', 'multiplier': 1000, 'volatility_factor': 1.3}

    @staticmethod
    def detect_market_regime(df: pd.DataFrame, symbol: str, external_data: Dict = None) -> MarketRegime:
        """
        Detecta dinamicamente o regime de mercado para XAUUSD e similares
        Baseado no comportamento real dos dados, não apenas classificação estática
        """
        symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        
        if symbol_clean not in MarketAwareDataProcessor.COMMODITY_FOREX_SYMBOLS:
            return MarketRegime.HYBRID

        try:
            # Pré-calcular valores usados múltiplas vezes
            close_pct_change = df['Close'].pct_change()
            gold_volatility = close_pct_change.std()
            
            # ANÁLISE DE SAZONALIDADE INTRADIÁRIA
            current_hour = datetime.now().hour
            is_asia_session = 22 <= current_hour or current_hour < 6
            is_london_session = 6 <= current_hour < 14
            is_ny_session = 14 <= current_hour < 22
            
            # Evitar múltiplas operações no mesmo DataFrame
            price_changes = close_pct_change.dropna()
            large_moves = len(price_changes[abs(price_changes) > 0.005])
            
            # Calcular MAs apenas se necessário
            if len(df) >= 20:
                short_ma = df['Close'].rolling(5, min_periods=1).mean()
                long_ma = df['Close'].rolling(20, min_periods=1).mean()
                trend_strength = abs(short_ma.iloc[-1] - long_ma.iloc[-1]) / df['Close'].iloc[-1]
            else:
                trend_strength = 0
            
            regime_scores = {
                MarketRegime.FOREX_DRIVEN: 0,
                MarketRegime.COMMODITY_DRIVEN: 0,
                MarketRegime.RISK_OFF: 0,
                MarketRegime.RATES_DRIVEN: 0
            }
            
            # Fatores que indicam comportamento Forex
            if is_london_session or is_ny_session:
                regime_scores[MarketRegime.FOREX_DRIVEN] += 2
            if gold_volatility < 0.008:  # Baixa volatilidade
                regime_scores[MarketRegime.FOREX_DRIVEN] += 1
            if trend_strength < 0.02:  # Tendência fraca
                regime_scores[MarketRegime.FOREX_DRIVEN] += 1
                
            # Fatores que indicam comportamento Commodity
            if is_asia_session:
                regime_scores[MarketRegime.COMMODITY_DRIVEN] += 2
            if gold_volatility > 0.012:  # Alta volatilidade
                regime_scores[MarketRegime.COMMODITY_DRIVEN] += 2
            if large_moves > len(price_changes) * 0.3:  # Muitos movimentos grandes
                regime_scores[MarketRegime.COMMODITY_DRIVEN] += 1
                
            # Fatores de risco
            if gold_volatility > 0.015 and large_moves > len(price_changes) * 0.4:
                regime_scores[MarketRegime.RISK_OFF] += 3
                
            # Dados externos (se disponíveis)
            if external_data:
                if external_data.get('vix', 0) > 25:
                    regime_scores[MarketRegime.RISK_OFF] += 2
                if external_data.get('usd_strength', 0) > 0.5:
                    regime_scores[MarketRegime.FOREX_DRIVEN] += 1
            
            # Determina o regime vencedor
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            # Só retorna regime específico se tiver score significativo
            if best_regime[1] >= 3:
                logger.info(f"Regime detectado para {symbol}: {best_regime[0].value} (score: {best_regime[1]})")
                return best_regime[0]
            else:
                logger.info(f"Regime híbrido para {symbol} (scores: {regime_scores})")
                return MarketRegime.HYBRID
                
        except Exception as e:
            logger.warning(f"Erro na detecção de regime para {symbol}: {e}")
            return MarketRegime.HYBRID

    @staticmethod
    def process_market_aware_data(data: Dict, 
                                 symbol: str,
                                 operation_type: str = "daytrade",
                                 external_data: Dict = None) -> pd.DataFrame:
        """
        Processamento inteligente e dinâmico que se adapta ao regime de mercado
        """
        # USAR CACHE PARA DETECÇÃO DE TIPO DE MERCADO
        market_type = MarketAwareDataProcessor.detect_market_type(symbol)
        
        # Processamento base comum
        df = MarketAwareDataProcessor._process_base_ohlcv(data, symbol)
        if df.empty:
            return df
            
        # DETECÇÃO DINÂMICA DE REGIME
        symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        
        if symbol_clean in MarketAwareDataProcessor.COMMODITY_FOREX_SYMBOLS:
            market_regime = MarketAwareDataProcessor.detect_market_regime(df, symbol, external_data)
            logger.info(f"Processando {symbol} - Mercado: {market_type.value}, Regime: {market_regime.value}")
            
            return MarketAwareDataProcessor._process_dynamic_commodity_forex(
                df, symbol, operation_type, market_regime
            )
            
        # Processamento normal para outros mercados
        elif market_type == MarketType.FOREX:
            logger.info(f"Processando {symbol} como Forex tradicional")
            return MarketAwareDataProcessor._enhance_forex_data(df, symbol, operation_type)
            
        elif market_type == MarketType.CRYPTO:
            logger.info(f"Processando {symbol} como Crypto")
            return MarketAwareDataProcessor._enhance_crypto_data(df, symbol, operation_type)
            
        elif market_type == MarketType.COMMODITY:
            logger.info(f"Processando {symbol} como Commodity pura")
            return MarketAwareDataProcessor._enhance_commodity_data(df, symbol, operation_type)
            
        else:
            logger.info(f"Processando {symbol} como mercado genérico")
            return MarketAwareDataProcessor._enhance_generic_data(df, symbol, operation_type)

    @staticmethod
    def _process_dynamic_commodity_forex(df: pd.DataFrame, symbol: str, 
                                       operation_type: str, regime: MarketRegime) -> pd.DataFrame:
        """
        Processamento dinâmico para commodity-forex baseado no regime detectado
        """
        try:
            # APLICA PROCESSAMENTO BASE COMUM
            df = MarketAwareDataProcessor._enhance_forex_data(df, symbol, operation_type)
            
            # APLICA MELHORIAS ESPECÍFICAS POR REGIME
            if regime == MarketRegime.FOREX_DRIVEN:
                df = MarketAwareDataProcessor._apply_forex_driven_enhancements(df, symbol, operation_type)
            elif regime == MarketRegime.COMMODITY_DRIVEN:
                df = MarketAwareDataProcessor._apply_commodity_driven_enhancements(df, symbol, operation_type)
            elif regime == MarketRegime.RISK_OFF:
                df = MarketAwareDataProcessor._apply_risk_off_enhancements(df, symbol, operation_type)
            elif regime == MarketRegime.RATES_DRIVEN:
                df = MarketAwareDataProcessor._apply_rates_driven_enhancements(df, symbol, operation_type)
            else:  # HYBRID
                df = MarketAwareDataProcessor._apply_hybrid_enhancements(df, symbol, operation_type)
            
            # MARCA O REGIME DETECTADO
            df['MarketRegime'] = regime.value
            df['RegimeConfidence'] = MarketAwareDataProcessor._calculate_regime_confidence(df, regime)
            
            logger.info(f"Processamento dinâmico completo para {symbol} no regime {regime.value}")
            return df
            
        except Exception as e:
            logger.error(f"Erro no processamento dinâmico para {symbol}: {e}")
            return df

    @staticmethod
    def _apply_forex_driven_enhancements(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Melhorias quando XAUUSD se comporta mais como Forex"""
        logger.debug(f"Aplicando enhancements Forex-driven para {symbol}")
        
        try:
            # Pré-calcular valores comuns
            effective_volume = df['EffectiveVolume']
            close_prices = df['Close']
            sessions = df['Session']
            
            # ÊNFASE EM FATORES FOREX
            df['ForexBias'] = 1.0
            
            session_means = effective_volume.groupby(sessions).mean()
            df['SessionStrength'] = effective_volume / sessions.map(session_means)

            # Análise de correlação com USD (mais importante neste regime)
            df['USDCorrelationStrength'] = df['Close'].rolling(10).corr(
                df['EffectiveVolume']  # Proxy para fluxo USD
            ).abs()
            
            session_map = {'europe': 1.0, 'us': 1.0, 'asia': 0.3, 'other': 0.3}
            df['OptimalForexTiming'] = sessions.map(session_map).fillna(0.3)
            
            # Indicadores técnicos otimizados para Forex
            df['ForexMomentum'] = close_prices.pct_change(periods=3)
            
            # Calcular session momentum de forma vetorizada
            session_changes = close_prices.groupby(sessions).pct_change(periods=2)
            df['SessionMomentum'] = session_changes
            
            logger.debug("Enhancements Forex-driven aplicados")
            return df
            
        except Exception as e:
            logger.error(f"Erro nos enhancements Forex-driven: {e}")
            return df

    @staticmethod
    def _apply_commodity_driven_enhancements(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Melhorias quando XAUUSD se comporta mais como Commodity"""
        logger.debug(f"Aplicando enhancements Commodity-driven para {symbol}")
        
        try:
            # Usar cache de sazonalidade
            df['Month'] = df.index.month
            df['CommoditySeasonality'] = df['Month'].map(
                MarketAwareDataProcessor.SEASONALITY_MAP
            ).fillna(1.0)
            
            # Pré-calcular valores
            close_pct_change = df['Close'].pct_change()
            effective_volume = df['EffectiveVolume']
            close_prices = df['Close']
            
            # ÊNFASE EM FATORES COMMODITY
            df['CommodityBias'] = 1.0
            
            # Volatilidade adaptativa para commodities
            df['CommodityVolatility'] = close_pct_change.rolling(10, min_periods=1).std() * np.sqrt(252)
            
            # Evitar múltiplas operações de shift
            price_diff = close_prices - close_prices.shift(1)
            df['PriceVolumeTrend'] = effective_volume * price_diff
            df['AccumulationDistribution'] = df['PriceVolumeTrend'].cumsum()
            
            # Fatores macroeconômicos proxy
            df['InflationHedgeScore'] = close_pct_change.rolling(5, min_periods=1).mean() * 100
            
            logger.debug("Enhancements Commodity-driven aplicados")
            return df
            
        except Exception as e:
            logger.error(f"Erro nos enhancements Commodity-driven: {e}")
            return df

    @staticmethod
    def _apply_risk_off_enhancements(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Melhorias para regime de aversão ao risco"""
        logger.debug(f"Aplicando enhancements Risk-Off para {symbol}")
        
        try:
            # Pré-calcular valores
            close_prices = df['Close']
            close_pct_change = close_prices.pct_change()
            
            # CARACTERÍSTICAS DE SAFE HAVEN
            df['SafeHavenBias'] = 1.0
            
            # Análise de fluxo de segurança
            df['RiskOffSignal'] = (close_prices > close_prices.shift(5)).astype(int)
            
            # Volatilidade em tempos de crise
            df['CrisisVolatility'] = close_pct_change.rolling(5, min_periods=1).std() * 100
            
            # Calcular rolling statistics uma vez
            rolling_20_mean = close_prices.rolling(20, min_periods=1).mean()
            rolling_20_std = close_prices.rolling(20, min_periods=1).std()
            
            df['ReversionPotential'] = (close_prices - rolling_20_mean) / rolling_20_std.replace(0, 1e-9)
            
            # Timing para eventos de risco
            crisis_vol_quantile = df['CrisisVolatility'].quantile(0.7)
            df['RiskEventTiming'] = np.where(df['CrisisVolatility'] > crisis_vol_quantile, 1.5, 1.0)
            
            logger.debug("Enhancements Risk-Off aplicados")
            return df
            
        except Exception as e:
            logger.error(f"Erro nos enhancements Risk-Off: {e}")
            return df

    @staticmethod
    def _apply_rates_driven_enhancements(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Melhorias para regime direcionado por taxas de juros"""
        logger.debug(f"Aplicando enhancements Rates-Driven para {symbol}")
        
        try:
            # Pré-calcular valores
            close_pct_change = df['Close'].pct_change()
            effective_volume = df['EffectiveVolume']
            
            # FOCUS EM TAXAS DE JUROS
            df['RatesBias'] = 1.0
            
            # Proxy para expectativas de taxas
            df['RateSensitivity'] = close_pct_change.rolling(3, min_periods=1).mean() * 100
            
            # Análise de carry trade reverso
            df['NegativeCarry'] = np.where(close_pct_change < 0, 1.2, 1.0)
            
            # Sensibilidade a notícias macro
            vol_20_mean = effective_volume.rolling(20, min_periods=1).mean()
            df['MacroEventSensitivity'] = effective_volume / vol_20_mean.replace(0, 1e-9)
            
            logger.debug("Enhancements Rates-Driven aplicados")
            return df
            
        except Exception as e:
            logger.error(f"Erro nos enhancements Rates-Driven: {e}")
            return df

    @staticmethod
    def _apply_hybrid_enhancements(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Abordagem balanceada quando não há regime claro"""
        logger.debug(f"Aplicando enhancements Híbridos para {symbol}")
        
        try:
            # Aplicar enhancements de forma seletiva
            # Apenas colunas necessárias para hybrid score
            df = MarketAwareDataProcessor._apply_forex_driven_enhancements(df, symbol, operation_type)
            df = MarketAwareDataProcessor._apply_commodity_driven_enhancements(df, symbol, operation_type)
            
            # MÉTRICAS HÍBRIDAS com fallback seguro
            session_strength = df.get('SessionStrength', pd.Series(1, index=df.index))
            commodity_seasonality = df.get('CommoditySeasonality', pd.Series(1, index=df.index))
            risk_timing = df.get('RiskEventTiming', pd.Series(1, index=df.index))
            
            df['HybridScore'] = session_strength * commodity_seasonality * risk_timing
            
            # INDICADORES ADAPTATIVOS
            adaptive_span = MarketAwareDataProcessor._get_adaptive_span(df, operation_type)
            df['AdaptiveMA'] = df['Close'].ewm(span=adaptive_span, adjust=False).mean()
            
            logger.debug("Enhancements Híbridos aplicados")
            return df
            
        except Exception as e:
            logger.error(f"Erro nos enhancements Híbridos: {e}")
            return df

    @staticmethod
    def _get_detailed_seasonality(month_series: pd.Series) -> float:
        """Sazonalidade detalhada para metais preciosos"""
        return month_series.map(MarketAwareDataProcessor.SEASONALITY_MAP).fillna(1.0)

    @staticmethod
    def _get_adaptive_span(df: pd.DataFrame, operation_type: str) -> int:
        """Span adaptativo baseado na volatilidade"""
        volatility = df['Close'].pct_change().std()
        
        # Usar cache de spans
        spans = {
            'scalping': {0.005: 6, 0.01: 8, 'default': 12},
            'daytrade': {0.005: 12, 0.01: 14, 'default': 20},
            'swing': {0.005: 25, 0.01: 30, 'default': 40}
        }
        
        op_spans = spans.get(operation_type, spans['daytrade'])
        if volatility < 0.005:
            return op_spans[0.005]
        elif volatility < 0.01:
            return op_spans[0.01]
        else:
            return op_spans['default']

    @staticmethod
    def _calculate_regime_confidence(df: pd.DataFrame, regime: MarketRegime) -> float:
        """Calcula a confiança no regime detectado"""
        try:
            confidence_indicators = []
            
            if regime == MarketRegime.FOREX_DRIVEN and 'SessionStrength' in df.columns:
                session_strength = df['SessionStrength'].iloc[-10:].mean()
                confidence_indicators.append(min(session_strength, 2.0))
                    
            elif regime == MarketRegime.COMMODITY_DRIVEN and 'CommodityVolatility' in df.columns:
                vol_strength = df['CommodityVolatility'].iloc[-1] / 0.01
                confidence_indicators.append(min(vol_strength, 2.0))
            
            return float(np.mean(confidence_indicators)) if confidence_indicators else 0.5
            
        except:
            return 0.5

    @staticmethod
    def _process_base_ohlcv(data: Dict, symbol: str) -> pd.DataFrame:
        """Processamento base comum para todos os mercados - OTIMIZADO"""
        if not data or "history" not in data or not data["history"]:
            logger.warning(f"Dados vazios para {symbol}")
            return pd.DataFrame()

        try:
            # Converter para DataFrame de forma
            df = pd.DataFrame(data["history"]).copy()
            
            rename_map = {
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
                "tickVolume": "TickVolume", "real_volume": "RealVolume",
                "time": "timestamp", "datetime": "timestamp"
            }
            
            # Renomear apenas colunas existentes
            existing_columns = set(df.columns)
            rename_dict = {k: v for k, v in rename_map.items() if k in existing_columns}
            df.rename(columns=rename_dict, inplace=True)
            
            # Converter para numérico em lote
            numeric_cols = ['Open', 'High', 'Low', 'Close']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                else:
                    logger.error(f"Coluna {col} não encontrada para {symbol}")
                    return pd.DataFrame()
            
            # Processamento de volume
            volume_cols = ['Volume', 'TickVolume', 'RealVolume']
            available_volume_cols = [col for col in volume_cols if col in df.columns]
            if available_volume_cols:
                df['Volume'] = pd.to_numeric(df[available_volume_cols[0]], errors="coerce")
            else:
                df['Volume'] = np.nan
                
            # Processamento de timestamp
            if "timestamp" in df.columns:
                try:
                    # Tentar converter como timestamp UNIX primeiro
                    df['Date'] = pd.to_datetime(df['timestamp'], unit='s', errors="coerce")
                    # Se falhar, tentar formato string
                    if df['Date'].isna().all():
                        df['Date'] = pd.to_datetime(df['timestamp'], errors="coerce")
                except:
                    df['Date'] = pd.to_datetime(df['timestamp'], errors="coerce")
            else:
                # Fallback
                df['Date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="min")
            
            # Set index e sorting
            df = df.set_index('Date').sort_index()
            
            return df.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
        except Exception as e:
            logger.error(f"Erro no processamento base para {symbol}: {e}")
            return pd.DataFrame()

    staticmethod
    def _enhance_commodity_forex_hybrid(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """
        Enhancements ESPECIAIS para pares Forex de commodities (XAUUSD, XAGUSD, etc.)
        Combina características de Forex e Commodity
        """
        logger.debug(f"Aplicando enhancements híbridos Forex-Commodity para {symbol}")
        
        try:
            symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
            
            # Identifica o tipo de commodity
            if 'XAU' in symbol_clean or 'GOLD' in symbol_clean:
                commodity_type = 'gold'
                commodity_params = {
                    'type': 'precious_metal', 
                    'multiplier': 100, 
                    'volatility_factor': 1.2,
                    'session_sensitivity': 'medium'
                }
            elif 'XAG' in symbol_clean or 'SILVER' in symbol_clean:
                commodity_type = 'silver'
                commodity_params = {
                    'type': 'precious_metal', 
                    'multiplier': 5000, 
                    'volatility_factor': 1.5,
                    'session_sensitivity': 'high'
                }
            elif 'OIL' in symbol_clean:
                commodity_type = 'oil'
                commodity_params = {
                    'type': 'energy', 
                    'multiplier': 1000, 
                    'volatility_factor': 1.8,
                    'session_sensitivity': 'high'
                }
            else:
                commodity_type = 'other'
                commodity_params = {
                    'type': 'other', 
                    'multiplier': 1000, 
                    'volatility_factor': 1.3,
                    'session_sensitivity': 'medium'
                }
            
            # ADICIONA CARACTERÍSTICAS ESPECÍFICAS DE COMMODITY AO DATAFRAME FOREX
            
            # 1. Metadata do commodity
            df['CommodityType'] = commodity_params['type']
            df['CommoditySymbol'] = commodity_type
            
            # 2. Volatilidade ajustada para commodity
            if 'Volatility' in df.columns:
                df['CommodityVolatility'] = df['Volatility'] * commodity_params['volatility_factor']
            else:
                # Calcula volatilidade se não existir
                df['DailyRangePct'] = (df['High'] - df['Low']) / df['Close'] * 100
                df['CommodityVolatility'] = df['DailyRangePct'].rolling(14).std() * commodity_params['volatility_factor']
            
            # 3. Sazonalidade para metais preciosos
            df['Month'] = df.index.month
            if commodity_params['type'] == 'precious_metal':
                # Metais preciosos: forte sazonalidade no final do ano
                df['MetalSeasonality'] = np.where(df['Month'].isin([8, 9, 10, 11, 12]), 1.1, 
                                                 np.where(df['Month'].isin([1, 2]), 0.95, 1.0))
            elif commodity_params['type'] == 'energy':
                # Energia: sazonalidade por estação
                df['EnergySeasonality'] = np.where(df['Month'].isin([12, 1, 2]), 1.15,  # Inverno
                                                  np.where(df['Month'].isin([6, 7, 8]), 1.1,   # Verão
                                                          1.0))
            else:
                df['CommoditySeasonality'] = 1.0
            
            # 4. Ajuste de volume para commodity (se necessário)
            if 'EffectiveVolume' in df.columns:
                df['CommodityVolume'] = df['EffectiveVolume'] * commodity_params['multiplier']
            else:
                df['CommodityVolume'] = df['Volume'] * commodity_params['multiplier']
            
            # 5. Correlação com USD (inversa para commodities)
            df['USDCorrelation'] = -0.7 if commodity_params['type'] == 'precious_metal' else -0.5
            
            # 6. Horários de negociação otimizados para commodities
            df['Hour'] = df.index.hour
            if commodity_params['session_sensitivity'] == 'high':
                # Commodities sensíveis a sessões específicas
                df['OptimalTradingHour'] = df['Hour'].apply(
                    lambda x: 1 if x in [14, 15, 16, 20, 21, 22] else 0.5  # London + NY overlap
                )
            else:
                df['OptimalTradingHour'] = 1.0
            
            # 7. Indicadores técnicos específicos para commodities
            df['CommodityMA'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['CommodityStrength'] = df['Close'] / df['CommodityMA']
            
            # 8. Momentum específico para commodities
            df['CommodityMomentum'] = df['Close'].pct_change(periods=5)
            
            logger.debug(f"Hybrid enhancements aplicados para {symbol} ({commodity_type})")
            return df
            
        except Exception as e:
            logger.error(f"Erro nos enhancements híbridos para {symbol}: {e}")
            return df
  
    @staticmethod
    def _enhance_forex_data(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Processamento Forex base"""
        logger.debug(f"Aplicando enhancements Forex base para {symbol}")
        
        try:
            # Processamento de volume mais eficiente
            if 'tickVolume' in df.columns:
                df['EffectiveVolume'] = pd.to_numeric(df['tickVolume'], errors="coerce")
                df['VolumeType'] = 'tick_volume'
            else:
                df['EffectiveVolume'] = df['Volume'].fillna(1000) * 1000
                df['VolumeType'] = 'estimated_tick_volume'

            # Pré-calcular valores comuns
            effective_volume = df['EffectiveVolume']
            close_prices = df['Close']
            high_prices = df['High']
            low_prices = df['Low']

            df['VolumeRatio'] = effective_volume / effective_volume.rolling(20, min_periods=1).mean()
            df['Range'] = high_prices - low_prices
            df['ActivityScore'] = (df['Range'] / close_prices) * df['VolumeRatio'] * 10000
            
            # Session mapping mais eficiente
            df['Hour'] = df.index.hour
            df['Session'] = df['Hour'].apply(MarketAwareDataProcessor._get_forex_session)
            
            # Substituir apply() por operação vetorizada
            session_volumes = effective_volume.groupby(df['Session']).mean()
            df['SessionVolumeRatio'] = effective_volume / df['Session'].map(session_volumes).fillna(1)

            df['Typical'] = (high_prices + low_prices + close_prices) / 3
            
            # Usar cache para EMA spans
            ema_span = MarketAwareDataProcessor.EMA_SPANS_CACHE['forex'].get(
                operation_type, 21
            )
            df['MA'] = close_prices.ewm(span=ema_span, adjust=False).mean()
            
            df['SpreadProxy'] = (high_prices - low_prices) / close_prices * 10000

            return df

        except Exception as e:
            logger.error(f"Erro nos enhancements Forex base para {symbol}: {e}")
            return df

    @staticmethod
    def _enhance_crypto_data(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Processamento Crypto"""
        logger.debug(f"Aplicando enhancements específicos para Crypto: {symbol}")
        
        try:
            # Usar volume real disponível
            df['EffectiveVolume'] = df['Volume']
            df['VolumeType'] = 'real_volume'
            
            effective_volume = df['EffectiveVolume']
            close_prices = df['Close']
            
            df['VolumeProfile'] = effective_volume / effective_volume.rolling(50, min_periods=1).sum()
            df['NotionalVolume'] = close_prices * effective_volume
            
            # Calcular volume cluster de forma mais eficiente
            vol_20_mean = effective_volume.rolling(20, min_periods=1).mean()
            df['VolumeCluster'] = (effective_volume > vol_20_mean * 1.5).astype(int)
            
            df['LogReturns'] = np.log(close_prices / close_prices.shift(1))
            df['Volatility'] = df['LogReturns'].rolling(20, min_periods=1).std() * np.sqrt(365)

            df['Typical'] = (df['High'] + df['Low'] + close_prices) / 3
            
            # OTIMIZAÇÃO: Usar cache para EMA spans
            ema_span = MarketAwareDataProcessor.EMA_SPANS_CACHE['crypto'].get(
                operation_type, 14
            )
            df['MA'] = close_prices.ewm(span=ema_span, adjust=False).mean()

            logger.debug(f"Crypto enhancements aplicados: {len(df.columns)} features")
            return df

        except Exception as e:
            logger.error(f"Erro nos enhancements Crypto para {symbol}: {e}")
            return df

    @staticmethod
    def _enhance_commodity_data(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Processamento Commodity"""
        logger.debug(f"Aplicando enhancements específicos para Commodity pura: {symbol}")
        
        try:
            symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
            
            # Determinar tipo de commodity
            commodity_type = 'other'
            volume_multiplier = 100
            
            if any(metal in symbol_clean for metal in ['GOLD', 'XAU', 'GC']):
                commodity_type = 'precious_metal'
                volume_multiplier = 100
            elif any(metal in symbol_clean for metal in ['SILVER', 'XAG', 'SI']):
                commodity_type = 'precious_metal'  
                volume_multiplier = 5000
            elif any(oil in symbol_clean for oil in ['OIL', 'CL', 'WTI', 'BRENT']):
                commodity_type = 'energy'
                volume_multiplier = 1000
            
            df['CommodityType'] = commodity_type
            
            # Tratamento de volume mais eficiente
            if df['Volume'].isna().all():
                df['EffectiveVolume'] = (df['High'] - df['Low']) * volume_multiplier
                df['VolumeType'] = 'estimated_commodity_volume'
            else:
                df['EffectiveVolume'] = df['Volume'] * volume_multiplier
                df['VolumeType'] = 'contract_volume'
            
            # Pré-calcular valores
            high_prices = df['High']
            low_prices = df['Low']
            close_prices = df['Close']
            effective_volume = df['EffectiveVolume']
            
            df['DailyRange'] = high_prices - low_prices
            df['RangePct'] = (df['DailyRange'] / close_prices) * 100
            
            vol_window = 20 if commodity_type == 'precious_metal' else 14
            df['Volatility'] = df['RangePct'].rolling(vol_window, min_periods=1).std()
            
            # Seasonality com cache
            df['Month'] = df.index.month
            if commodity_type == 'precious_metal':
                df['SeasonalFactor'] = np.where(df['Month'].isin([9, 10, 11]), 1.1, 1.0)
            elif commodity_type == 'energy':
                df['SeasonalFactor'] = np.where(df['Month'].isin([12, 1, 2, 6, 7, 8]), 1.15, 1.0)
            else:
                df['SeasonalFactor'] = 1.0
            
            # Indicadores específicos
            df['Typical'] = (high_prices + low_prices + close_prices) / 3
            
            # Usar cache para EMA spans
            ema_spans = MarketAwareDataProcessor.EMA_SPANS_CACHE['commodity'][commodity_type].get(
                operation_type, (14, 50)
            )
            df['MA_Fast'] = close_prices.ewm(span=ema_spans[0], adjust=False).mean()
            df['MA_Slow'] = close_prices.ewm(span=ema_spans[1], adjust=False).mean()
            
            df['Momentum'] = close_prices / close_prices.shift(5) - 1
            
            # Calcular liquidity score
            vol_20_mean = effective_volume.rolling(20, min_periods=1).mean()
            price_20_mean = close_prices.rolling(20, min_periods=1).mean()
            df['LiquidityScore'] = (effective_volume * close_prices) / (vol_20_mean * price_20_mean).replace(0, 1e-9)
            
            price_diff = close_prices - close_prices.shift(1)
            df['PriceVolumeTrend'] = effective_volume * price_diff
            df['PVT'] = df['PriceVolumeTrend'].cumsum()
            
            logger.debug(f"Commodity pura enhancements aplicados: {len(df.columns)} features")
            return df

        except Exception as e:
            logger.error(f"Erro nos enhancements Commodity pura para {symbol}: {e}")
            return df

    @staticmethod
    def _enhance_generic_data(df: pd.DataFrame, symbol: str, operation_type: str) -> pd.DataFrame:
        """Processamento genérico - OTIMIZADO"""
        logger.debug(f"Aplicando enhancements genéricos para {symbol}")
        
        try:
            # Processamento mínimo
            df['EffectiveVolume'] = df['Volume'].fillna(1000)
            df['VolumeType'] = 'generic_volume'
            df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['MA'] = df['Close'].ewm(span=14, adjust=False).mean()
            
            return df
        except Exception as e:
            logger.error(f"Erro nos enhancements genéricos para {symbol}: {e}")
            return df
            
    @staticmethod
    def _get_forex_session(hour: int) -> str:
        if 22 <= hour or hour < 6:
            return "asia"
        elif 6 <= hour < 14:
            return "europe"
        elif 14 <= hour < 22:
            return "us"
        else:
            return "other"
            
    @staticmethod
    def get_market_specific_advice(symbol: str, df: pd.DataFrame = None) -> Dict:
        """Conselhos específicos considerando o regime dinâmico"""
        symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        
        if symbol_clean in MarketAwareDataProcessor.COMMODITY_FOREX_SYMBOLS and df is not None:
            # Detecção de regime em tempo real para conselhos
            regime = MarketAwareDataProcessor.detect_market_regime(df, symbol)
            
            regime_advice = {
                MarketRegime.FOREX_DRIVEN: {
                    'focus': 'Análise técnica Forex + Sessões',
                    'key_metrics': ['SessionVolumeRatio', 'SpreadProxy', 'ForexMomentum'],
                    'time_sensitivity': 'ALTA',
                    'warning': 'Comportando-se como par de moeda'
                },
                MarketRegime.COMMODITY_DRIVEN: {
                    'focus': 'Fundamentos commodity + Sazonalidade',
                    'key_metrics': ['CommodityVolatility', 'Seasonality', 'AccumulationDistribution'],
                    'time_sensitivity': 'MÉDIA',
                    'warning': 'Comportando-se como commodity'
                },
                MarketRegime.RISK_OFF: {
                    'focus': 'Safe-haven + Análise de risco',
                    'key_metrics': ['CrisisVolatility', 'ReversionPotential', 'RiskEventTiming'],
                    'time_sensitivity': 'MUITO ALTA',
                    'warning': 'Regime de aversão ao risco ativo'
                },
                MarketRegime.RATES_DRIVEN: {
                    'focus': 'Taxas de juros + Eventos macro',
                    'key_metrics': ['RateSensitivity', 'MacroEventSensitivity'],
                    'time_sensitivity': 'ALTA',
                    'warning': 'Sensível a notícias de juros'
                },
                MarketRegime.HYBRID: {
                    'focus': 'Abordagem balanceada',
                    'key_metrics': ['HybridScore', 'AdaptiveMA', 'SessionStrength'],
                    'time_sensitivity': 'VARIÁVEL',
                    'warning': 'Múltiplos fatores influenciando'
                }
            }
            
            advice = regime_advice.get(regime, regime_advice[MarketRegime.HYBRID])
            advice['detected_regime'] = regime.value
            return advice
        
        # Conselhos padrão para outros símbolos
        return MarketAwareDataProcessor._get_static_advice(symbol)

    @staticmethod
    def _get_static_advice(symbol: str) -> Dict:
        """Fallback para conselhos estáticos"""
        return {
            'focus': 'Análise técnica geral',
            'key_metrics': ['MA', 'Volume', 'Volatility'],
            'time_sensitivity': 'MÉDIA',
            'warning': 'Mercado genérico - usar abordagem padrão'
        }

    @staticmethod
    def get_commodity_specific_parameters(symbol: str) -> Dict:
        """Parâmetros específicos para commodities com cache inteligente"""
        if CACHE_AVAILABLE:
            return global_cache.get_for_or_refresh(
                pair=symbol,
                operation_type="commodity_parameters", 
                params={},
                refresh_fn=lambda: AdvancedAnalyser._get_commodity_params_original(symbol),
                expiry=43200,  # 12 horas - parâmetros são estáticos
                cache_namespace="commodity_config"
            )
        else:
            return AdvancedAnalyser._get_commodity_params_original(symbol)

    @staticmethod
    def _get_commodity_params_original(symbol: str) -> Dict:
        """Implementação original com expansão para mais símbolos"""
        symbol_clean = symbol.replace("/", "").replace("_", "").replace("-", "").upper()
        
        params = {
            # FOREX-COMMODITY PAIRS
            'XAUUSD': {
                'market_type': 'forex_commodity',
                'contract_size': 100,
                'tick_size': 0.01,
                'typical_daily_range': 1.2,
                'volume_multiplier': 100,
                'correlated_assets': ['USD', 'US10Y', 'SPX', 'DXY'],
                'trading_hours': '24/5 con melhor liquidez 14-22h UTC',
                'special_notes': 'Fortemente influenciado por políticas monetárias',
                'volatility_class': 'medium'
            },
            'XAGUSD': {
                'market_type': 'forex_commodity', 
                'contract_size': 5000,
                'tick_size': 0.005,
                'typical_daily_range': 2.5,
                'volume_multiplier': 5000,
                'correlated_assets': ['XAUUSD', 'USD', 'IndustrialMetals'],
                'trading_hours': '24/5 con melhor liquidez 14-22h UTC',
                'special_notes': 'Maior volatilidade que ouro, influência industrial',
                'volatility_class': 'high'
            },
            'XPTUSD': {
                'market_type': 'forex_commodity',
                'contract_size': 100,
                'tick_size': 0.01,
                'typical_daily_range': 1.8,
                'volume_multiplier': 100,
                'correlated_assets': ['XAUUSD', 'USD', 'AutomotiveSector'],
                'volatility_class': 'medium'
            },
            'OILUSD': {
                'market_type': 'forex_commodity',
                'contract_size': 1000,
                'tick_size': 0.01,
                'typical_daily_range': 3.0,
                'volume_multiplier': 1000,
                'correlated_assets': ['USD', 'SPX', 'EnergyStocks', 'GEOPOLITICS'],
                'volatility_class': 'high'
            },
            
            # PURE COMMODITIES (FUTURES/CFDs)
            'GOLD': {
                'market_type': 'precious_metal',
                'contract_size': 100,
                'tick_size': 0.10,
                'typical_daily_range': 1.5,
                'volume_multiplier': 100,
                'correlated_assets': ['USD', 'US10Y', 'SPX'],
                'trading_hours': '24/5',
                'volatility_class': 'medium'
            },
            'SILVER': {
                'market_type': 'precious_metal',
                'contract_size': 5000,
                'tick_size': 0.005,
                'typical_daily_range': 2.5,
                'volume_multiplier': 5000,
                'correlated_assets': ['GOLD', 'USD', 'IndustrialMetals'],
                'volatility_class': 'high'
            },
            'OIL': {
                'market_type': 'energy',
                'contract_size': 1000,
                'tick_size': 0.01,
                'typical_daily_range': 3.0,
                'volume_multiplier': 1000,
                'correlated_assets': ['USD', 'SPX', 'EnergyStocks'],
                'volatility_class': 'high'
            },
            'BRENT': {
                'market_type': 'energy',
                'contract_size': 1000,
                'tick_size': 0.01,
                'typical_daily_range': 2.8,
                'volume_multiplier': 1000,
                'correlated_assets': ['OIL', 'USD', 'SPX'],
                'volatility_class': 'high'
            },
            'WTI': {
                'market_type': 'energy', 
                'contract_size': 1000,
                'tick_size': 0.01,
                'typical_daily_range': 3.2,
                'volume_multiplier': 1000,
                'correlated_assets': ['OIL', 'USD', 'SPX'],
                'volatility_class': 'high'
            },
            'NG': {
                'market_type': 'energy',
                'contract_size': 10000,
                'tick_size': 0.001,
                'typical_daily_range': 4.0,
                'volume_multiplier': 10000,
                'correlated_assets': ['OIL', 'Weather', 'StorageLevels'],
                'volatility_class': 'very_high'
            },
            'NATGAS': {
                'market_type': 'energy',
                'contract_size': 10000,
                'tick_size': 0.001,
                'typical_daily_range': 4.0,
                'volume_multiplier': 10000,
                'correlated_assets': ['OIL', 'Weather', 'StorageLevels'],
                'volatility_class': 'very_high'
            },
            'COPPER': {
                'market_type': 'base_metal',
                'contract_size': 25000,
                'tick_size': 0.0005,
                'typical_daily_range': 2.0,
                'volume_multiplier': 25000,
                'correlated_assets': ['IndustrialProduction', 'CNH', 'GrowthExpectations'],
                'volatility_class': 'medium'
            },
            'PLATINUM': {
                'market_type': 'precious_metal',
                'contract_size': 50,
                'tick_size': 0.10,
                'typical_daily_range': 2.2,
                'volume_multiplier': 50,
                'correlated_assets': ['GOLD', 'Automotive', 'Industrial'],
                'volatility_class': 'medium'
            },
            'PALLADIUM': {
                'market_type': 'precious_metal',
                'contract_size': 100,
                'tick_size': 0.05,
                'typical_daily_range': 3.5,
                'volume_multiplier': 100,
                'correlated_assets': ['PLATINUM', 'Automotive', 'Industrial'],
                'volatility_class': 'high'
            },
            
            # COMMODITY INDICES
            'DXY': {
                'market_type': 'index',
                'contract_size': 1000,
                'tick_size': 0.01,
                'typical_daily_range': 0.8,
                'volume_multiplier': 1000,
                'correlated_assets': ['USD', 'US10Y', 'GOLD'],
                'special_notes': 'Índice dólar - inversamente correlacionado con commodities',
                'volatility_class': 'low'
            }
        }
        
        # BUSCA POR SUBSTRING (mais flexível)
        for key, value in params.items():
            if key in symbol_clean:
                logger.debug(f"Parâmetros específicos encontrados para {symbol}: {key}")
                return value
        
        # FALLBACK PARA COMMODITIES NÃO MAPEADAS
        if any(comm in symbol_clean for comm in ['GOLD', 'XAU', 'GC']):
            fallback = params['GOLD'].copy()
        elif any(comm in symbol_clean for comm in ['SILVER', 'XAG', 'SI']):
            fallback = params['SILVER'].copy() 
        elif any(comm in symbol_clean for comm in ['OIL', 'CL', 'WTI', 'BRENT']):
            fallback = params['OIL'].copy()
        elif any(comm in symbol_clean for comm in ['NG', 'NATGAS']):
            fallback = params['NG'].copy()
        elif any(comm in symbol_clean for comm in ['COPPER', 'HG']):
            fallback = params['COPPER'].copy()
        else:
            # FALLBACK GENÉRICO
            fallback = {
                'market_type': 'unknown_commodity',
                'contract_size': 100,
                'tick_size': 0.01,
                'typical_daily_range': 2.0,
                'volume_multiplier': 100,
                'correlated_assets': [],
                'volatility_class': 'medium'
            }
        
        logger.debug(f"Usando parâmetros fallback para {symbol}")
        return fallback
