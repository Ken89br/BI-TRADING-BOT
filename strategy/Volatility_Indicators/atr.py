import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import talib
from typing import Dict, Optional, Union
from strategy.config_strategy import CONFIG

class ATRStrategy:
    """
    Estratégia ATR Universal Pro:
    - Otimização do período do ATR de acordo com o regime de volatilidade
    - Multi-ATR auto-adaptativo (short/medium/long)
    - Ajuste dinâmico de bandas e parâmetros para cada timeframe (1m até 1d)
    - Boost de confiança via padrões de candlestick (apenas apoio, nunca gatilho)
    - Filtros de tendência (EMA, ADX), momentum (RSI), volume inteligente
    - Compatível com DataFrame universal do pipeline, inclusive com colunas de padrões
    - Sinais de breakout e pullback com múltiplas confirmações
    - Gestão de risco dinâmica e position sizing
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or CONFIG['atr_config']
        self.volume_threshold    = config['volume_threshold']
        self.min_confidence     = config['min_confidence']
        self.position_risk_pct  = config['position_risk_pct']
        self.risk_reward_ratio  = config['risk_reward_ratio']
        self.period_min         = config['period_min']
        self.period_max         = config['period_max']
        self.timeframe_adjustments = config['timeframe_adjustments']
        self.lookback           = config.get("lookback", 14)  # fallback
        self.current_timeframe  = None
        self.current_params     = None

    def set_timeframe(self, timeframe: str) -> None:
        self.current_timeframe = timeframe
        if timeframe not in self.timeframe_adjustments:
            raise ValueError(f"Timeframe '{timeframe}' não definido em config['atr_config']['timeframe_adjustments']")
        self.current_params = self.timeframe_adjustments[timeframe]

    def _calculate_true_range(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def _optimize_atr_period(self, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """
        Otimiza o período do ATR para a volatilidade atual.
        Busca o período que deixa o ATR mais estável (menor variância), para evitar sinais falsos em regimes anormais.
        """
        def objective(period):
            period = int(round(period))
            tr = self._calculate_true_range(high, low, close)
            atr = tr.rolling(window=period).mean()
            return atr.dropna().var()
        result = minimize_scalar(objective, bounds=(self.period_min, self.period_max), method='bounded')
        return int(round(result.x))

    def _calculate_multi_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ATR short/medium/long para o timeframe"""
        periods = {
            '1m': (5, 14, 60),
            '5m': (7, 21, 80),
            '15m': (10, 21, 100),
            '1h': (14, 21, 100),
            '4h': (14, 50, 150),
            '1d': (14, 50, 200)
        }
        if self.current_timeframe not in periods:
            raise ValueError(f"Timeframe '{self.current_timeframe}' não suportado")
        p_short, p_medium, p_long = periods[self.current_timeframe]
        df['ATR_short'] = self._calculate_atr(df, p_short)
        df['ATR_medium'] = self._calculate_atr(df, p_medium)
        df['ATR_long'] = self._calculate_atr(df, p_long)
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    
    def _select_best_atr(self, df: pd.DataFrame, lookback: Optional[int] = None) -> pd.Series:
    """
    Seleciona dinamicamente o ATR mais apropriado ao regime do mercado, usando lookback institucional.
    """
        lookback = lookback or self.lookback
        df = df.copy()
        df['range'] = df['high'] - df['low']
        df['avg_range'] = df['range'].rolling(lookback).mean()
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['range']
        df['body_ratio'] = (df['close'] - df['open']).abs() / df['range']
        df['doji'] = df['body_ratio'] < 0.1
        df['doji_ratio'] = df['doji'].rolling(lookback).mean()
        df['volatility_std'] = df['close'].rolling(lookback).std()
        selected_atr = []
        for i in range(len(df)):
            high_volatility = df['volatility_std'].iloc[i] > (df['avg_range'].iloc[i] * 1.2)
            low_noise = (df['wick_ratio'].iloc[i] < 0.3) and (df['doji_ratio'].iloc[i] < 0.2)
            strong_trend = (df['adx_value'].iloc[i] > 25) if 'adx_value' in df else False
            high_volume = df['volume'].iloc[i] > df['volume'].rolling(lookback).mean().iloc[i] * 1.5
            if strong_trend and high_volatility and high_volume:
                selected_atr.append(df['ATR_short'].iloc[i])
            elif not strong_trend and (df['doji_ratio'].iloc[i] > 0.3 or df['wick_ratio'].iloc[i] > 0.4):
                selected_atr.append(df['ATR_long'].iloc[i])
            else:
                selected_atr.append(df['ATR_medium'].iloc[i])
        return pd.Series(selected_atr, index=df.index)

    def _calculate_adaptive_bands(self, close: pd.Series, atr: float) -> tuple:
        """Bandas dinâmicas baseadas no ATR"""
        if not self.current_params:
            raise ValueError("Timeframe não definido. Chame set_timeframe() primeiro.")
        multiplier = self.current_params['atr_multiplier']
        upper_band = close.iloc[-1] + (atr * multiplier)
        lower_band = close.iloc[-1] - (atr * multiplier)
        return upper_band, lower_band

    def _get_trend_direction(self, close: pd.Series, ema_short: pd.Series, ema_long: pd.Series) -> str:
        price = close.iloc[-1]
        ema_s = ema_short.iloc[-1]
        ema_l = ema_long.iloc[-1]
        if price > ema_s > ema_l:
            return 'up'
        elif price < ema_s < ema_l:
            return 'down'
        return 'neutral'

    def _calculate_volume_profile(self, volume: pd.Series, lookback: int = 20) -> dict:
        recent_vol = volume.iloc[-lookback:]
        avg_vol = recent_vol.mean()
        current_vol = volume.iloc[-1]
        return {
            'avg_volume': avg_vol,
            'current_volume': current_vol,
            'volume_ratio': current_vol / avg_vol if avg_vol > 0 else 1,
            'is_volume_spike': current_vol > (avg_vol * self.volume_threshold)
        }

    def _pattern_confidence_boost(self, patterns: Optional[Union[list, str]], direction: str) -> int:
        """
        Dá boost de confiança caso padrões relevantes estejam presentes.
        Não é gatilho de entrada, só apoio/confirmação.
        """
        if not patterns or not isinstance(patterns, list):
            return 0
        from strategy.candlestick_patterns import REVERSAL_UP, REVERSAL_DOWN, CONTINUATION
        boost = 0
        if direction == "up":
            relevant = set(REVERSAL_UP + CONTINUATION)
        elif direction == "down":
            relevant = set(REVERSAL_DOWN + CONTINUATION)
        else:
            relevant = set()
        # Cada padrão relevante aumenta 2 pontos de confiança
        boost = sum(2 for p in patterns if p in relevant)
        return min(10, boost)  # Limite máximo de boost

    def generate_signal(self, df: pd.DataFrame, account_balance: Optional[float] = None) -> Optional[dict]:
        """
        Gera sinais de trading adaptativos
        Compatível com DataFrame universal do pipeline (precisa de colunas: open, high, low, close, volume, patterns, ADX)
        """
        try:
            if df is None or len(df) < self.period_max * 2:
                return None
            if not self.current_timeframe:
                raise ValueError("Timeframe não definido. Chame set_timeframe() primeiro.")

            # 1. Otimização do período do ATR (opcional, ultra-adaptativo)
            opt_period = self._optimize_atr_period(df['high'], df['low'], df['close'])
            df['ATR_opt'] = self._calculate_atr(df, opt_period)

            # 2. Multi-ATR adaptativo
            df = self._calculate_multi_atr(df)
            df['ATR_selected'] = self._select_best_atr(df)
            atr = df['ATR_selected'].iloc[-1]

            # 3. Indicadores adaptativos por timeframe
            rsi_period = self.current_params['rsi_period']
            ema_short_period = self.current_params['ema_short']
            ema_long_period = self.current_params['ema_long']

            rsi = talib.RSI(df['close'], timeperiod=rsi_period)
            ema_short = df['close'].ewm(span=ema_short_period, adjust=False).mean()
            ema_long  = df['close'].ewm(span=ema_long_period, adjust=False).mean()
            
            trend = self._get_trend_direction(df['close'], ema_short, ema_long)
            rsi_value = rsi.iloc[-1]
            momentum = 'bullish' if rsi_value > 50 else 'bearish'

            # 4. Bandas dinâmicas e análise de preço
            upper_band, lower_band = self._calculate_adaptive_bands(df['close'], atr)
            last_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df['close']) > 1 else last_close

            # 5. Volume profile
            volume_profile = self._calculate_volume_profile(df['volume'])

            # 6. Boost confiança via padrões de candle
            patterns = df['patterns'].iloc[-1] if 'patterns' in df.columns else []
            pattern_boost = self._pattern_confidence_boost(patterns, trend)

            # 7. Sinais com múltiplas confirmações
            signal = None
            confidence = 0
            event = ""
            # Regra 1: Breakout com tendência/momentum
            if (last_close > upper_band and prev_close <= upper_band and trend == 'up' and momentum == 'bullish'):
                signal = 'up'
                event = 'breakout_with_trend'
                confidence = 80
            elif (last_close < lower_band and prev_close >= lower_band and trend == 'down' and momentum == 'bearish'):
                signal = 'down'
                event = 'breakout_with_trend'
                confidence = 80
            # Regra 2: Pullback com volume spike
            elif (trend == 'up' and last_close > ema_short.iloc[-1] and 40 < rsi_value < 70 and volume_profile['is_volume_spike']):
                signal = 'up'
                event = 'pullback_with_volume'
                confidence = 75
            elif (trend == 'down' and last_close < ema_short.iloc[-1] and 30 < rsi_value < 60 and volume_profile['is_volume_spike']):
                signal = 'down'
                event = 'pullback_with_volume'
                confidence = 75

            # 8. Aplica boost por padrão (apenas se sinal foi gerado)
            if signal:
                confidence = min(100, confidence + pattern_boost)

            if not signal or confidence < self.min_confidence:
                return None

            # 9. Stop, take profit e sizing
            if signal == 'up':
                stop_loss = min(df['low'].iloc[-1], df['low'].iloc[-2] if len(df['low']) > 1 else df['low'].iloc[-1]) - (atr * 0.5)
                take_profit = last_close + (2 * atr)
            else:
                stop_loss = max(df['high'].iloc[-1], df['high'].iloc[-2] if len(df['high']) > 1 else df['high'].iloc[-1]) + (atr * 0.5)
                take_profit = last_close - (2 * atr)
            position_size = None
            if account_balance and stop_loss:
                risk_amount = account_balance * self.position_risk_pct
                stop_distance = abs(last_close - stop_loss)
                position_size = round(risk_amount / stop_distance, 2) if stop_distance > 0 else 0

            # 10. Monta resultado universal
            result = {
                'signal': signal,
                'event': event,
                'confidence': confidence,
                'price': last_close,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size': position_size,
                'atr': round(atr, 5),
                'atr_period': 'dynamic',
                'atr_period_opt': opt_period,
                'rsi': round(rsi_value, 2),
                'trend': trend,
                'momentum': momentum,
                'volume_ratio': round(volume_profile['volume_ratio'], 2),
                'timeframe': self.current_timeframe,
                'timestamp': df.index[-1],
                'atr_type': self._get_atr_type(df)
            }

            return result

        except Exception as e:
            print(f"ATRStrategy error: {str(e)}")
            return None

    def _get_atr_type(self, df: pd.DataFrame) -> str:
        """Identifica qual ATR está sendo usado (short/medium/long)"""
        current_atr = df['ATR_selected'].iloc[-1]
        if np.isclose(current_atr, df['ATR_short'].iloc[-1]):
            return 'short'
        elif np.isclose(current_atr, df['ATR_medium'].iloc[-1]):
            return 'medium'
        else:
            return 'long'
