# strategy/ai_filter.py
# AI Filter aprimorado: rejeita/penaliza sinais com base na força de padrões de candlestick,
# além dos critérios clássicos de qualidade, risco, volatilidade, suporte/resistência, etc.
# Filtro AI robustecido com controle aprimorado de penalização acumulada

import logging
from typing import Dict, List, Optional, Tuple, Union
from strategy.candlestick_patterns import PATTERN_STRENGTH

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_filter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SmartAIFilter:
    def __init__(
        self,
        min_confidence: int = 40,
        min_volatility: float = 0.0001,
        min_volume: float = 150,
        allowed_risk: Tuple[str, ...] = ("low", "moderate"),
        allowed_volatility: Tuple[str, ...] = ("high", "moderate"),
        min_support_distance: float = 0.0001,
        min_pattern_reject_strength: float = 0.85,
        pattern_penalty_factor: int = 8,
        price_tolerance: float = 0.001,
        max_total_penalty: int = 30  # Limite máximo de penalidade acumulada
    ):
        if min_confidence < 10 or min_confidence > 90:
            raise ValueError("min_confidence deve estar entre 10 e 90")
        if min_volatility <= 0:
            raise ValueError("min_volatility deve ser positivo")

        self.min_confidence = min_confidence
        self.min_volatility = min_volatility
        self.min_volume = min_volume
        self.allowed_risk = set(allowed_risk)
        self.allowed_volatility = set(allowed_volatility)
        self.min_support_distance = min_support_distance
        self.min_pattern_reject_strength = min_pattern_reject_strength
        self.pattern_penalty_factor = pattern_penalty_factor
        self.price_tolerance = price_tolerance
        self.max_total_penalty = max_total_penalty

    def _get_numeric_value(self, value: Union[str, float, int]) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        try:
            cleaned = value.replace('%', '').replace('(', '').replace(')', '').strip()
            return float(cleaned.split()[0]) if cleaned else None
        except (ValueError, TypeError):
            return None

    def pattern_strength(self, patterns: List[str], direction: str) -> float:
        if not patterns or not direction:
            return 0.0
        total_strength = 0.0
        valid_direction = direction.lower() in ("up", "down")
        for pattern in patterns:
            if not isinstance(pattern, str):
                continue
            p_lower = pattern.lower()
            strength = PATTERN_STRENGTH.get(p_lower, 0.1)
            if valid_direction and direction.lower() == "up":
                if any(keyword in p_lower for keyword in ["bear", "engulfing", "shooting", "dark_cloud", "evening_star"]):
                    total_strength += strength
            elif valid_direction and direction.lower() == "down":
                if any(keyword in p_lower for keyword in ["bull", "hammer", "morning_star", "piercing"]):
                    total_strength += strength
            if "doji" in p_lower:
                total_strength += 0.1
        return min(total_strength, 5.0)

    def _apply_volume_filter(self, signal_data: Dict, volume: float, penalties: List[int]) -> bool:
        if volume < self.min_volume:
            penalties.append(25)
            signal_data["rejection_reason"] = "volume_below_min"
            logger.warning(f"Volume abaixo do mínimo: {volume} < {self.min_volume}")
            return False
        return True

    def _apply_candle_filter(self, signal_data: Dict, body_ratio: float, penalties: List[int]) -> bool:
        if body_ratio < 0.2:
            penalties.append(20)
            signal_data["rejection_reason"] = "weak_candle_body"
            logger.warning(f"Corpo do candle fraco: {body_ratio:.2f}")
            return False
        return True

    def _apply_pattern_filter(self, signal_data: Dict, patterns: List[str], direction: str, penalties: List[int]) -> bool:
        if not patterns or not direction:
            return True
        pattern_strength = self.pattern_strength(patterns, direction)
        if pattern_strength >= self.min_pattern_reject_strength:
            signal_data["rejection_reason"] = f"strong_contrary_patterns ({pattern_strength:.2f})"
            logger.warning(f"Rejeitado: força de padrões contrários {pattern_strength:.2f} > {self.min_pattern_reject_strength}")
            return False
        elif pattern_strength > 0:
            penalty = int(pattern_strength * self.pattern_penalty_factor)
            penalties.append(penalty)
            signal_data["pattern_penalty"] = penalty
            logger.info(f"Penalização aplicada: {penalty} pontos (força: {pattern_strength:.2f})")
        return True

    def _apply_technical_filters(self, signal_data: Dict, direction: str, penalties: List[int]) -> bool:
        # RSI
        rsi_val = self._get_numeric_value(signal_data.get("rsi", ""))
        if rsi_val is not None:
            if direction == "up" and rsi_val > 75:
                logger.warning(f"Rejeitado: RSI sobrecompra ({rsi_val})")
                return False
            if direction == "down" and rsi_val < 25:
                logger.warning(f"Rejeitado: RSI sobrevenda ({rsi_val})")
                return False
        # MACD
        macd_data = signal_data.get("macd", {})
        if isinstance(macd_data, dict):
            macd_hist = macd_data.get("hist", self._get_numeric_value(macd_data.get("histogram")))
            if macd_hist is not None:
                if direction == "up" and macd_hist < -0.05:
                    penalties.append(10)
                elif direction == "down" and macd_hist > 0.05:
                    penalties.append(10)
        # Bollinger Bands
        bb_data = signal_data.get("bollinger", {})
        if isinstance(bb_data, dict):
            bb_pos = bb_data.get("pos", self._get_numeric_value(bb_data.get("position")))
            if bb_pos is not None:
                if direction == "up" and bb_pos < -0.1:
                    penalties.append(10)
                elif direction == "down" and bb_pos > 0.1:
                    penalties.append(10)
        # Osciladores e Médias
        for indicator in ["osc_rating", "ma_rating"]:
            value = signal_data.get(indicator)
            if value is None:
                continue
            if direction == "up" and value == -1:
                penalties.append(10)
            elif direction == "down" and value == 1:
                penalties.append(10)
        return True

    def _apply_risk_filters(self, signal_data: Dict) -> bool:
        risk = str(signal_data.get("risk", "")).lower()
        if risk and risk not in self.allowed_risk:
            logger.warning(f"Rejeitado: risco não permitido ({risk})")
            return False
        volatility = str(signal_data.get("volatility", "")).lower()
        if volatility and volatility not in self.allowed_volatility:
            logger.warning(f"Rejeitado: volatilidade não permitida ({volatility})")
            return False
        sentiment = signal_data.get("sentiment")
        direction = signal_data.get("signal", "")
        if sentiment is not None and direction:
            if direction == "up" and sentiment < -0.5:
                logger.warning("Rejeitado: sinal de alta com sentimento negativo")
                return False
            if direction == "down" and sentiment > 0.5:
                logger.warning("Rejeitado: sinal de baixa com sentimento positivo")
                return False
        return True

    def _apply_price_filters(self, signal_data: Dict, price: float) -> bool:
        direction = signal_data.get("signal", "")
        if not direction or price <= 0:
            return True
        support = self._get_numeric_value(signal_data.get("support"))
        resistance = self._get_numeric_value(signal_data.get("resistance"))
        if direction == "up" and resistance is not None:
            distance = resistance - price
            if distance < self.min_support_distance and distance > -self.price_tolerance:
                logger.warning(f"Rejeitado: compra muito próxima da resistência ({distance:.6f})")
                return False
        if direction == "down" and support is not None:
            distance = price - support
            if distance < self.min_support_distance and distance > -self.price_tolerance:
                logger.warning(f"Rejeitado: venda muito próxima do suporte ({distance:.6f})")
                return False
        variation = self._get_numeric_value(signal_data.get("variation"))
        if variation is not None and abs(variation) > 3.0:
            logger.warning(f"Rejeitado: variação extrema ({variation}%)")
            return False
        return True

    def apply(self, signal_data: Dict, candles: List[Dict]) -> Optional[Dict]:
        """
        Aplica todos os filtros sequencialmente, controla penalização acumulada
        """
        try:
            if not signal_data or not candles:
                logger.error("Entrada inválida: signal_data ou candles vazios")
                return None
            if not isinstance(signal_data, dict) or not isinstance(candles, list):
                logger.error("Tipos inválidos: signal_data deve ser dict, candles list")
                return None
            latest_candle = candles[-1]
            required_keys = {"open", "high", "low", "close", "volume"}
            if not required_keys.issubset(latest_candle.keys()):
                logger.error("Candle incompleto: faltam campos essenciais")
                return None

            price = float(latest_candle["close"])
            open_price = float(latest_candle["open"])
            body_size = abs(price - open_price)
            wick_size = abs(float(latest_candle["high"]) - float(latest_candle["low"]))
            body_ratio = body_size / (wick_size + 1e-10)
            volume = float(latest_candle.get("volume", 0))
            direction = signal_data.get("signal", "")

            if "confidence" not in signal_data:
                signal_data["confidence"] = 50

            penalties = []

            filters = [
                lambda: self._apply_volume_filter(signal_data, volume, penalties),
                lambda: self._apply_candle_filter(signal_data, body_ratio, penalties),
                lambda: self._apply_pattern_filter(
                    signal_data, 
                    signal_data.get("patterns", []), 
                    direction,
                    penalties
                ),
                lambda: self._apply_technical_filters(signal_data, direction, penalties),
                lambda: self._apply_risk_filters(signal_data),
                lambda: self._apply_price_filters(signal_data, price)
            ]

            for filter_fn in filters:
                if not filter_fn():
                    return None

            # Penalização acumulada controlada
            total_penalty = min(sum(penalties), self.max_total_penalty)
            old_conf = signal_data["confidence"]
            signal_data["confidence"] = max(self.min_confidence, signal_data["confidence"] - total_penalty)
            signal_data["total_penalty"] = total_penalty
            if total_penalty > 0:
                logger.info(f"Penalidade total aplicada: {total_penalty} (confiança de {old_conf} para {signal_data['confidence']})")

            price_range = float(signal_data.get("high", 0)) - float(signal_data.get("low", 0))
            if price_range < self.min_volatility:
                logger.warning(f"Rejeitado: volatilidade baixa ({price_range:.6f})")
                return None

            if signal_data["confidence"] < self.min_confidence:
                logger.warning(f"Rejeitado: confiança baixa ({signal_data['confidence']}%)")
                return None

            logger.info(f"Sinal aprovado com confiança {signal_data['confidence']}%")
            return signal_data

        except Exception as e:
            logger.error(f"Erro crítico no AI Filter: {str(e)}", exc_info=True)
            return None
