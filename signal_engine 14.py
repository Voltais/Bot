"""
SIGNAL ENGINE V14.0 – с поддержкой гибких условий входа, ослабленными порогами и fallback-механизмами
- Основной сигнал (SMC, трендовый) – теперь не требует обязательного свипа и структуры
- Контртрендовый сигнал (от уровней в диапазоне) – расширенный коридор и смягченные условия входа
- Убран жесткий фильтр range_market для контртренда (работает и в трендах)
- Всегда возвращает MultiSignal с двумя полями: swing и counter_trend
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import logging
import time
import math

logger = logging.getLogger(__name__)

# ===== RISK PARAMETERS =====
class Risk:
    MIN_STOP_PERCENT = 0.015           # Увеличено с 0.010 для более широких и надежных стопов (для ETH это 1.5%)
    STRUCTURE_BUFFER_ATR = 0.20
    MAX_CONFIDENCE = 0.92
    MIN_CONFIDENCE_SWING = 0.30
    MIN_CONFIDENCE_COUNTER = 0.30

    RR_BASE = 1
    RR_GOOD = 1.5
    RR_STRONG = 2.0

    TTL = {
        "BTCUSDT": 48 * 3600,
        "ETHUSDT": 36 * 3600,
        "DEFAULT": 24 * 3600,
    }
    DISTANCE_ATR_MULT = 2.0
    DISTANCE_MIN = 0.02
    DISTANCE_MAX = 0.05
    TP1_SAFETY_FACTOR = 0.9

    STRUCTURE_BUFFER_PERCENT = 0.005
    STRUCTURE_BUFFER_ATR_MULT = 0.3

    MATCH_TOL_ENTRY = 0.002
    MATCH_TOL_STOP = 0.003
    MATCH_TOL_TP1 = 0.005
    ANTI_RECREATION_WINDOW = 3600

    WEIGHT_1D = 0.30
    WEIGHT_4H = 0.25
    WEIGHT_1H = 0.20
    WEIGHT_15M = 0.15
    WEIGHT_VOLUME = 0.10

    ADX_TREND = 25
    ADX_RANGE = 20
    RSI_EXTREME_HIGH = 75
    RSI_EXTREME_LOW = 25
    RVOL_CONFIRM = 1.2
    SWEEP_ATR_MULT = 0.2
    MIN_LIQ_DISTANCE_PERCENT = 0.005
    MIN_LIQ_DISTANCE_ATR_MULT = 0.5
    DISPLACEMENT_MULT = 1.2

    CONFLUENCE_OB = 0.4
    CONFLUENCE_FVG = 0.3
    CONFLUENCE_LIQ = 0.3

    # Контртрендовые параметры (V14.0 - расширенные коридоры для ETH и подобных)
    COUNTER_TREND_MIN_TP_PERCENT = 0.04  # Минимальный тейк 4% от цены (для ETH ~ $100+)
    COUNTER_TREND_ENTRY_DIST_PERCENT = 0.03 # Расширенное расстояние до уровня
    COUNTER_TREND_STOP_ATR_MULT = 2.5    # Стоп = 2.5 * ATR(4H) (обычно $100-120)
    COUNTER_TREND_TTL_HOURS = 6          # Интрадей сигнал живет 6 часов
    COUNTER_TREND_MIN_VOLUME = 0.0
    COUNTER_TREND_RSI_SHORT = 70         # Порог RSI для шорта (у вас 86 -> отлично)
    COUNTER_TREND_RSI_LONG = 30          # Порог RSI для лонга
    COUNTER_TREND_ALLOW_MARKET_ENTRY = True  # Разрешаем вход по рынку при экстремальном RSI

TICK_SIZES = {
    "BTCUSDT": 0.10,
    "ETHUSDT": 0.01,
    "ETHBTC": 0.000001,
}

def get_tick_size(price: float) -> float:
    if price < 0.1:
        return 0.0001
    elif price < 1:
        return 0.001
    else:
        return 0.01

def round_to_tick(symbol: str, price: float) -> float:
    tick = get_tick_size(price)
    return round(price / tick) * tick

def round_stop(symbol: str, stop: float, direction: str) -> float:
    tick = get_tick_size(stop)
    if direction == "LONG":
        return math.floor(stop / tick) * tick
    else:
        return math.ceil(stop / tick) * tick

@dataclass
class MarketState:
    symbol: str
    price: float
    close_4h: float
    close_1h: float
    close_15m: float
    trend_1d: str
    trend_4h: str
    trend_1h: str
    trend_15m: str
    structure_1d: str
    structure_4h: str
    structure_1h: str
    structure_15m: str
    rsi_1d: float
    rsi_4h: float
    rsi_1h: float
    rsi_15m: float
    adx_1d: float
    adx_4h: float
    adx_1h: float
    atr_1d: float
    atr_4h: float
    atr_1h: float
    atr_15m: float
    volume_trend_1h: str
    volume_trend_4h: str
    relative_volume_1h: float
    relative_volume_4h: float
    cvd_1h: float
    cvd_4h: float
    support_1d: float
    resistance_1d: float
    support_4h: float
    resistance_4h: float
    support_1h: float
    resistance_1h: float
    support_15m: float
    resistance_15m: float
    liquidity_1h: Optional[Dict[str, List[float]]] = None
    liquidity_4h: Optional[Dict[str, List[float]]] = None
    liquidity_1d: Optional[Dict[str, List[float]]] = None
    vwap: Optional[float] = None
    oi: float = 0.0
    funding: float = 0.0
    btc_dominance: float = 0.0
    alt_season: bool = False
    liquidation_long: float = 0.0
    liquidation_short: float = 0.0
    ls_ratio: float = 1.0
    fear_greed: Optional[int] = None
    ema50_1h: float = 0.0
    ema200_1h: float = 0.0
    ema50_15m: float = 0.0
    ema200_15m: float = 0.0
    ema50_4h: float = 0.0
    ema200_4h: float = 0.0
    macd_4h: float = 0.0
    macd_signal_4h: float = 0.0
    fib_levels: Optional[Dict[str, float]] = None
    range_market: bool = False

@dataclass
class TradeSetup:
    direction: str
    entry: Optional[float]
    stop: Optional[float]
    tp1: Optional[float]
    tp2: Optional[float]
    confidence: float
    reason: str
    timeframe: str
    pd_zone: str = ""
    sweep_detected: bool = False
    entry_type: str = "market"
    is_counter_trend: bool = False
    tp1_percent: int = 50
    tp2_percent: int = 30
    runner_percent: int = 20
    hold_note: str = ""

@dataclass
class MultiSignal:
    swing: TradeSetup
    counter_trend: Optional[TradeSetup] = None
    regime: str = ""
    global_trend: str = ""
    context: str = ""
    is_active_signal: bool = False
    signal_age_hours: float = 0.0

@dataclass
class Signal:
    symbol: str
    direction: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    confidence: float
    reason: str
    timeframe: str
    pd_zone: str
    sweep_detected: bool
    created_at: float
    updated_at: float
    status: str = "ACTIVE"
    entry_support: Optional[float] = None
    entry_resistance: Optional[float] = None
    distance_limit: float = 0.0
    tp1_reached_at: Optional[float] = None
    entry_type: str = "market"
    is_counter_trend: bool = False

class SignalStorage:
    def __init__(self):
        self._signals: Dict[str, Signal] = {}
        self._last_deleted: Dict[str, Tuple[Signal, float]] = {}
        self._on_set = None
        self._on_delete = None

    def set_persistence_hooks(self, on_set=None, on_delete=None) -> None:
        self._on_set = on_set
        self._on_delete = on_delete

    def load_from(self, signals: List["Signal"]) -> None:
        for s in signals:
            self._signals[s.symbol] = s

    def get(self, symbol: str) -> Optional[Signal]:
        return self._signals.get(symbol)

    def set(self, signal: Signal) -> None:
        signal.updated_at = time.time()
        self._signals[signal.symbol] = signal
        if self._on_set:
            try:
                self._on_set(signal)
            except Exception:
                logger.error(f"Ошибка сохранения сигнала {signal.symbol} в БД", exc_info=True)

    def delete(self, symbol: str) -> None:
        if symbol in self._signals:
            self._last_deleted[symbol] = (self._signals[symbol], time.time())
            del self._signals[symbol]
            if self._on_delete:
                try:
                    self._on_delete(symbol)
                except Exception:
                    logger.error(f"Ошибка удаления сигнала {symbol} из БД", exc_info=True)

    def get_last_deleted(self, symbol: str) -> Optional[Tuple[Signal, float]]:
        return self._last_deleted.get(symbol)

    def clear(self) -> None:
        self._signals.clear()
        self._last_deleted.clear()

storage = SignalStorage()

# ========== Вспомогательные функции ==========
def get_pd_zone(price: float, support: float, resistance: float) -> str:
    if resistance <= support:
        return "neutral"
    range_size = resistance - support
    premium_threshold = support + (range_size * 0.62)
    discount_threshold = support + (range_size * 0.38)
    if price > premium_threshold:
        return "premium"
    elif price < discount_threshold:
        return "discount"
    return "neutral"

def global_trend(t1d: str, t4h: str, t1h: str) -> str:
    score = 0
    if t1d == "bullish": score += 3
    if t4h == "bullish": score += 2
    if t1h == "bullish": score += 1
    if t1d == "bearish": score -= 3
    if t4h == "bearish": score -= 2
    if t1h == "bearish": score -= 1
    if score >= 2: return "bullish"
    if score <= -2: return "bearish"
    return "sideways"

def detect_regime(adx_4h: float, adx_1d: float) -> str:
    if adx_4h > Risk.ADX_TREND or adx_1d > Risk.ADX_TREND:
        return "TREND"
    if adx_4h < Risk.ADX_RANGE and adx_1d < Risk.ADX_RANGE:
        return "RANGE"
    return "WEAK_TREND"

def adaptive_atr(a1: float, a4: float, a1d: float, price: float) -> float:
    avg = (a1 + a4 + a1d) / 3
    ratio = avg / price
    if ratio > 0.02: m = 1.25
    elif ratio > 0.015: m = 1.1
    elif ratio > 0.01: m = 1.0
    else: m = 0.85
    return avg * m

def liquidity_targets(price: float, s1: float, s4: float, r1: float, r4: float, liquidity: Optional[Dict[str, List[float]]] = None, min_dist: float = None) -> Tuple[Optional[float], Optional[float]]:
    if liquidity:
        buy_side = liquidity.get("buy_side", [])
        sell_side = liquidity.get("sell_side", [])
        if min_dist is not None:
            buy_side = [lvl for lvl in buy_side if lvl > price and (lvl - price) >= min_dist]
            sell_side = [lvl for lvl in sell_side if lvl < price and (price - lvl) >= min_dist]
        above_liq = min(buy_side) if buy_side else None
        below_liq = max(sell_side) if sell_side else None
        return above_liq, below_liq
    up = [x for x in [r1, r4] if x > price]
    down = [x for x in [s1, s4] if x < price]
    above_liq = min(up) if up else None
    below_liq = max(down) if down else None
    return above_liq, below_liq

def structural_stop(direction: str, support: float, resistance: float, atr: float) -> float:
    buffer = atr * Risk.STRUCTURE_BUFFER_ATR
    if direction == "LONG":
        return support - buffer
    return resistance + buffer

def enforce_min_stop(entry: float, stop: float) -> float:
    min_dist = entry * Risk.MIN_STOP_PERCENT
    if abs(entry - stop) < min_dist:
        if stop < entry:
            stop = entry - min_dist
        else:
            stop = entry + min_dist
    return stop

def target_engine(direction: str, entry: float, stop: float, above_liq: Optional[float], below_liq: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    risk = abs(entry - stop)
    if risk <= 0:
        return None, None
    if direction == "LONG":
        tp1 = above_liq if above_liq and above_liq > entry else entry + risk * 2
        tp2 = tp1 + risk * 1.5 if tp1 else entry + risk * 3.5
    else:
        tp1 = below_liq if below_liq and below_liq < entry else entry - risk * 2
        tp2 = tp1 - risk * 1.5 if tp1 else entry - risk * 3.5
    return tp1, tp2

def calculate_rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    reward = abs(target - entry)
    return reward / risk

def dynamic_rr(state: MarketState) -> float:
    if state.relative_volume_1h > 1.5:
        return Risk.RR_STRONG
    elif state.relative_volume_1h > 1.2:
        return Risk.RR_GOOD
    return Risk.RR_BASE

def get_liquidity_levels(state: MarketState) -> Optional[Dict[str, List[float]]]:
    return state.liquidity_1h or state.liquidity_4h

def detect_sweep_v10_6(state: MarketState, candles_15m: List[Tuple[float, float, float, float, float]], bias: str, atr: float) -> Optional[Tuple[str, int]]:
    liq = get_liquidity_levels(state)
    if not liq:
        return None
    min_dist = max(state.price * Risk.MIN_LIQ_DISTANCE_PERCENT, atr * Risk.MIN_LIQ_DISTANCE_ATR_MULT)
    if bias == "bullish":
        sell_levels = liq.get("sell_side", [])
        if not sell_levels:
            return None
        valid_sell = [lvl for lvl in sell_levels if lvl < state.price and (state.price - lvl) >= min_dist]
        if not valid_sell:
            return None
        closest_sell = max(valid_sell)
        for i in range(2, len(candles_15m)):
            low = candles_15m[i][1]
            prev_low = candles_15m[i-1][1]
            if low < closest_sell and (prev_low - low) > atr * Risk.SWEEP_ATR_MULT:
                return ("bullish", i)
    else:
        buy_levels = liq.get("buy_side", [])
        if not buy_levels:
            return None
        valid_buy = [lvl for lvl in buy_levels if lvl > state.price and (lvl - state.price) >= min_dist]
        if not valid_buy:
            return None
        closest_buy = min(valid_buy)
        for i in range(2, len(candles_15m)):
            high = candles_15m[i][0]
            prev_high = candles_15m[i-1][0]
            if high > closest_buy and (high - prev_high) > atr * Risk.SWEEP_ATR_MULT:
                return ("bearish", i)
    return None

def detect_displacement_multi_directional(candles_15m: List[Tuple[float, float, float, float, float]], start_idx: int, atr: float, bias: str) -> bool:
    if start_idx >= len(candles_15m):
        return False
    close = candles_15m[start_idx][2]
    open_ = candles_15m[start_idx][3]
    if bias == "bullish" and close > open_:
        return True
    if bias == "bearish" and close < open_:
        return True
    return False

def detect_order_block(candles_4h: List[Tuple[float, float, float, float, float]], bias: str) -> Optional[float]:
    if len(candles_4h) < 10:
        return None
    avg_vol = sum(c[4] for c in candles_4h[-20:]) / 20
    for i in range(len(candles_4h)-3, 2, -1):
        if bias == "bullish":
            if candles_4h[i][2] < candles_4h[i][3] and candles_4h[i+1][2] > candles_4h[i+1][3]:
                body_prev = abs(candles_4h[i][2] - candles_4h[i][3])
                body_next = abs(candles_4h[i+1][2] - candles_4h[i+1][3])
                if body_next > body_prev * 1.5 and candles_4h[i+1][4] > avg_vol * 1.5:
                    return candles_4h[i][3]
        else:
            if candles_4h[i][2] > candles_4h[i][3] and candles_4h[i+1][2] < candles_4h[i+1][3]:
                body_prev = abs(candles_4h[i][2] - candles_4h[i][3])
                body_next = abs(candles_4h[i+1][2] - candles_4h[i+1][3])
                if body_next > body_prev * 1.5 and candles_4h[i+1][4] > avg_vol * 1.5:
                    return candles_4h[i][3]
    return None

def detect_fvg(candles_4h: List[Tuple[float, float, float, float, float]], bias: str) -> Optional[float]:
    for i in range(2, len(candles_4h)):
        if candles_4h[i][1] > candles_4h[i-2][0] and bias == "bullish":
            return candles_4h[i][1]
        if candles_4h[i][0] < candles_4h[i-2][1] and bias == "bearish":
            return candles_4h[i][0]
    return None

def get_liquidity_zone(state: MarketState, bias: str) -> Optional[float]:
    if not state.liquidity_4h:
        return None
    min_dist = max(state.price * Risk.MIN_LIQ_DISTANCE_PERCENT, state.atr_4h * Risk.MIN_LIQ_DISTANCE_ATR_MULT)
    if bias == "bullish":
        levels = [lvl for lvl in state.liquidity_4h.get("sell_side", []) if lvl < state.price]
        if not levels:
            return None
        valid_levels = [lvl for lvl in levels if (state.price - lvl) >= min_dist]
        if not valid_levels:
            return None
        return max(valid_levels)
    else:
        levels = [lvl for lvl in state.liquidity_4h.get("buy_side", []) if lvl > state.price]
        if not levels:
            return None
        valid_levels = [lvl for lvl in levels if (lvl - state.price) >= min_dist]
        if not valid_levels:
            return None
        return min(valid_levels)

def compute_poi_confluence(state: MarketState, bias: str, candles_4h: List[Tuple[float, float, float, float, float]]) -> Tuple[Optional[float], float]:
    candidates = []
    liq = get_liquidity_zone(state, bias)
    if liq is not None:
        if (bias == "bullish" and liq < state.price) or (bias == "bearish" and liq > state.price):
            candidates.append((liq, Risk.CONFLUENCE_LIQ))
    ob = detect_order_block(candles_4h, bias)
    if ob is not None:
        if (bias == "bullish" and ob < state.price) or (bias == "bearish" and ob > state.price):
            candidates.append((ob, Risk.CONFLUENCE_OB))
    fvg = detect_fvg(candles_4h, bias)
    if fvg is not None:
        if (bias == "bullish" and fvg < state.price) or (bias == "bearish" and fvg > state.price):
            candidates.append((fvg, Risk.CONFLUENCE_FVG))
    if bias == "bullish":
        sr = state.support_4h
        if sr < state.price and sr is not None and sr != state.price:
            candidates.append((sr, 0.0))
    else:
        sr = state.resistance_4h
        if sr > state.price and sr is not None and sr != state.price:
            candidates.append((sr, 0.0))
    
    # === Fallback: если ничего не найдено, используем ближайший уровень ===
    if not candidates:
        if bias == "bullish":
            fallback = state.support_4h if state.support_4h != state.price else state.support_1d
        else:
            fallback = state.resistance_4h if state.resistance_4h != state.price else state.resistance_1d
        if fallback is not None and fallback != state.price:
            return fallback, 0.2  # низкая уверенность, но хоть что-то
        return None, 0.0

    clusters = {}
    for price, score in candidates:
        found = False
        for cluster_price in list(clusters.keys()):
            if abs(price - cluster_price) / cluster_price < 0.005:
                clusters[cluster_price] += score
                found = True
                break
        if not found:
            clusters[price] = score
    best_price = max(clusters.items(), key=lambda x: x[1])[0]
    best_score = min(clusters[best_price], 1.0)
    return best_price, best_score

def analyze_1d_bias(state: MarketState) -> Tuple[str, float]:
    score = 0.0
    if state.trend_1d == "bullish":
        score += 0.5
    elif state.trend_1d == "bearish":
        score -= 0.5
    if state.structure_1d in ("HL", "BOS_UP", "CHOCH_UP"):
        score += 0.3
    elif state.structure_1d in ("LH", "BOS_DOWN", "CHOCH_DOWN"):
        score -= 0.3
    if state.adx_1d > Risk.ADX_TREND:
        score *= 1.2
    # Снижен порог с 0.2 до 0.1
    if score > 0.1:
        return "bullish", score
    if score < -0.1:
        return "bearish", abs(score)
    return "neutral", 0.0

def check_mss_1h(state: MarketState, bias: str) -> Tuple[bool, float]:
    if bias == "bullish":
        if state.structure_1h in ("CHOCH_UP", "BOS_UP", "HL"):
            return True, 1.0
        elif state.structure_1h == "range":
            return True, 0.5   # разрешаем с пониженной уверенностью
    elif bias == "bearish":
        if state.structure_1h in ("CHOCH_DOWN", "BOS_DOWN", "LH"):
            return True, 1.0
        elif state.structure_1h == "range":
            return True, 0.5
    return False, 0.0

def compute_score(state: MarketState, bias: str) -> float:
    score = 0.0
    if bias == "bullish" and state.rsi_1h < 60:
        score += 0.1
    elif bias == "bearish" and state.rsi_1h > 40:
        score += 0.1
    if state.relative_volume_1h > Risk.RVOL_CONFIRM:
        score += 0.1
    if bias == "bullish" and state.btc_dominance < 45:
        score += 0.05
    elif bias == "bearish" and state.btc_dominance > 55:
        score += 0.05
    if bias == "bullish" and state.funding < -0.00005:
        score += 0.05
    elif bias == "bearish" and state.funding > 0.0001:
        score += 0.05
    return min(score, 0.5)

def compute_entry_confidence(displacement_strength: float, retest_quality: float, distance_to_poi: float, entry_type: str) -> float:
    conf = 0.6
    if displacement_strength > 1.8:
        conf += 0.1
    elif displacement_strength > 1.5:
        conf += 0.05
    if retest_quality > 0.8:
        conf += 0.1
    elif retest_quality > 0.5:
        conf += 0.05
    if distance_to_poi < 0.005:
        conf += 0.1
    if entry_type == "limit":
        conf += 0.05
    return min(conf, 1.0)

def aggregate_confidence(bias_score: float, poi_conf: float, mss_conf: float, entry_conf: float, volume_conf: float) -> float:
    total = (bias_score * Risk.WEIGHT_1D + poi_conf * Risk.WEIGHT_4H + mss_conf * Risk.WEIGHT_1H + entry_conf * Risk.WEIGHT_15M + volume_conf * Risk.WEIGHT_VOLUME)
    return min(total, Risk.MAX_CONFIDENCE)

def build_entry_execution(state: MarketState, poi_price: float, regime: str) -> Tuple[float, str]:
    if regime == "TREND":
        return state.price, "market"
    else:
        return poi_price, "limit"

def swing_signal_v10_6(state: MarketState, candles_4h: List[Tuple[float, float, float, float, float]], candles_15m: List[Tuple[float, float, float, float, float]]) -> Optional[TradeSetup]:
    bias, bias_score = analyze_1d_bias(state)
    if bias == "neutral":
        return None

    pd_4h = get_pd_zone(state.price, state.support_4h, state.resistance_4h)
    regime = detect_regime(state.adx_4h, state.adx_1d)

    # Разрешаем нейтральную зону, но с штрафом
    if bias == "bullish" and pd_4h == "premium":
        return None
    if bias == "bearish" and pd_4h == "discount":
        return None

    rsi_penalty = 0.0
    if bias == "bullish" and state.rsi_1h > Risk.RSI_EXTREME_HIGH:
        rsi_penalty = 0.10
        logger.info(f"{state.symbol}: RSI(1h)={state.rsi_1h:.1f} > {Risk.RSI_EXTREME_HIGH} при bullish bias — штраф -0.10")
    elif bias == "bearish" and state.rsi_1h < Risk.RSI_EXTREME_LOW:
        rsi_penalty = 0.10
        logger.info(f"{state.symbol}: RSI(1h)={state.rsi_1h:.1f} < {Risk.RSI_EXTREME_LOW} при bearish bias — штраф -0.10")

    atr = adaptive_atr(state.atr_1h, state.atr_4h, state.atr_1d, state.price)
    if atr == 0 or (isinstance(atr, float) and math.isnan(atr)):
        atr = state.price * 0.01

    poi, poi_conf = compute_poi_confluence(state, bias, candles_4h)
    if poi is None or (isinstance(poi, float) and math.isnan(poi)):
        return None

    mss_ok, mss_conf = check_mss_1h(state, bias)
    # Разрешаем даже если структура range, но с низкой уверенностью
    if not mss_ok:
        # Если нет структуры, но есть другие подтверждения, можно продолжить с пониженной уверенностью
        if state.relative_volume_1h > 1.5 and abs(state.price - poi) / state.price < 0.005:
            mss_conf = 0.3
        else:
            return None

    # === Обработка свипа (необязательный, но даёт бонус) ===
    sweep_result = detect_sweep_v10_6(state, candles_15m, bias, atr)
    sweep_detected = sweep_result is not None
    sweep_idx = 0
    if sweep_detected:
        sweep_dir, sweep_idx = sweep_result
        if sweep_dir != bias:
            return None
        # Если свип есть, проверяем импульс
        if not detect_displacement_multi_directional(candles_15m, sweep_idx + 1, atr, bias):
            # Импульса нет, но если цена у POI, всё равно входим со штрафом
            if abs(state.price - poi) / state.price > 0.005:
                return None
            else:
                # Штраф за отсутствие импульса
                poi_conf *= 0.8
    else:
        # Свипа нет – разрешаем вход, только если цена близка к POI и есть объём
        if abs(state.price - poi) / state.price > 0.005:
            return None
        if state.relative_volume_1h < 0.8:
            return None
        # Штраф за отсутствие свипа
        poi_conf *= 0.6

    # Определяем тип входа
    entry_price, entry_type = build_entry_execution(state, poi, regime)

    # Расчёт уверенности
    total_range = 0.0
    if sweep_detected and sweep_idx + 4 <= len(candles_15m):
        for i in range(sweep_idx + 1, min(sweep_idx + 4, len(candles_15m))):
            total_range += (candles_15m[i][0] - candles_15m[i][1])
        displacement_strength = total_range / 3.0 / atr if atr > 0 else 1.0
    else:
        displacement_strength = 0.5  # усреднённое значение

    retest_quality = 0.7
    distance_to_poi = abs(state.price - poi) / state.price
    entry_conf = compute_entry_confidence(displacement_strength, retest_quality, distance_to_poi, entry_type)

    volume_conf = compute_score(state, bias)

    final_conf = aggregate_confidence(bias_score, poi_conf, mss_conf, entry_conf, volume_conf)
    final_conf = max(final_conf - rsi_penalty, 0.0)

    # Штраф за нейтральную pd_zone
    if pd_4h == "neutral":
        final_conf *= 0.9

    if final_conf < Risk.MIN_CONFIDENCE_SWING:
        logger.info(f"{state.symbol}: итоговая уверенность {final_conf:.2f} < порога {Risk.MIN_CONFIDENCE_SWING} — сигнал отклонён")
        return None

    # === Расчёт уровней ===
    if bias == "bullish":
        stop = poi - atr * Risk.STRUCTURE_BUFFER_ATR
        stop = enforce_min_stop(entry_price, stop)
        min_dist = max(state.price * Risk.MIN_LIQ_DISTANCE_PERCENT, state.atr_4h * Risk.MIN_LIQ_DISTANCE_ATR_MULT)
        above_liq, _ = liquidity_targets(entry_price, state.support_4h, state.support_1d, state.resistance_4h, state.resistance_1d, liquidity=state.liquidity_4h, min_dist=min_dist)
        tp1, tp2 = target_engine("LONG", entry_price, stop, above_liq, None)
        if not tp1:
            tp1 = entry_price + abs(entry_price - stop) * 2
            tp2 = tp1 + abs(entry_price - stop) * 1.5
        rr = calculate_rr(entry_price, stop, tp1)
        min_rr = dynamic_rr(state)
        if rr < min_rr:
            return None
        return TradeSetup("LONG", round_to_tick(state.symbol, entry_price), round_stop(state.symbol, stop, "LONG"), round_to_tick(state.symbol, tp1), round_to_tick(state.symbol, tp2), final_conf, "V14.0 трендовый сигнал (bullish)", "SWING", pd_4h, sweep_detected, entry_type, is_counter_trend=False)
    else:
        stop = poi + atr * Risk.STRUCTURE_BUFFER_ATR
        stop = enforce_min_stop(entry_price, stop)
        min_dist = max(state.price * Risk.MIN_LIQ_DISTANCE_PERCENT, state.atr_4h * Risk.MIN_LIQ_DISTANCE_ATR_MULT)
        _, below_liq = liquidity_targets(entry_price, state.support_4h, state.support_1d, state.resistance_4h, state.resistance_1d, liquidity=state.liquidity_4h, min_dist=min_dist)
        tp1, tp2 = target_engine("SHORT", entry_price, stop, None, below_liq)
        if not tp1:
            tp1 = entry_price - abs(entry_price - stop) * 2
            tp2 = tp1 - abs(entry_price - stop) * 1.5
        rr = calculate_rr(entry_price, stop, tp1)
        min_rr = dynamic_rr(state)
        if rr < min_rr:
            return None
        return TradeSetup("SHORT", round_to_tick(state.symbol, entry_price), round_stop(state.symbol, stop, "SHORT"), round_to_tick(state.symbol, tp1), round_to_tick(state.symbol, tp2), final_conf, "V14.0 трендовый сигнал (bearish)", "SWING", pd_4h, sweep_detected, entry_type, is_counter_trend=False)

def detect_range_market(state: MarketState) -> bool:
    if state.adx_1d >= Risk.ADX_TREND or state.adx_4h >= Risk.ADX_TREND:
        return False
    support = state.support_4h if state.support_4h and state.support_4h != state.price else state.support_1d
    resistance = state.resistance_4h if state.resistance_4h and state.resistance_4h != state.price else state.resistance_1d
    if support is None or resistance is None or support >= resistance:
        return False
    return True

# ==========================================================
# НОВАЯ ЛОГИКА КОНТРТРЕНДА (V14.0)
# ==========================================================
def generate_counter_trend_signal(state: MarketState, candles_15m: List[Tuple[float, float, float, float, float]]) -> Optional[TradeSetup]:
    # Убрали жесткий фильтр range_market для контртренда
    # Теперь он работает и в тренде, но с меньшей уверенностью

    atr = state.atr_4h if state.atr_4h > 0 else state.atr_1h
    if atr <= 0:
        atr = state.price * 0.01
        logger.warning(f"Контртренд: ATR=0 для {state.symbol}, используем 1% = {atr:.6f}")

    def get_tick(price: float) -> float:
        if price < 0.1:
            return 0.0001
        elif price < 1:
            return 0.001
        else:
            return 0.01

    def round_price(price: float) -> float:
        tick = get_tick(price)
        return round(price / tick) * tick

    def round_stop_price(stop: float, direction: str) -> float:
        tick = get_tick(stop)
        if direction == "LONG":
            return math.floor(stop / tick) * tick
        else:
            return math.ceil(stop / tick) * tick

    support = state.support_4h if state.support_4h and state.support_4h != state.price else state.support_1d
    resistance = state.resistance_4h if state.resistance_4h and state.resistance_4h != state.price else state.resistance_1d

    if support is None or resistance is None:
        logger.info(f"Контртренд: нет уровней для {state.symbol}")
        return None

    dist_to_support = abs(state.price - support) / state.price
    dist_to_resistance = abs(state.price - resistance) / state.price
    
    # Смягчаем условия: 3% до уровня или экстремальный RSI
    near_support = dist_to_support <= Risk.COUNTER_TREND_ENTRY_DIST_PERCENT
    near_resistance = dist_to_resistance <= Risk.COUNTER_TREND_ENTRY_DIST_PERCENT

    rsi_ok_long = state.rsi_1h < Risk.COUNTER_TREND_RSI_LONG
    rsi_ok_short = state.rsi_1h > Risk.COUNTER_TREND_RSI_SHORT
    
    # Добавляем возможность входа при экстремальном RSI без ожидания у уровня
    entry_type = "limit"
    if Risk.COUNTER_TREND_ALLOW_MARKET_ENTRY:
        if rsi_ok_short and not near_resistance:
            # Если RSI очень высокий, мы позволяем шорт от текущей цены
            near_resistance = True
            entry = state.price
            entry_type = "market"
        elif rsi_ok_long and not near_support:
            # Аналогично для лонга
            near_support = True
            entry = state.price
            entry_type = "market"

    volume_ok = state.relative_volume_1h > Risk.COUNTER_TREND_MIN_VOLUME
    ema_bullish = state.ema50_1h > state.ema200_1h
    ema_bearish = state.ema50_1h < state.ema200_1h

    # Бонус к уверенности, если рынок во флэте (range_market)
    range_bonus = 0.15 if state.range_market else 0.0

    confidence = 0.35

    if near_support and rsi_ok_long and volume_ok:
        # Определяем точку входа: если мы разрешили маркет, берем state.price, иначе support
        if entry_type == "market":
            entry = state.price
        else:
            entry = support
        # Расширенный стоп: 2.5 * ATR, но не меньше 1.5% от цены
        stop = entry - atr * Risk.COUNTER_TREND_STOP_ATR_MULT
        stop = enforce_min_stop(entry, stop)
        stop = round_stop_price(stop, "LONG")
        
        # Ищем цель вниз по ликвидности
        above_liq, _ = liquidity_targets(entry, state.support_4h, state.support_1d, state.resistance_4h, state.resistance_1d, liquidity=state.liquidity_4h)
        if above_liq and above_liq > entry:
            tp1 = above_liq
        else:
            # Если ликвидности нет, берем 4% от цены (минимум для интрадея)
            tp1 = entry + entry * Risk.COUNTER_TREND_MIN_TP_PERCENT
        
        # Проверяем минимальный TP в процентах (4%)
        tp_percent = (tp1 - entry) / entry
        if tp_percent < Risk.COUNTER_TREND_MIN_TP_PERCENT:
            # Форсируем достижение минимума
            tp1 = entry + entry * Risk.COUNTER_TREND_MIN_TP_PERCENT
            tp_percent = Risk.COUNTER_TREND_MIN_TP_PERCENT
            
        tp2 = tp1 + (tp1 - entry) * 0.5
        entry = round_price(entry)
        tp1 = round_price(tp1)
        tp2 = round_price(tp2)
        
        # Начисляем уверенность
        confidence = 0.35
        if rsi_ok_long: confidence += 0.15
        if volume_ok: confidence += 0.05
        if above_liq: confidence += 0.15
        if ema_bearish: confidence += 0.20
        if state.adx_1d < 25: confidence += 0.10
        if state.adx_1d > 25: confidence -= 0.10
        confidence += range_bonus  # Добавляем бонус за флэт
        confidence = min(max(confidence, Risk.MIN_CONFIDENCE_COUNTER), 0.60)
        
        logger.info(f"Контртренд LONG сгенерирован для {state.symbol}, entry={entry:.6f}, stop={stop:.6f}, tp1={tp1:.6f}, уверенность {confidence:.2f}")
        return TradeSetup("LONG", entry, stop, tp1, tp2, confidence,
                          "Контртренд (диапазон): лонг от поддержки/перепроданности", "INTRADAY", pd_zone=get_pd_zone(entry, support, resistance),
                          sweep_detected=False, entry_type=entry_type, is_counter_trend=True)

    if near_resistance and rsi_ok_short and volume_ok:
        # Определяем точку входа: если мы разрешили маркет, берем state.price, иначе resistance
        if entry_type == "market":
            entry = state.price
        else:
            entry = resistance
        # Расширенный стоп: 2.5 * ATR, но не меньше 1.5% от цены
        stop = entry + atr * Risk.COUNTER_TREND_STOP_ATR_MULT
        stop = enforce_min_stop(entry, stop)
        stop = round_stop_price(stop, "SHORT")
        
        _, below_liq = liquidity_targets(entry, state.support_4h, state.support_1d, state.resistance_4h, state.resistance_1d, liquidity=state.liquidity_4h)
        if below_liq and below_liq < entry:
            tp1 = below_liq
        else:
            tp1 = entry - entry * Risk.COUNTER_TREND_MIN_TP_PERCENT
        
        tp_percent = (entry - tp1) / entry
        if tp_percent < Risk.COUNTER_TREND_MIN_TP_PERCENT:
            tp1 = entry - entry * Risk.COUNTER_TREND_MIN_TP_PERCENT
            tp_percent = Risk.COUNTER_TREND_MIN_TP_PERCENT
            
        tp2 = tp1 - (entry - tp1) * 0.5
        entry = round_price(entry)
        tp1 = round_price(tp1)
        tp2 = round_price(tp2)
        
        confidence = 0.35
        if rsi_ok_short: confidence += 0.15
        if volume_ok: confidence += 0.05
        if below_liq: confidence += 0.15
        if ema_bullish: confidence += 0.20
        if state.adx_1d < 25: confidence += 0.10
        if state.adx_1d > 25: confidence -= 0.10
        confidence += range_bonus  # Добавляем бонус за флэт
        confidence = min(max(confidence, Risk.MIN_CONFIDENCE_COUNTER), 0.60)
        
        logger.info(f"Контртренд SHORT сгенерирован для {state.symbol}, entry={entry:.6f}, stop={stop:.6f}, tp1={tp1:.6f}, уверенность {confidence:.2f}")
        return TradeSetup("SHORT", entry, stop, tp1, tp2, confidence,
                          "Контртренд (диапазон): шорт от сопротивления/перекупленности", "INTRADAY", pd_zone=get_pd_zone(entry, support, resistance),
                          sweep_detected=False, entry_type=entry_type, is_counter_trend=True)

    logger.info(f"Контртренд: условия не выполнены для {state.symbol} (near_support={near_support}, rsi_long={rsi_ok_long}, volume_ok={volume_ok}, near_resistance={near_resistance}, rsi_short={rsi_ok_short})")
    return None

def is_signal_valid(signal: Signal, state: MarketState) -> Tuple[bool, str]:
    now = time.time()
    age = now - signal.created_at
    symbol = signal.symbol
    ttl = Risk.TTL.get(symbol, Risk.TTL["DEFAULT"])
    if signal.is_counter_trend:
        ttl = Risk.COUNTER_TREND_TTL_HOURS * 3600
    if age > ttl:
        return False, f"EXPIRED: TTL {ttl/3600:.0f}h exceeded"
    if signal.direction == "LONG":
        if state.close_4h < signal.stop:
            return False, f"INVALID: stop breached"
    else:
        if state.close_4h > signal.stop:
            return False, f"INVALID: stop breached"
    if not signal.is_counter_trend:
        if signal.direction == "LONG" and (state.trend_1d == "bearish" or state.trend_4h == "bearish"):
            return False, f"INVALID: trend turned bearish"
        if signal.direction == "SHORT" and (state.trend_1d == "bullish" or state.trend_4h == "bullish"):
            return False, f"INVALID: trend turned bullish"
    buffer = max(signal.entry * 0.002, state.atr_4h * 0.2)
    if signal.direction == "LONG" and signal.entry_support is not None:
        if state.support_4h < signal.entry_support - buffer:
            return False, f"INVALID: structure broken"
    if signal.direction == "SHORT" and signal.entry_resistance is not None:
        if state.resistance_4h > signal.entry_resistance + buffer:
            return False, f"INVALID: structure broken"
    if signal.direction == "LONG" and state.price >= signal.tp2:
        return False, f"COMPLETED: TP2 reached"
    if signal.direction == "SHORT" and state.price <= signal.tp2:
        return False, f"COMPLETED: TP2 reached"
    return True, "valid"

def is_similar_signal(s1: Signal, s2: Signal) -> bool:
    if s1.direction != s2.direction:
        return False
    if abs(s1.entry - s2.entry) / s1.entry > Risk.MATCH_TOL_ENTRY:
        return False
    if abs(s1.stop - s2.stop) / s1.stop > Risk.MATCH_TOL_STOP:
        return False
    if abs(s1.tp1 - s2.tp1) / s1.tp1 > Risk.MATCH_TOL_TP1:
        return False
    return True

def should_skip_recreation(symbol: str, new_setup: TradeSetup) -> bool:
    deleted = storage.get_last_deleted(symbol)
    if not deleted:
        return False
    old_signal, del_time = deleted
    if time.time() - del_time > Risk.ANTI_RECREATION_WINDOW:
        return False
    old = Signal(symbol=symbol, direction=old_signal.direction, entry=old_signal.entry, stop=old_signal.stop, tp1=old_signal.tp1, tp2=old_signal.tp2, confidence=old_signal.confidence, reason=old_signal.reason, timeframe=old_signal.timeframe, pd_zone=old_signal.pd_zone, sweep_detected=old_signal.sweep_detected, created_at=old_signal.created_at, updated_at=old_signal.updated_at, status=old_signal.status)
    new = Signal(symbol=symbol, direction=new_setup.direction, entry=new_setup.entry, stop=new_setup.stop, tp1=new_setup.tp1, tp2=new_setup.tp2, confidence=new_setup.confidence, reason=new_setup.reason, timeframe=new_setup.timeframe, pd_zone=new_setup.pd_zone, sweep_detected=new_setup.sweep_detected, created_at=0, updated_at=0)
    return is_similar_signal(old, new)

def generate_context(state: MarketState, swing: TradeSetup, regime: str, gtrend: str, is_active: bool = False, age_hours: float = 0.0) -> str:
    price_format = ".4f" if state.price < 1 else ".0f"
    def safe_format(value, format_spec):
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return "0"
        try:
            return f"{value:{format_spec}}"
        except (TypeError, ValueError):
            return "0"
    lines = ["▫️ **Комментарий**", "• Мультитаймфреймовый анализ"]
    pd_4h = get_pd_zone(state.price, state.support_4h, state.resistance_4h)
    lines.append(f"• Уровни 4H: поддержка {safe_format(state.support_4h, price_format)}, сопротивление {safe_format(state.resistance_4h, price_format)}.")
    if state.liquidity_4h:
        buy = state.liquidity_4h.get("buy_side", [])
        sell = state.liquidity_4h.get("sell_side", [])
        min_dist = max(state.price * Risk.MIN_LIQ_DISTANCE_PERCENT, state.atr_4h * Risk.MIN_LIQ_DISTANCE_ATR_MULT)
        liq_parts = []
        if buy:
            valid_buy = [lvl for lvl in buy if lvl > state.price and (lvl - state.price) >= min_dist]
            if valid_buy:
                liq_parts.append(f"выше {safe_format(valid_buy[0], price_format)}")
        if sell:
            valid_sell = [lvl for lvl in sell if lvl < state.price and (state.price - lvl) >= min_dist]
            if valid_sell:
                liq_parts.append(f"ниже {safe_format(valid_sell[0], price_format)}")
        if liq_parts:
            lines.append(f"• Скопление ликвидности: {' и '.join(liq_parts)}.")
    rsi4 = state.rsi_4h
    rsi1 = state.rsi_1h
    rsi4_desc = "высокий (>60)" if rsi4 > 60 else "низкий (<40)" if rsi4 < 40 else "нейтральный"
    rsi1_desc = "высокий (>60)" if rsi1 > 60 else "низкий (<40)" if rsi1 < 40 else "нейтральный"
    lines.append(f"• RSI: на 4H {rsi4:.0f} ({rsi4_desc}), на 1H {rsi1:.0f} ({rsi1_desc}).")
    atr_percent = (state.atr_4h / state.price) * 100
    lines.append(f"• Волатильность (ATR) на 4H: {state.atr_4h:.0f} ({atr_percent:.1f}% от цены).")
    if is_active and swing.direction != "NO_TRADE":
        lines.append(f"✅ Активный сигнал (возраст: {age_hours:.1f} ч).")
    if state.btc_dominance > 0:
        lines.append(f"• Доминация BTC: {state.btc_dominance:.1f}%.")
    if state.alt_season:
        lines.append("• На рынке наблюдается альтсезон (повышенная волатильность альтов).")    
    oi_usd = state.oi * state.price
    if oi_usd >= 1_000_000_000:
        lines.append(f"• Открытый интерес (OI): ${oi_usd/1_000_000_000:.1f}B.")
    elif oi_usd >= 1_000_000:
        lines.append(f"• Открытый интерес (OI): ${oi_usd/1_000_000:.1f}M.")
    elif oi_usd >= 1_000:
        lines.append(f"• Открытый интерес (OI): ${oi_usd/1_000:.1f}K.")
    else:
        lines.append(f"• Открытый интерес (OI): ${oi_usd:.0f}.")
    if abs(state.funding) > 0.0001:
        funding_pct = state.funding * 100
        lines.append(f"• Фандинг: {funding_pct:.2f}%.")
    if swing.direction != "NO_TRADE" and not swing.is_counter_trend:
        dist = abs(state.price - swing.entry) / swing.entry if swing.entry else 1.0
        if dist < 0.01:
            lines.append("💡 Цена находится в зоне ожидания входа.")
        if swing.direction == "LONG":
            if pd_4h == "discount":
                lines.append("💡 Оптимально входить на откате к поддержке.")
            else:
                lines.append("💡 Лучше дождаться подтверждения у уровня поддержки.")
        else:
            if pd_4h == "premium":
                lines.append("💡 Оптимально входить после отката от сопротивления.")
            else:
                lines.append("💡 Лучше дождаться подхода к сопротивлению.")
        lines.append(f"• Тип входа: {swing.entry_type}")
        lines.append("• Точка входа уточняется на 15M таймфрейме.")
        stop_format = ".4f" if state.price < 1 else ".0f"
        lines.append(f"▪️ Сценарий теряет силу при закреплении цены {'выше' if swing.direction=='SHORT' else 'ниже'} {safe_format(swing.stop, stop_format)}.")
    elif swing.direction != "NO_TRADE" and swing.is_counter_trend:
        lines.append("⚠️ Контртрендовая идея (риск выше среднего).")
        lines.append(f"• Точка входа: {safe_format(swing.entry, price_format)}")
        lines.append(f"• Стоп-лосс: {safe_format(swing.stop, price_format)}")
        lines.append(f"• Тейк-профит 1: {safe_format(swing.tp1, price_format)}")
        lines.append(f"• Тейк-профит 2: {safe_format(swing.tp2, price_format)}")
        lines.append(f"• Уверенность: {swing.confidence*100:.0f}%")
    else:
        if pd_4h == "premium":
            lines.append("• Основной (свинг) сигнал: цена в premium-зоне, но факторов для трендового входа недостаточно.")
        elif pd_4h == "discount":
            lines.append("• Основной (свинг) сигнал: цена в discount-зоне, но факторов для трендового входа недостаточно.")
        else:
            lines.append("• Основной (свинг) сигнал: цена в нейтральной зоне — возможность появится при подходе к уровням поддержки/сопротивления.")
    return "\n".join(lines)

def _build_hold_note(state: MarketState, setup: TradeSetup) -> str:
    if setup.direction not in ("LONG", "SHORT"):
        return ""
    ema_val = state.ema50_4h if state.ema50_4h else state.ema50_1h
    if not ema_val:
        return f"Удерживать остаток позиции ({setup.runner_percent}%) по трейлинг-стопу до появления признаков разворота структуры."
    price_format = ".4f" if state.price < 1 else ".2f"
    if setup.direction == "LONG":
        return f"Удерживать остаток позиции ({setup.runner_percent}%), пока цена закрытия 4H выше EMA50 ({ema_val:{price_format}}). Закрытие ниже этого уровня — сигнал к выходу из остатка."
    else:
        return f"Удерживать остаток позиции ({setup.runner_percent}%), пока цена закрытия 4H ниже EMA50 ({ema_val:{price_format}}). Закрытие выше этого уровня — сигнал к выходу из остатка."

def generate_signals(state: MarketState, candles_4h: List[Tuple[float, float, float, float, float]] = None, candles_15m: List[Tuple[float, float, float, float, float]] = None) -> MultiSignal:
    now = time.time()
    symbol = state.symbol
    logger.info(f"GENERATE_SIGNALS: symbol={symbol}, price={state.price:.2f} (V14.0)")

    state.range_market = detect_range_market(state)

    existing = storage.get(symbol)
    if existing and not existing.is_counter_trend:
        if existing.status == "ACTIVE":
            if (existing.direction == "LONG" and state.price >= existing.tp1) or (existing.direction == "SHORT" and state.price <= existing.tp1):
                existing.status = "PARTIAL"
                existing.tp1_reached_at = now
                storage.set(existing)
                logger.info(f"Signal {symbol} reached TP1, status -> PARTIAL")
        is_valid, reason = is_signal_valid(existing, state)
        if is_valid:
            setup = TradeSetup(direction=existing.direction, entry=existing.entry, stop=existing.stop, tp1=existing.tp1, tp2=existing.tp2, confidence=existing.confidence, reason=existing.reason, timeframe=existing.timeframe, pd_zone=existing.pd_zone, sweep_detected=existing.sweep_detected, entry_type=existing.entry_type, is_counter_trend=existing.is_counter_trend)
            age_hours = (now - existing.created_at) / 3600.0
            context = generate_context(state, setup, "", "", is_active=True, age_hours=age_hours)
            regime = detect_regime(state.adx_4h, state.adx_1d)
            gtrend = global_trend(state.trend_1d, state.trend_4h, state.trend_1h)
            counter = generate_counter_trend_signal(state, candles_15m or [])
            setup.hold_note = _build_hold_note(state, setup)
            if counter:
                counter.hold_note = _build_hold_note(state, counter)
            return MultiSignal(swing=setup, counter_trend=counter, regime=regime, global_trend=gtrend, context=context, is_active_signal=True, signal_age_hours=age_hours)
        else:
            logger.info(f"Existing signal NOT valid: {reason}. Removing.")
            storage.delete(symbol)

    if candles_4h is None:
        candles_4h = []
    if candles_15m is None:
        candles_15m = []

    new_setup = swing_signal_v10_6(state, candles_4h, candles_15m)
    regime = detect_regime(state.adx_4h, state.adx_1d)
    gtrend = global_trend(state.trend_1d, state.trend_4h, state.trend_1h)

    counter = generate_counter_trend_signal(state, candles_15m)
    if counter:
        counter.hold_note = _build_hold_note(state, counter)

    if new_setup and new_setup.direction != "NO_TRADE":
        if should_skip_recreation(symbol, new_setup):
            logger.info(f"Skipping recreation of identical signal for {symbol}")
            empty_setup = TradeSetup("NO_TRADE", None, None, None, None, 0, "No setup (recent)", "SWING", is_counter_trend=False)
            context = generate_context(state, empty_setup, regime, gtrend, is_active=False, age_hours=0.0)
            return MultiSignal(swing=empty_setup, counter_trend=counter, regime=regime, global_trend=gtrend, context=context, is_active_signal=False, signal_age_hours=0.0)
        atr = adaptive_atr(state.atr_1h, state.atr_4h, state.atr_1d, state.price)
        if atr == 0 or math.isnan(atr):
            atr = state.price * 0.01
        atr_pct = atr / state.price
        base_distance = max(Risk.DISTANCE_MIN, min(Risk.DISTANCE_MAX, atr_pct * Risk.DISTANCE_ATR_MULT))
        if new_setup.direction == "LONG" and new_setup.tp1:
            tp_dist = (new_setup.tp1 - new_setup.entry) / new_setup.entry
            distance_limit = min(base_distance, tp_dist * Risk.TP1_SAFETY_FACTOR)
        elif new_setup.direction == "SHORT" and new_setup.tp1:
            tp_dist = (new_setup.entry - new_setup.tp1) / new_setup.entry
            distance_limit = min(base_distance, tp_dist * Risk.TP1_SAFETY_FACTOR)
        else:
            distance_limit = base_distance
        signal = Signal(symbol=symbol, direction=new_setup.direction, entry=new_setup.entry, stop=new_setup.stop, tp1=new_setup.tp1, tp2=new_setup.tp2, confidence=new_setup.confidence, reason=new_setup.reason, timeframe=new_setup.timeframe, pd_zone=new_setup.pd_zone, sweep_detected=new_setup.sweep_detected, created_at=now, updated_at=now, entry_support=state.support_4h if new_setup.direction == "LONG" else None, entry_resistance=state.resistance_4h if new_setup.direction == "SHORT" else None, distance_limit=distance_limit, entry_type=new_setup.entry_type, is_counter_trend=False)
        storage.set(signal)
        context = generate_context(state, new_setup, regime, gtrend, is_active=False, age_hours=0.0)
        new_setup.hold_note = _build_hold_note(state, new_setup)
        logger.info(f"New signal generated: {new_setup.direction} at {new_setup.entry:.2f}")
        return MultiSignal(swing=new_setup, counter_trend=counter, regime=regime, global_trend=gtrend, context=context, is_active_signal=False, signal_age_hours=0.0)
    else:
        empty_setup = TradeSetup("NO_TRADE", None, None, None, None, 0, "No setup", "SWING", is_counter_trend=False)
        context = generate_context(state, empty_setup, regime, gtrend, is_active=False, age_hours=0.0)
        return MultiSignal(swing=empty_setup, counter_trend=counter, regime=regime, global_trend=gtrend, context=context, is_active_signal=False, signal_age_hours=0.0)