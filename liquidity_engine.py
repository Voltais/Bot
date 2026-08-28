"""
Liquidity Heatmap Engine
Определяет реальные кластеры ликвидности (стоп-кластеры) на основе:
equal highs / lows
локальных экстремумов
круглых уровней
объёмных зон
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


def _cluster_levels(levels: List[float], tolerance: float) -> List[float]:
    """
    Простая кластеризация: группирует уровни, отстоящие менее чем на tolerance.
    Возвращает средние значения кластеров.
    """
    if len(levels) < 2:
        return levels    
    sorted_levels = sorted(levels)
    clusters = []
    current_cluster = [sorted_levels[0]]    
    for level in sorted_levels[1:]:
        if abs(level - np.mean(current_cluster)) <= tolerance:
            current_cluster.append(level)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [level]    
    clusters.append(np.mean(current_cluster))
    return clusters

def _find_local_maxima(highs: np.ndarray) -> List[float]:
    """Находит локальные максимумы (больше соседей)."""
    maxima = []
    for i in range(1, len(highs) - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            maxima.append(highs[i])
    return maxima

def _find_local_minima(lows: np.ndarray) -> List[float]:
    """Находит локальные минимумы (меньше соседей)."""
    minima = []
    for i in range(1, len(lows) - 1):
        if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            minima.append(lows[i])
    return minima

def detect_liquidity_heatmap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    price: float,
    round_level_tolerance: float = 0.001,
    volume_threshold: float = 1.5,
    include_volume: bool = True
) -> Dict[str, List[float]]:
    """
    Возвращает словарь с уровнями ликвидности:
    {
        "buy_side": [цена1, цена2, ...],
        "sell_side": [цена1, цена2, ...],
        "round_levels": [цена1, цена2, ...]
    }
    """
    local_max = _find_local_maxima(highs)
    local_min = _find_local_minima(lows)
    all_levels = local_max + local_min
    
    def round_to_nearest(x, base):
        return round(x / base) * base
    
    round_levels = []
    for base in [1000, 500, 100, 50, 10]:
        r = round_to_nearest(price, base)
        if r not in round_levels:
            round_levels.append(r)
    
    if include_volume and len(volumes) > 0:
        avg_vol = np.mean(volumes)
        high_vol_idx = np.where(volumes > volume_threshold * avg_vol)[0]
        for idx in high_vol_idx:
            level = closes[idx]
            if level not in all_levels:
                all_levels.append(level)
    
    # ВАЖНО: круглые уровни (round_levels) НЕ подмешиваются в all_levels.
    # Раньше они попадали в buy_side/sell_side наравне с реальными уровнями
    # (экстремумы, объёмные зоны), и если ближайший реальный уровень не проходил
    # по риск/прибыли, механизм выбора TP в signal_engine.py "проскакивал" до
    # первого круглого числа (например, 1000 для цены ~700) просто потому, что
    # оно физически далеко и формально даёт большой RR — хотя это не реальный
    # уровень, а артефакт округления. round_levels остаются доступны отдельным
    # полем ниже, для тех, кому нужны именно круглые уровни как таковые.
    
    tolerance = price * round_level_tolerance
    clustered = _cluster_levels(all_levels, tolerance)
    
    buy_side = sorted([lvl for lvl in clustered if lvl > price])
    sell_side = sorted([lvl for lvl in clustered if lvl < price], reverse=True)
    round_levels_sorted = sorted(round_levels)
    
    return {
        "buy_side": buy_side,
        "sell_side": sell_side,
        "round_levels": round_levels_sorted
    }


def compute_fib_levels(highs: np.ndarray, lows: np.ndarray, period: int = 100) -> Dict[str, float]:
    """
    Строит сетку Фибоначчи по фактическому хай/лою за последние `period` баров.
    Чистая математика поверх уже загруженных свечей — новых запросов к бирже не требует.
    """
    if highs is None or lows is None or len(highs) == 0 or len(lows) == 0:
        return {}
    recent_highs = highs[-period:]
    recent_lows = lows[-period:]
    global_max = float(np.max(recent_highs))
    global_min = float(np.min(recent_lows))
    price_range = global_max - global_min
    if price_range <= 0:
        return {}
    return {
        "fib_0": round(global_max, 6),
        "fib_236": round(global_max - price_range * 0.236, 6),
        "fib_382": round(global_max - price_range * 0.382, 6),
        "fib_500": round(global_max - price_range * 0.500, 6),
        "fib_618": round(global_max - price_range * 0.618, 6),
        "fib_786": round(global_max - price_range * 0.786, 6),
        "fib_100": round(global_min, 6),
    }


def detect_sweep(price: float, support: float, resistance: float, atr: float) -> Optional[str]:
    """
    Обнаружение сбора ликвидности (ДОБАВЛЕНО для совместимости).
    """
    buffer = atr * 0.4
    if price > resistance + buffer:
        return "BUY_SWEEP"
    if price < support - buffer:
        return "SELL_SWEEP"
    return None


def _find_swing_points(highs: np.ndarray, lows: np.ndarray, left: int = 2, right: int = 2) -> List[Tuple[int, float, str]]:
    """
    Находит подтверждённые свинг-хай/лоу (фракталы): точка считается свингом,
    только если она строго выше/ниже `left` свечей слева и `right` свечей справа.
    Возвращает список (index, price, 'high'|'low') в хронологическом порядке.
    """
    n = len(highs)
    swings: List[Tuple[int, float, str]] = []
    if n < left + right + 1:
        return swings
    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        if highs[i] == np.max(window_h) and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swings.append((i, float(highs[i]), "high"))
        window_l = lows[i - left:i + right + 1]
        if lows[i] == np.min(window_l) and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swings.append((i, float(lows[i]), "low"))
    swings.sort(key=lambda s: s[0])
    return swings


def detect_market_structure(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, left: int = 2, right: int = 2) -> str:
    """
    Настоящая рыночная структура (SMC-стиль) на основе подтверждённых свинг-хай/лоу,
    а не позиции цены относительно EMA.

    Логика:
    - Восходящая структура = последний свинг-хай выше предыдущего (Higher High)
      И последний свинг-лоу выше предыдущего (Higher Low).
    - Нисходящая структура = Lower High + Lower Low.
    - BOS (Break of Structure) — текущая цена закрылась за последним свинг-хай/лоу
      В СТОРОНУ существующей структуры (продолжение тренда).
    - CHOCH (Change of Character) — текущая цена закрылась за свинг-пойнт
      ПРОТИВ существующей структуры (первый признак разворота).
    - Если чётких HH+HL / LH+LL нет — считаем структуру неопределённой ("range").

    Возвращает одно из: "BOS_UP", "CHOCH_UP", "HL", "BOS_DOWN", "CHOCH_DOWN", "LH", "range".
    """
    if highs is None or lows is None or closes is None or len(closes) == 0:
        return "range"

    swings = _find_swing_points(highs, lows, left, right)
    swing_highs = [s for s in swings if s[2] == "high"]
    swing_lows = [s for s in swings if s[2] == "low"]
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "range"

    last_high, prev_high = swing_highs[-1], swing_highs[-2]
    last_low, prev_low = swing_lows[-1], swing_lows[-2]

    higher_high = last_high[1] > prev_high[1]
    higher_low = last_low[1] > prev_low[1]
    lower_high = last_high[1] < prev_high[1]
    lower_low = last_low[1] < prev_low[1]

    current_close = float(closes[-1])
    uptrend_structure = higher_high and higher_low
    downtrend_structure = lower_high and lower_low

    if uptrend_structure:
        if current_close > last_high[1]:
            return "BOS_UP"
        if current_close < last_low[1]:
            return "CHOCH_DOWN"
        return "HL"

    if downtrend_structure:
        if current_close < last_low[1]:
            return "BOS_DOWN"
        if current_close > last_high[1]:
            return "CHOCH_UP"
        return "LH"

    # Нет чёткой HH+HL или LH+LL — структура смешанная/неопределённая.
    # Всё равно проверяем, не сломала ли цена ближайший значимый свинг.
    if current_close > last_high[1]:
        return "CHOCH_UP" if lower_high else "BOS_UP"
    if current_close < last_low[1]:
        return "CHOCH_DOWN" if higher_low else "BOS_DOWN"

    return "range"