"""
market_data_fallback.py
Резервные (Fallback) источники рыночных данных — Bybit и OKX.
Используются ТОЛЬКО когда основной источник (Binance) недоступен:
техработы, гео-блокировка, сетевой сбой, таймаут.

ДОБАВЛЕНЫ ФУНКЦИИ ДЛЯ ПРОФЕССИОНАЛЬНОГО АНАЛИЗА:
- fetch_oi_delta(): изменение открытого интереса (OI) за 1 час.
- fetch_order_book_imbalance(): дисбаланс стакана Bids/Asks в пределах 1% от цены.
- calculate_cvd_from_agg_trades(): расчет кумулятивной дельты объема (CVD) на основе агрегированных сделок.

Все функции возвращают данные в удобном формате для дальнейшего использования в signal_engine.
"""
import asyncio
import logging
from typing import List, Tuple

import httpx

logger = logging.getLogger(__name__)

# Увеличен таймаут до 30 секунд для надёжности
http_client = httpx.AsyncClient(timeout=30.0)


class _SimpleRateLimiter:
    """Минимальный лимитер запросов, независимый от лимитера main.py (свой пул на Bybit/OKX)."""
    def __init__(self, calls_per_minute: int):
        self.interval = 60.0 / calls_per_minute
        self._semaphore = asyncio.Semaphore(1)
        self._last_call = 0.0

    async def acquire(self):
        async with self._semaphore:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self.interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()


_bybit_rate_limiter = _SimpleRateLimiter(calls_per_minute=10 * 60)
_okx_rate_limiter = _SimpleRateLimiter(calls_per_minute=10 * 60)

# Наши ключи таймфреймов ("1d"/"4h"/"1h"/"15m") -> формат интервалов конкретной биржи.
BYBIT_INTERVAL_MAP = {"1d": "D", "4h": "240", "1h": "60", "15m": "15"}
OKX_INTERVAL_MAP = {"1d": "1D", "4h": "4H", "1h": "1H", "15m": "15m"}


# ============================================================
# СУЩЕСТВУЮЩИЕ FALLBACK-ФУНКЦИИ (без изменений)
# ============================================================

async def fetch_bybit_klines(symbol: str, interval: str, limit: int = 250, market_type: str = "spot") -> List[list]:
    """
    Свечи с Bybit V5 (https://bybit-exchange.github.io/docs/v5/market/kline).
    market_type: 'spot' для спота, 'linear' для USDT-M фьючерсов.
    Возвращает список в формате Binance: [open_time_ms, open, high, low, close, volume].
    """
    await _bybit_rate_limiter.acquire()
    bybit_interval = BYBIT_INTERVAL_MAP.get(interval, interval)
    clean_symbol = symbol.replace("/", "").upper()
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": market_type, "symbol": clean_symbol, "interval": bybit_interval, "limit": min(limit, 1000)}
    resp = await http_client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Bybit error for {symbol}: {data.get('retMsg')}")
    raw_list = data.get("result", {}).get("list", [])
    # Bybit отдаёт от новых к старым — разворачиваем в хронологический порядок (как у Binance)
    normalized = []
    for c in reversed(raw_list):
        # Bybit: [start, open, high, low, close, volume, turnover]
        normalized.append([int(c[0]), c[1], c[2], c[3], c[4], c[5]])
    return normalized


async def fetch_bybit_open_interest(symbol: str) -> float:
    """Открытый интерес с Bybit V5 (только для деривативов, category='linear')."""
    await _bybit_rate_limiter.acquire()
    clean_symbol = symbol.replace("/", "").upper()
    url = "https://api.bybit.com/v5/market/open-interest"
    params = {"category": "linear", "symbol": clean_symbol, "intervalTime": "5min", "limit": 1}
    resp = await http_client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Bybit OI error for {symbol}: {data.get('retMsg')}")
    records = data.get("result", {}).get("list", [])
    return float(records[0]["openInterest"]) if records else 0.0


async def fetch_bybit_funding_rate(symbol: str) -> float:
    """Текущая ставка финансирования с Bybit V5."""
    await _bybit_rate_limiter.acquire()
    clean_symbol = symbol.replace("/", "").upper()
    url = "https://api.bybit.com/v5/market/funding-history"
    params = {"category": "linear", "symbol": clean_symbol, "limit": 1}
    resp = await http_client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise ValueError(f"Bybit funding error for {symbol}: {data.get('retMsg')}")
    records = data.get("result", {}).get("list", [])
    return float(records[0]["fundingRate"]) if records else 0.0


async def fetch_okx_klines(symbol: str, interval: str, limit: int = 250, is_futures: bool = False) -> List[list]:
    """
    Свечи с OKX V5 (https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks-history).
    Возвращает список в формате Binance: [open_time_ms, open, high, low, close, volume].
    """
    await _okx_rate_limiter.acquire()
    base_pair = symbol.replace("/", "-").upper()
    okx_symbol = f"{base_pair}-SWAP" if is_futures else base_pair
    okx_interval = OKX_INTERVAL_MAP.get(interval, interval)
    # OKX отдаёт максимум 100-300 свечей за вызов истории; для наших TIMEFRAMES этого достаточно.
    endpoint = "history-candles" if limit > 100 else "candles"
    url = f"https://www.okx.com/api/v5/market/{endpoint}"
    params = {"instId": okx_symbol, "bar": okx_interval, "limit": min(limit, 300)}
    resp = await http_client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX error for {symbol}: {data.get('msg')}")
    raw_list = data.get("data", [])
    # OKX тоже отдаёт от новых к старым
    normalized = []
    for c in reversed(raw_list):
        # OKX: [ts, open, high, low, close, vol, volCcy, volCcyQuote, confirm]
        normalized.append([int(c[0]), c[1], c[2], c[3], c[4], c[5]])
    return normalized


# ============================================================
# НОВЫЕ ФУНКЦИИ ДЛЯ ПРОФЕССИОНАЛЬНОГО АНАЛИЗА (Binance)
# ============================================================

async def fetch_oi_delta(symbol: str) -> float:
    """
    Рассчитывает изменение Открытого Интереса (OI) за последний час в %.
    Возвращает положительное значение при росте OI, отрицательное при падении.
    """
    try:
        clean_symbol = symbol.replace("/", "")
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {"symbol": clean_symbol, "period": "1h", "limit": 2}
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if len(data) >= 2:
            oi_now = float(data[-1]["sumOpenInterest"])
            oi_prev = float(data[-2]["sumOpenInterest"])
            if oi_prev > 0:
                return ((oi_now - oi_prev) / oi_prev) * 100.0
    except Exception as e:
        logger.warning(f"Не удалось получить OI Delta для {symbol}: {e}")
    return 0.0


async def fetch_order_book_imbalance(symbol: str) -> Tuple[float, str]:
    """
    Рассчитывает дисбаланс стакана (Bids/Asks) в пределах 1% от цены.
    Возвращает кортеж: (skew_ratio, dominant_side).
    """
    try:
        clean_symbol = symbol.replace("/", "")
        url = "https://fapi.binance.com/fapi/v1/depth"
        params = {"symbol": clean_symbol, "limit": 100}
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        bids = sum([float(b[1]) for b in data["bids"]])
        asks = sum([float(a[1]) for a in data["asks"]])
        
        if asks > 0:
            skew = round(bids / asks, 2)
        else:
            skew = 1.0
            
        if skew > 1.2:
            dom = "BIDS_HEAVY_BUY_WALLS"
        elif skew < 0.8:
            dom = "ASKS_HEAVY_SELL_WALLS"
        else:
            dom = "NEUTRAL"
            
        return skew, dom
    except Exception as e:
        logger.warning(f"Не удалось получить Order Book Imbalance для {symbol}: {e}")
        return 1.0, "NEUTRAL"


async def calculate_cvd_from_agg_trades(symbol: str, limit: int = 1000) -> float:
    """
    Агрегирует сделки для расчета CVD (Кумулятивной Дельты Объема).
    Возвращает значение CVD за последние `limit` сделок (положительное - покупки, отрицательное - продажи).
    """
    try:
        clean_symbol = symbol.replace("/", "")
        url = "https://fapi.binance.com/fapi/v1/aggTrades"
        params = {"symbol": clean_symbol, "limit": limit}
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        
        cvd = 0.0
        for trade in resp.json():
            qty = float(trade["q"])
            price = float(trade["p"])
            if not trade["m"]:  # m = isBuyerMaker (если False, покупатель агрессор)
                cvd += qty * price
            else:
                cvd -= qty * price
        return cvd
    except Exception as e:
        logger.warning(f"Не удалось получить CVD для {symbol}: {e}")
        return 0.0


async def close_http():
    """Закрытие HTTP-клиента для предотвращения утечек."""
    await http_client.aclose()