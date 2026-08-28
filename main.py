import os
import logging
import asyncio
import re
import pandas as pd
import pandas_ta as ta
import time
import httpx
import json
import math
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from telegram import (Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice)
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes,
    PreCheckoutQueryHandler, CallbackQueryHandler
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from dotenv import load_dotenv

load_dotenv(dotenv_path="/opt/tradercryptoai-bot/.env")

from crypto_pay import create_invoice as create_crypto_invoice, get_invoice
from config import SUB_PRICE_USDT, SUB_DAYS, SUB_PRICE_STARS, INACTIVE_USERS_LIMIT, INACTIVE_USERS_RETENTION_DAYS
from db import (
    init_db, append_message, get_history, clear_history, clear_pair_history,
    check_subscription, create_invoice_record, mark_invoice_paid,
    get_invoice_user, is_invoice_paid, activate_subscription, get_or_create_user,
    get_active_users_full, get_stats, increment_daily_count,
    get_daily_requests_count, check_daily_limit, get_last_pair, get_last_analysis,
    save_signal_outcome, get_inactive_users, get_inactive_users_count,
    save_active_signal, delete_active_signal, get_all_active_signals
)
from system_prompt import SYSTEM_PROMPT
from cache import cached, cache_cleanup_task
from market_data_fallback import (
    fetch_bybit_klines, fetch_bybit_open_interest, fetch_bybit_funding_rate, fetch_okx_klines,
    close_http as close_fallback_http   # закрытие fallback клиента
)
from error_handler import global_error_handler, init_error_reporter
from validators import validate_ticker, sanitize_text
from engine.liquidity_engine import detect_liquidity_heatmap, detect_market_structure, compute_fib_levels
from engine.signal_engine import MarketState, generate_signals, TradeSetup, MultiSignal, Signal, storage as signal_storage

# ---------- FAQ ----------
FAQ_DIR = "/opt/tradercryptoai-bot/faq"
FAQ_DATA = {}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
CRYPTO_PAY_API_TOKEN = os.getenv('CRYPTO_PAY_API_TOKEN')

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("❌ Ключ Telegram не найден!")
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ Ключ DeepSeek не найден!")
if not CRYPTO_PAY_API_TOKEN:
    logger.warning("⚠️ CRYPTO_PAY_API_TOKEN не задан — CryptoPay отключён")

http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=20.0, read=60.0, write=60.0, pool=10.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=30, keepalive_expiry=15.0)
)

MODELS: List[Tuple[str, int, bool]] = [("deepseek-v4-flash", 60, False)]

GLOBAL_LLM_CONCURRENCY = 8
FAILURE_THRESHOLD = 5
COOLDOWN_SECONDS = 30
RACE_MAX_TIMEOUT = 20

_llm_semaphore = asyncio.Semaphore(GLOBAL_LLM_CONCURRENCY)
_model_failures: Dict[str, int] = {}
_model_cooldown_until: Dict[str, float] = {}
_circuit_lock = asyncio.Lock()

def load_faq():
    global FAQ_DATA
    json_path = os.path.join(FAQ_DIR, "faq.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                FAQ_DATA = json.load(f)
            logger.info(f"FAQ loaded: {len(FAQ_DATA)} questions")
        except Exception as e:
            logger.error(f"Failed to load FAQ: {e}")
            FAQ_DATA = {}
    else:
        logger.warning("faq.json not found")
        FAQ_DATA = {}

class AsyncRateLimiter:
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

_llm_rate_limiter = AsyncRateLimiter(calls_per_minute=190)
_binance_rate_limiter = AsyncRateLimiter(calls_per_minute=20 * 60)
_futures_rate_limiter = AsyncRateLimiter(calls_per_minute=20 * 60)

def _is_rate_limit_error(e: Exception) -> bool:
    s = str(e).lower()
    return "429" in s or "quota" in s

def _is_server_error(e: Exception) -> bool:
    s = str(e).lower()
    return "500" in s or "503" in s or "unavailable" in s

async def _model_available(model_name: str) -> bool:
    async with _circuit_lock:
        cooldown_until = _model_cooldown_until.get(model_name)
        if not cooldown_until:
            return True
        if time.time() >= cooldown_until:
            _model_cooldown_until.pop(model_name, None)
            return True
        return False

async def _register_failure(model_name: str):
    async with _circuit_lock:
        count = _model_failures.get(model_name, 0) + 1
        _model_failures[model_name] = count
        if count >= FAILURE_THRESHOLD:
            _model_cooldown_until[model_name] = time.time() + COOLDOWN_SECONDS
            _model_failures[model_name] = 0
            logger.warning(f"[LLM] Circuit breaker OPEN for {model_name}")

async def _register_success(model_name: str):
    async with _circuit_lock:
        _model_failures[model_name] = 0
        _model_cooldown_until.pop(model_name, None)

async def _call_model(model_name: str, timeout: int, chat_context: Any) -> Optional[str]:
    if not await _model_available(model_name):
        return None
    await _llm_rate_limiter.acquire()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set")
        return None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (chat_context or []):
        content = turn.get("content", "")
        if not content:
            continue
        role = "assistant" if turn.get("role") == "model" else "user"
        messages.append({"role": role, "content": content})
    if len(messages) == 1:
        return None
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 3000,
        "top_p": 0.9,
        "stream": True,
        "thinking": {"type": "disabled"},  # Задача модели — только оформление уже готовых данных
    }
    async with _llm_semaphore:
        try:
            async with http_client.stream("POST", "https://api.deepseek.com/v1/chat/completions", json=data, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                full_content = ""
                reasoning_content = ""
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]
                        if chunk_data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_data)
                            delta = chunk["choices"][0]["delta"]
                            if "content" in delta and delta["content"] is not None:
                                full_content += delta["content"]
                            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                                reasoning_content += delta["reasoning_content"]
                        except json.JSONDecodeError:
                            continue
                text = full_content.strip()
                if text:
                    await _register_success(model_name)
                    return text
                if reasoning_content.strip():
                    logger.warning(f"{model_name}: получен только reasoning_content без content (обрезано? см. max_tokens) — ответ отклонён")
                return None
        except httpx.TimeoutException:
            await _register_failure(model_name)
            return None
        except Exception as e:
            if _is_rate_limit_error(e):
                raise
            if _is_server_error(e):
                await _register_failure(model_name)
            return None

async def generate_with_routing(chat_context: Any, user_id: int, is_premium: bool = False) -> Optional[str]:
    available_models = [(name, timeout) for name, timeout, premium_only in MODELS if not premium_only or is_premium]
    if not available_models:
        return None
    model_name, timeout = available_models[0]
    try:
        result = await _call_model(model_name, timeout, chat_context)
        return result
    except Exception as e:
        if _is_rate_limit_error(e):
            return None
        return None

import numpy as np

TIMEFRAMES = ["1d", "4h", "1h", "15m"]

def to_native(obj):
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def sanitize_dict(d):
    return {k: to_native(v) for k, v in d.items()}

def classify_trend(price, ema50, ema200):
    if price > ema50 > ema200:
        return "strong_up"
    if price < ema50 < ema200:
        return "strong_down"
    if price > ema200:
        return "weak_up"
    if price < ema200:
        return "weak_down"
    return "sideways"

def compute_levels(df):
    highs = df["high"]
    lows = df["low"]
    resistance = highs.rolling(5, center=True).max()
    support = lows.rolling(5, center=True).min()
    r = resistance.dropna().tail(20).max()
    s = support.dropna().tail(20).min()
    if pd.isna(r):
        r = df["close"].iloc[-1]
    if pd.isna(s):
        s = df["close"].iloc[-1]
    return float(s), float(r)    

def calculate_last_indicators(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.atr(append=True)
    df.ta.adx(length=14, append=True)
    last = df.iloc[-1]
    price = float(last["close"])
    ema50 = float(last["EMA_50"])
    ema200 = float(last["EMA_200"])
    trend = classify_trend(price, ema50, ema200)
    support, resistance = compute_levels(df)
    if pd.isna(support):
        support = price
    if pd.isna(resistance):
        resistance = price
    atr_value = float(last.get("ATR_14", last.get("ATRr_14", 0.0)))
    if pd.isna(atr_value) or atr_value == 0:
        atr_value = price * 0.01
    adx_value = float(last["ADX_14"]) if "ADX_14" in df.columns else 0.0
    result = {
        "price": price, "close": price, "ema50": ema50, "ema100": float(last["EMA_100"]),
        "ema200": ema200, "rsi": float(last["RSI_14"]), "macd": float(last["MACD_12_26_9"]),
        "macd_signal": float(last["MACDs_12_26_9"]), "macd_hist": float(last["MACDh_12_26_9"]),
        "atr": atr_value, "adx": adx_value, "trend": trend, "support": support, "resistance": resistance,
        "volume": float(last["volume"]),
    }
    return sanitize_dict(result)

def alignment_score(tf_map):
    trends = [v["trend"] for v in tf_map.values()]
    up = sum("up" in t for t in trends)
    down = sum("down" in t for t in trends)
    score = max(up, down) / len(trends) if trends else 0
    return round(score, 2)

def map_interval(tf: str) -> str:
    mapping = {"1d": "1d", "4h": "4h", "1h": "1h", "15m": "15m"}
    return mapping.get(tf, "1h")

async def fetch_klines(symbol: str, interval: str, limit: int = 250) -> list:
    await _binance_rate_limiter.acquire()
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.replace("/", ""), "interval": interval, "limit": limit}
    try:
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            raise ValueError(f"Invalid symbol: {symbol}")
        logger.error(f"Error fetching klines for {symbol} {interval}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching klines for {symbol} {interval}: {e}")
        raise

async def fetch_open_interest(symbol: str) -> float:
    await _binance_rate_limiter.acquire()
    url = "https://fapi.binance.com/fapi/v1/openInterest"
    params = {"symbol": symbol.replace("/", "")}
    try:
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        return float(resp.json()["openInterest"])
    except Exception:
        return 0.0

async def fetch_funding_rate(symbol: str) -> float:
    await _binance_rate_limiter.acquire()
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol.replace("/", ""), "limit": 1}
    try:
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return float(data[0]["fundingRate"]) if data else 0.0
    except Exception:
        return 0.0

async def fetch_agg_trades(symbol: str, limit: int = 1000) -> List[Dict]:
    await _binance_rate_limiter.acquire()
    url = "https://api.binance.com/api/v3/aggTrades"
    params = {"symbol": symbol.replace("/", ""), "limit": limit}
    try:
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

async def fetch_coingecko_global() -> Dict:
    await _binance_rate_limiter.acquire()
    url = "https://api.coingecko.com/api/v3/global"
    try:
        resp = await http_client.get(url)
        resp.raise_for_status()
        return resp.json().get("data", {})
    except Exception:
        return {}

# ========== РЕЗИЛИЕНТНЫЕ ОБЁРТКИ (Binance -> Bybit -> OKX) ==========
async def fetch_klines_resilient(symbol: str, interval: str, limit: int = 250) -> list:
    try:
        return await fetch_klines(symbol, interval, limit)
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Binance klines недоступен для {symbol} {interval}, пробую Bybit: {e}")
    try:
        return await fetch_bybit_klines(symbol, interval, limit, market_type="spot")
    except Exception as e:
        logger.warning(f"Bybit klines недоступен для {symbol} {interval}, пробую OKX: {e}")
    return await fetch_okx_klines(symbol, interval, limit, is_futures=False)

async def fetch_futures_klines_resilient(symbol: str, interval: str, limit: int = 250) -> list:
    try:
        return await fetch_futures_klines(symbol, interval, limit)
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"Binance futures klines недоступен для {symbol} {interval}, пробую Bybit: {e}")
    try:
        return await fetch_bybit_klines(symbol, interval, limit, market_type="linear")
    except Exception as e:
        logger.warning(f"Bybit futures klines недоступен для {symbol} {interval}, пробую OKX: {e}")
    return await fetch_okx_klines(symbol, interval, limit, is_futures=True)

async def fetch_open_interest_resilient(symbol: str) -> float:
    await _binance_rate_limiter.acquire()
    try:
        resp = await http_client.get("https://fapi.binance.com/fapi/v1/openInterest", params={"symbol": symbol.replace("/", "")})
        resp.raise_for_status()
        return float(resp.json()["openInterest"])
    except Exception as e:
        logger.warning(f"Binance OI недоступен для {symbol}, пробую Bybit: {e}")
    try:
        return await fetch_bybit_open_interest(symbol)
    except Exception as e:
        logger.warning(f"Bybit OI тоже недоступен для {symbol}: {e}")
        return 0.0

async def fetch_funding_rate_resilient(symbol: str) -> float:
    await _binance_rate_limiter.acquire()
    try:
        resp = await http_client.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol": symbol.replace("/", ""), "limit": 1})
        resp.raise_for_status()
        data = resp.json()
        return float(data[0]["fundingRate"]) if data else 0.0
    except Exception as e:
        logger.warning(f"Binance funding недоступен для {symbol}, пробую Bybit: {e}")
    try:
        return await fetch_bybit_funding_rate(symbol)
    except Exception as e:
        logger.warning(f"Bybit funding тоже недоступен для {symbol}: {e}")
        return 0.0

def df_to_candles(df: pd.DataFrame) -> List[Tuple[float, float, float, float, float]]:
    if df is None or df.empty:
        return []
    candles = []
    for _, row in df.iterrows():
        candles.append((float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"]), float(row["volume"])))
    return candles

# ========== НОВЫЕ ФУНКЦИИ ДЛЯ ФЬЮЧЕРСОВ ==========
async def fetch_futures_klines(symbol: str, interval: str, limit: int = 250) -> list:
    """Получение свечей для USDT-M фьючерсов (fapi)"""
    await _futures_rate_limiter.acquire()
    clean_symbol = symbol.replace("/", "")
    url = "https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": clean_symbol, "interval": interval, "limit": limit}
    try:
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 400:
            raise ValueError(f"Фьючерсная пара {symbol} не найдена")
        logger.error(f"Ошибка fetch_futures_klines {symbol} {interval}: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка fetch_futures_klines {symbol} {interval}: {e}")
        raise

async def fetch_futures_agg_trades(symbol: str, limit: int = 1000) -> List[Dict]:
    """Агрегированные сделки для USDT-M фьючерсов"""
    await _futures_rate_limiter.acquire()
    clean_symbol = symbol.replace("/", "")
    url = "https://fapi.binance.com/fapi/v1/aggTrades"
    params = {"symbol": clean_symbol, "limit": limit}
    try:
        resp = await http_client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []

@cached(ttl=60, key_prefix="binance:")
async def get_spot_data(symbol: str):
    try:
        ohlcv_dfs = {}
        last_values = {}
        ema50_1h = 0.0
        ema200_1h = 0.0
        ema50_15m = 0.0
        ema200_15m = 0.0

        for tf in TIMEFRAMES:
            interval = map_interval(tf)
            try:
                raw = await fetch_klines_resilient(symbol, interval, limit=250)
            except ValueError as e:
                return {"error": f"Тикер {symbol} не найден. Укажите существующую пару, например BTC/USDT."}
            filtered = [candle[:6] for candle in raw]
            df = pd.DataFrame(filtered, columns=["timestamp", "open", "high", "low", "close", "volume"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            if df.empty:
                raise ValueError(f"Получены пустые данные для {symbol} {interval}")
            ohlcv_dfs[tf] = df
            last = calculate_last_indicators(df)
            last_values[tf] = last
            if tf == "1h":
                ema50_1h = last.get("ema50", 0.0)
                ema200_1h = last.get("ema200", 0.0)
            elif tf == "15m":
                ema50_15m = last.get("ema50", 0.0)
                ema200_15m = last.get("ema200", 0.0)

        oi = await fetch_open_interest_resilient(symbol)
        funding = await fetch_funding_rate_resilient(symbol)
        agg_trades = await fetch_agg_trades(symbol, limit=1000)
        global_data = await fetch_coingecko_global()
        btc_dominance = global_data.get("market_cap_percentage", {}).get("btc", 0.0)
        alt_season = btc_dominance < 45.0

        cvd_1h = 0.0
        for trade in agg_trades:
            p = float(trade['p'])
            q = float(trade['q'])
            if not trade['m']:
                cvd_1h += q * p
            else:
                cvd_1h -= q * p

        multi_tf = last_values or {}
        def safe_get(tf, key, default=None):
            return multi_tf.get(tf, {}).get(key, default)

        align = alignment_score(multi_tf)
        close_4h = safe_get("4h", "close", 0.0)
        close_15m = safe_get("15m", "close", 0.0)

        def map_trend(t):
            if t in ("strong_up", "weak_up"): return "bullish"
            if t in ("strong_down", "weak_down"): return "bearish"
            return "sideways"

        def _structure_from_df(tf):
            df = ohlcv_dfs.get(tf)
            if df is None or df.empty:
                return "range"
            return detect_market_structure(df["high"].values, df["low"].values, df["close"].values)

        price = safe_get("1h", "price", 0.0)
        trend_15m = map_trend(safe_get("15m", "trend"))
        trend_1h = map_trend(safe_get("1h", "trend"))
        trend_4h = map_trend(safe_get("4h", "trend"))
        trend_1d = map_trend(safe_get("1d", "trend"))
        structure_15m = _structure_from_df("15m")
        structure_1h = _structure_from_df("1h")
        structure_4h = _structure_from_df("4h")
        rsi_1h = safe_get("1h", "rsi", 50.0)
        rsi_4h = safe_get("4h", "rsi", 50.0)
        rsi_1d = safe_get("1d", "rsi", 50.0)
        adx_1h = safe_get("1h", "adx", 20.0)
        adx_4h = safe_get("4h", "adx", 20.0)
        adx_1d = safe_get("1d", "adx", 20.0)
        atr_1h = safe_get("1h", "atr", 0.0)
        atr_4h = safe_get("4h", "atr", 0.0)
        if atr_4h is None or (isinstance(atr_4h, float) and math.isnan(atr_4h)) or atr_4h == 0:
            atr_4h = price * 0.01
        atr_1d = safe_get("1d", "atr", 0.0)
        macd_4h = safe_get("4h", "macd", 0.0)
        macd_signal_4h = safe_get("4h", "macd_signal", 0.0)
        ema50_4h = safe_get("4h", "ema50", 0.0)
        ema200_4h = safe_get("4h", "ema200", 0.0)
        df_1d_for_fib = ohlcv_dfs.get("1d")
        fib_levels = compute_fib_levels(df_1d_for_fib["high"].values, df_1d_for_fib["low"].values, period=100) if df_1d_for_fib is not None and not df_1d_for_fib.empty else {}
        df_1h = ohlcv_dfs.get("1h")
        if df_1h is not None and len(df_1h) >= 2:
            volume_now = float(df_1h["volume"].iloc[-1])
            volume_prev = float(df_1h["volume"].iloc[-2])
            volume_trend_1h = "rising" if volume_now > volume_prev else "falling"
        else:
            volume_now = safe_get("1h", "volume", 0.0)
            volume_trend_1h = "falling"
        support_15m = safe_get("15m", "support", price)
        resistance_15m = safe_get("15m", "resistance", price)
        support_1h = safe_get("1h", "support", price)
        resistance_1h = safe_get("1h", "resistance", price)
        support_4h = safe_get("4h", "support", price)
        resistance_4h = safe_get("4h", "resistance", price)
        if support_4h is None or (isinstance(support_4h, float) and math.isnan(support_4h)):
            support_4h = price
        if resistance_4h is None or (isinstance(resistance_4h, float) and math.isnan(resistance_4h)):
            resistance_4h = price
        support_1d = safe_get("1d", "support", price)
        resistance_1d = safe_get("1d", "resistance", price)

        def get_arrays(tf):
            df = ohlcv_dfs[tf]
            if df is None or df.empty:
                return None, None, None, None
            return (df["high"].values, df["low"].values, df["close"].values, df["volume"].values)

        liquidity_1h = liquidity_4h = liquidity_1d = None
        if "1h" in ohlcv_dfs:
            h, l, c, v = get_arrays("1h")
            if h is not None:
                liquidity_1h = detect_liquidity_heatmap(h, l, c, v, price)
        if "4h" in ohlcv_dfs:
            h, l, c, v = get_arrays("4h")
            if h is not None:
                liquidity_4h = detect_liquidity_heatmap(h, l, c, v, price)
        if "1d" in ohlcv_dfs:
            h, l, c, v = get_arrays("1d")
            if h is not None:
                liquidity_1d = detect_liquidity_heatmap(h, l, c, v, price)

        close_1h = last_values.get("1h", {}).get("close", price)
        rsi_15m = last_values.get("15m", {}).get("rsi", 50.0)
        atr_15m = last_values.get("15m", {}).get("atr", atr_1h * 0.6 if atr_1h else price * 0.01)

        structure_1d = _structure_from_df("1d")

        df_4h = ohlcv_dfs.get("4h")
        if df_4h is not None and len(df_4h) >= 2:
            vol_now_4h = float(df_4h["volume"].iloc[-1])
            vol_prev_4h = float(df_4h["volume"].iloc[-2])
            volume_trend_4h = "rising" if vol_now_4h > vol_prev_4h else "falling"
        else:
            volume_trend_4h = "falling"

        def calc_rel_volume(df, lookback=20):
            if df is None or len(df) < lookback + 1:
                return 1.0
            avg_vol = df["volume"].iloc[-lookback-1:-1].mean()
            current_vol = df["volume"].iloc[-1]
            return float(current_vol / avg_vol) if avg_vol > 0 else 1.0

        relative_volume_1h = calc_rel_volume(ohlcv_dfs.get("1h"))
        relative_volume_4h = calc_rel_volume(ohlcv_dfs.get("4h"))
        cvd_4h = 0.0
        liquidation_long = liquidation_short = 0.0
        ls_ratio = 1.0
        fear_greed = None

        candles_4h = df_to_candles(ohlcv_dfs.get("4h"))
        candles_15m = df_to_candles(ohlcv_dfs.get("15m"))

        state = MarketState(
            symbol=symbol, price=price, trend_15m=trend_15m, trend_1h=trend_1h, trend_4h=trend_4h, trend_1d=trend_1d,
            close_4h=close_4h, close_15m=close_15m, structure_15m=structure_15m, structure_1h=structure_1h, structure_4h=structure_4h,
            rsi_1h=rsi_1h, rsi_4h=rsi_4h, rsi_1d=rsi_1d, adx_1h=adx_1h, adx_4h=adx_4h, adx_1d=adx_1d,
            atr_1h=atr_1h, atr_4h=atr_4h, atr_1d=atr_1d, volume_trend_1h=volume_trend_1h,
            support_15m=support_15m, resistance_15m=resistance_15m, support_1h=support_1h, resistance_1h=resistance_1h,
            support_4h=support_4h, resistance_4h=resistance_4h, support_1d=support_1d, resistance_1d=resistance_1d,
            liquidity_1h=liquidity_1h, liquidity_4h=liquidity_4h, liquidity_1d=liquidity_1d, vwap=None, fear_greed=fear_greed,
            oi=oi, funding=funding, cvd_1h=cvd_1h, btc_dominance=btc_dominance, alt_season=alt_season,
            close_1h=close_1h, rsi_15m=rsi_15m, atr_15m=atr_15m, structure_1d=structure_1d,
            volume_trend_4h=volume_trend_4h, relative_volume_1h=relative_volume_1h, relative_volume_4h=relative_volume_4h,
            cvd_4h=cvd_4h, liquidation_long=liquidation_long, liquidation_short=liquidation_short, ls_ratio=ls_ratio,
            ema50_1h=ema50_1h, ema200_1h=ema200_1h, ema50_15m=ema50_15m, ema200_15m=ema200_15m,
            ema50_4h=ema50_4h, ema200_4h=ema200_4h, macd_4h=macd_4h, macd_signal_4h=macd_signal_4h, fib_levels=fib_levels,
            range_market=False
        )

        signals = generate_signals(state, candles_4h=candles_4h, candles_15m=candles_15m)
        logger.info(f"Signals generated: swing={signals.swing.direction}")

        trend_strength = "strong" if adx_1d > 40 else "moderate" if adx_1d > 20 else "weak"

        return {
            "price": price, "market_type": "spot", "signals": signals, "trend_alignment": align,
            "daily_trend": trend_1d, "support_1d": support_1d, "resistance_1d": resistance_1d,
            "adx_1d": adx_1d, "trend_strength": trend_strength, "oi": oi, "funding": funding,
            "cvd_1h": cvd_1h, "btc_dominance": btc_dominance, "alt_season": alt_season,
            "rsi_1h": rsi_1h, "rsi_4h": rsi_4h, "atr_4h": atr_4h,
            "macd_4h": macd_4h, "macd_signal_4h": macd_signal_4h,
            "ema50_4h": ema50_4h, "ema200_4h": ema200_4h, "fib_levels": fib_levels,
            "relative_volume_1h": relative_volume_1h,
        }
    except Exception as e:
        logger.error(f"Ошибка в get_binance_data для {symbol}: {e}", exc_info=True)
        return {"error": str(e)}

@cached(ttl=60, key_prefix="binance:futures:")
async def get_futures_data(symbol: str):
    try:
        ohlcv_dfs = {}
        last_values = {}
        ema50_1h = 0.0
        ema200_1h = 0.0
        ema50_15m = 0.0
        ema200_15m = 0.0

        for tf in TIMEFRAMES:
            interval = map_interval(tf)
            try:
                raw = await fetch_futures_klines_resilient(symbol, interval, limit=250)
            except ValueError as e:
                return {"error": f"Фьючерсный тикер {symbol} не найден. Укажите существующую пару."}
            filtered = [candle[:6] for candle in raw]
            df = pd.DataFrame(filtered, columns=["timestamp", "open", "high", "low", "close", "volume"])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            if df.empty:
                raise ValueError(f"Получены пустые данные для {symbol} {interval}")
            ohlcv_dfs[tf] = df
            last = calculate_last_indicators(df)
            last_values[tf] = last
            if tf == "1h":
                ema50_1h = last.get("ema50", 0.0)
                ema200_1h = last.get("ema200", 0.0)
            elif tf == "15m":
                ema50_15m = last.get("ema50", 0.0)
                ema200_15m = last.get("ema200", 0.0)

        oi = await fetch_open_interest_resilient(symbol)
        funding = await fetch_funding_rate_resilient(symbol)
        agg_trades = await fetch_futures_agg_trades(symbol, limit=1000)
        global_data = await fetch_coingecko_global()
        btc_dominance = global_data.get("market_cap_percentage", {}).get("btc", 0.0)
        alt_season = btc_dominance < 45.0

        cvd_1h = 0.0
        for trade in agg_trades:
            p = float(trade['p'])
            q = float(trade['q'])
            if not trade['m']:
                cvd_1h += q * p
            else:
                cvd_1h -= q * p

        multi_tf = last_values or {}
        def safe_get(tf, key, default=None):
            return multi_tf.get(tf, {}).get(key, default)

        align = alignment_score(multi_tf)
        close_4h = safe_get("4h", "close", 0.0)
        close_15m = safe_get("15m", "close", 0.0)

        def map_trend(t):
            if t in ("strong_up", "weak_up"): return "bullish"
            if t in ("strong_down", "weak_down"): return "bearish"
            return "sideways"

        def _structure_from_df(tf):
            df = ohlcv_dfs.get(tf)
            if df is None or df.empty:
                return "range"
            return detect_market_structure(df["high"].values, df["low"].values, df["close"].values)

        price = safe_get("1h", "price", 0.0)
        trend_15m = map_trend(safe_get("15m", "trend"))
        trend_1h = map_trend(safe_get("1h", "trend"))
        trend_4h = map_trend(safe_get("4h", "trend"))
        trend_1d = map_trend(safe_get("1d", "trend"))
        structure_15m = _structure_from_df("15m")
        structure_1h = _structure_from_df("1h")
        structure_4h = _structure_from_df("4h")
        rsi_1h = safe_get("1h", "rsi", 50.0)
        rsi_4h = safe_get("4h", "rsi", 50.0)
        rsi_1d = safe_get("1d", "rsi", 50.0)
        adx_1h = safe_get("1h", "adx", 20.0)
        adx_4h = safe_get("4h", "adx", 20.0)
        adx_1d = safe_get("1d", "adx", 20.0)
        atr_1h = safe_get("1h", "atr", 0.0)
        atr_4h = safe_get("4h", "atr", 0.0)
        if atr_4h is None or (isinstance(atr_4h, float) and math.isnan(atr_4h)) or atr_4h == 0:
            atr_4h = price * 0.01
        atr_1d = safe_get("1d", "atr", 0.0)
        macd_4h = safe_get("4h", "macd", 0.0)
        macd_signal_4h = safe_get("4h", "macd_signal", 0.0)
        ema50_4h = safe_get("4h", "ema50", 0.0)
        ema200_4h = safe_get("4h", "ema200", 0.0)
        df_1d_for_fib = ohlcv_dfs.get("1d")
        fib_levels = compute_fib_levels(df_1d_for_fib["high"].values, df_1d_for_fib["low"].values, period=100) if df_1d_for_fib is not None and not df_1d_for_fib.empty else {}
        df_1h = ohlcv_dfs.get("1h")
        if df_1h is not None and len(df_1h) >= 2:
            volume_now = float(df_1h["volume"].iloc[-1])
            volume_prev = float(df_1h["volume"].iloc[-2])
            volume_trend_1h = "rising" if volume_now > volume_prev else "falling"
        else:
            volume_now = safe_get("1h", "volume", 0.0)
            volume_trend_1h = "falling"
        support_15m = safe_get("15m", "support", price)
        resistance_15m = safe_get("15m", "resistance", price)
        support_1h = safe_get("1h", "support", price)
        resistance_1h = safe_get("1h", "resistance", price)
        support_4h = safe_get("4h", "support", price)
        resistance_4h = safe_get("4h", "resistance", price)
        if support_4h is None or (isinstance(support_4h, float) and math.isnan(support_4h)):
            support_4h = price
        if resistance_4h is None or (isinstance(resistance_4h, float) and math.isnan(resistance_4h)):
            resistance_4h = price
        support_1d = safe_get("1d", "support", price)
        resistance_1d = safe_get("1d", "resistance", price)

        def get_arrays(tf):
            df = ohlcv_dfs[tf]
            if df is None or df.empty:
                return None, None, None, None
            return (df["high"].values, df["low"].values, df["close"].values, df["volume"].values)

        liquidity_1h = liquidity_4h = liquidity_1d = None
        if "1h" in ohlcv_dfs:
            h, l, c, v = get_arrays("1h")
            if h is not None:
                liquidity_1h = detect_liquidity_heatmap(h, l, c, v, price)
        if "4h" in ohlcv_dfs:
            h, l, c, v = get_arrays("4h")
            if h is not None:
                liquidity_4h = detect_liquidity_heatmap(h, l, c, v, price)
        if "1d" in ohlcv_dfs:
            h, l, c, v = get_arrays("1d")
            if h is not None:
                liquidity_1d = detect_liquidity_heatmap(h, l, c, v, price)

        close_1h = last_values.get("1h", {}).get("close", price)
        rsi_15m = last_values.get("15m", {}).get("rsi", 50.0)
        atr_15m = last_values.get("15m", {}).get("atr", atr_1h * 0.6 if atr_1h else price * 0.01)

        structure_1d = _structure_from_df("1d")

        df_4h = ohlcv_dfs.get("4h")
        if df_4h is not None and len(df_4h) >= 2:
            vol_now_4h = float(df_4h["volume"].iloc[-1])
            vol_prev_4h = float(df_4h["volume"].iloc[-2])
            volume_trend_4h = "rising" if vol_now_4h > vol_prev_4h else "falling"
        else:
            volume_trend_4h = "falling"

        def calc_rel_volume(df, lookback=20):
            if df is None or len(df) < lookback + 1:
                return 1.0
            avg_vol = df["volume"].iloc[-lookback-1:-1].mean()
            current_vol = df["volume"].iloc[-1]
            return float(current_vol / avg_vol) if avg_vol > 0 else 1.0

        relative_volume_1h = calc_rel_volume(ohlcv_dfs.get("1h"))
        relative_volume_4h = calc_rel_volume(ohlcv_dfs.get("4h"))
        cvd_4h = 0.0
        liquidation_long = liquidation_short = 0.0
        ls_ratio = 1.0
        fear_greed = None

        candles_4h = df_to_candles(ohlcv_dfs.get("4h"))
        candles_15m = df_to_candles(ohlcv_dfs.get("15m"))

        state = MarketState(
            symbol=symbol, price=price, trend_15m=trend_15m, trend_1h=trend_1h, trend_4h=trend_4h, trend_1d=trend_1d,
            close_4h=close_4h, close_15m=close_15m, structure_15m=structure_15m, structure_1h=structure_1h, structure_4h=structure_4h,
            rsi_1h=rsi_1h, rsi_4h=rsi_4h, rsi_1d=rsi_1d, adx_1h=adx_1h, adx_4h=adx_4h, adx_1d=adx_1d,
            atr_1h=atr_1h, atr_4h=atr_4h, atr_1d=atr_1d, volume_trend_1h=volume_trend_1h,
            support_15m=support_15m, resistance_15m=resistance_15m, support_1h=support_1h, resistance_1h=resistance_1h,
            support_4h=support_4h, resistance_4h=resistance_4h, support_1d=support_1d, resistance_1d=resistance_1d,
            liquidity_1h=liquidity_1h, liquidity_4h=liquidity_4h, liquidity_1d=liquidity_1d, vwap=None, fear_greed=fear_greed,
            oi=oi, funding=funding, cvd_1h=cvd_1h, btc_dominance=btc_dominance, alt_season=alt_season,
            close_1h=close_1h, rsi_15m=rsi_15m, atr_15m=atr_15m, structure_1d=structure_1d,
            volume_trend_4h=volume_trend_4h, relative_volume_1h=relative_volume_1h, relative_volume_4h=relative_volume_4h,
            cvd_4h=cvd_4h, liquidation_long=liquidation_long, liquidation_short=liquidation_short, ls_ratio=ls_ratio,
            ema50_1h=ema50_1h, ema200_1h=ema200_1h, ema50_15m=ema50_15m, ema200_15m=ema200_15m,
            ema50_4h=ema50_4h, ema200_4h=ema200_4h, macd_4h=macd_4h, macd_signal_4h=macd_signal_4h, fib_levels=fib_levels,
            range_market=False
        )

        signals = generate_signals(state, candles_4h=candles_4h, candles_15m=candles_15m)
        logger.info(f"Signals generated: swing={signals.swing.direction}")

        trend_strength = "strong" if adx_1d > 40 else "moderate" if adx_1d > 20 else "weak"

        return {
            "price": price,
            "market_type": "futures",
            "signals": signals,
            "trend_alignment": align,
            "daily_trend": trend_1d,
            "support_1d": support_1d,
            "resistance_1d": resistance_1d,
            "adx_1d": adx_1d,
            "trend_strength": trend_strength,
            "oi": oi,
            "funding": funding,
            "cvd_1h": cvd_1h,
            "btc_dominance": btc_dominance,
            "alt_season": alt_season,
            "rsi_1h": rsi_1h,
            "rsi_4h": rsi_4h,
            "atr_4h": atr_4h,
            "macd_4h": macd_4h,
            "macd_signal_4h": macd_signal_4h,
            "ema50_4h": ema50_4h,
            "ema200_4h": ema200_4h,
            "fib_levels": fib_levels,
            "relative_volume_1h": relative_volume_1h,
        }
    except Exception as e:
        logger.error(f"Ошибка в get_futures_data для {symbol}: {e}", exc_info=True)
        return {"error": str(e)}

async def get_combined_analysis(symbol: str):
    futures_task = asyncio.create_task(get_futures_data(symbol))
    spot_task = asyncio.create_task(get_spot_data(symbol))
    futures, spot = await asyncio.gather(futures_task, spot_task, return_exceptions=True)

    if isinstance(futures, Exception) or (isinstance(futures, dict) and futures.get("error")):
        logger.warning(f"Фьючерсы для {symbol} недоступны, использую спот")
        if isinstance(spot, dict) and spot.get("error"):
            return {"error": f"Не удалось получить данные для {symbol}"}
        return spot

    if isinstance(spot, Exception) or (isinstance(spot, dict) and spot.get("error")):
        logger.warning(f"Спот для {symbol} недоступен, выдаю только фьючерсный анализ")
        return futures

    fut_signal = futures["signals"]
    spot_signal = spot["signals"]

    if (fut_signal.swing.direction == spot_signal.swing.direction and 
        fut_signal.swing.direction not in ("NO_TRADE", "WAIT")):
        new_conf = min(fut_signal.swing.confidence + 0.15, 0.95)
        fut_signal.swing.confidence = new_conf
        if hasattr(fut_signal, 'context') and fut_signal.context:
            fut_signal.context += f"\n✅ Подтверждено спотовым рынком. Совместная уверенность: {new_conf*100:.0f}%."
        else:
            fut_signal.context = f"✅ Подтверждено спотовым рынком. Совместная уверенность: {new_conf*100:.0f}%."
    elif (fut_signal.swing.direction != spot_signal.swing.direction and 
          fut_signal.swing.direction not in ("NO_TRADE", "WAIT")):
        if fut_signal.swing.confidence > 0.6:
            fut_signal.swing.confidence = max(fut_signal.swing.confidence - 0.2, 0.3)
            if hasattr(fut_signal, 'context') and fut_signal.context:
                fut_signal.context += f"\n⚠️ Расхождение со спотовым сигналом. Уверенность снижена до {fut_signal.swing.confidence*100:.0f}%."
            else:
                fut_signal.context = f"⚠️ Расхождение со спотовым сигналом. Уверенность снижена до {fut_signal.swing.confidence*100:.0f}%."
        else:
            # Исправлено: заменяем "WAIT" на "NO_TRADE" и обнуляем все уровни
            fut_signal.swing.direction = "NO_TRADE"
            fut_signal.swing.entry = None
            fut_signal.swing.stop = None
            fut_signal.swing.tp1 = None
            fut_signal.swing.tp2 = None
            fut_signal.swing.confidence = 0.0
            if hasattr(fut_signal, 'context') and fut_signal.context:
                fut_signal.context += "\n⚠️ Расхождение со спотовым сигналом. Режим ожидания."
            else:
                fut_signal.context = "⚠️ Расхождение со спотовым сигналом. Режим ожидания."

    futures["spot_support"] = spot.get("support_1d")
    futures["spot_resistance"] = spot.get("resistance_1d")
    return futures

@cached(ttl=3600, key_prefix="fng:")
async def get_fng_index():
    try:
        response = await http_client.get("https://api.alternative.me/fng/?limit=1")
        response.raise_for_status()
        data = response.json()['data'][0]
        return {"value": data['value'], "sentiment": data['value_classification']}
    except Exception as e:
        logger.error(f"Ошибка F&G: {e}")
        return None

OWNER_ID = 1589425210
EXEMPT_USERS = {396753592, 575106686}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    text = f"""Здравствуйте, {user.mention_html()}
<b>ARIXEN</b> - платформа профессионального анализа в криптотрейдинге.
По вашему запросу проводит многоуровневый анализ рыночных данных.
▫️ Анализ ликвидности: Поиск ключевых уровней на основе биржевых стаканов и объемов.
Алгоритмическая обработка потоков данных для выявления реальных зон интереса крупных игроков.
▫️ Нейросетевая модель агрегирует и обрабатывает массивы данных в отчет-рекомендацию.
Вы получаете готовый план для открытия позиции.

▫️ Для запуска анализа: 
• Отправьте запрос по паре BTC/USDT или другой.
• Получите готовый отчет для открытия позиции.

🔐 <i>Доступ предоставляется по подписке. 
▫️ Процесс работы - в разделе «ОТЧЕТЫ».</i>"""
    keyboard = [
        [InlineKeyboardButton("ИНСТРУКЦИЯ", callback_data="mains"), InlineKeyboardButton("ДОСТУП", callback_data="subscription")], 
        [InlineKeyboardButton("ЛОГИКА АНАЛИЗА", callback_data="metod"), InlineKeyboardButton("ОТЧЕТЫ", url="https://t.me/arixen_official")],
        [InlineKeyboardButton("ВОПРОСЫ", callback_data="faq"), InlineKeyboardButton("ПОДДЕРЖКА", url="https://t.me/ruby_tie?direct")],
    ]    
    await update.message.reply_html(text, reply_markup=InlineKeyboardMarkup(keyboard), disable_web_page_preview=True)

async def users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Доступ запрещён")
        return
    rows = get_active_users_full()
    if not rows:
        await update.message.reply_text("▫️ Активных пользователей нет")
        return
    lines = ["▫️ Активные подписчики:\n"]
    for i, r in enumerate(rows, 1):
        lines.append(f"{i}. {r['first_name'] or '-'} (@{r['username'] or '-'})\nID: {r['user_id']}\nДо: {r['expires_at']}\n")
    await update.message.reply_text("\n".join(lines))

async def inactive_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Доступ запрещён")
        return
    rows = get_inactive_users()
    total = len(rows)
    if not rows:
        await update.message.reply_text("▫️ Неактивных пользователей нет")
        return

    now = datetime.utcnow()
    over_retention = 0
    for r in rows:
        try:
            expires_dt = datetime.fromisoformat(r["expires_at"])
            if (now - expires_dt).days >= INACTIVE_USERS_RETENTION_DAYS:
                over_retention += 1
        except Exception:
            pass

    lines = [f"▫️ Неактивных подписчиков (истекли, не продлены): {total}"]
    if total > INACTIVE_USERS_LIMIT:
        lines.append(f"⚠️ Превышен лимит {INACTIVE_USERS_LIMIT} — стоит почистить старых.")
    if over_retention:
        lines.append(f"🗑️ Старше {INACTIVE_USERS_RETENTION_DAYS} дней без подписки: {over_retention} (кандидаты на удаление)")
    lines.append("")

    show_n = 30
    for i, r in enumerate(rows[:show_n], 1):
        try:
            expires_dt = datetime.fromisoformat(r["expires_at"])
            days_inactive = (now - expires_dt).days
        except Exception:
            days_inactive = "?"
        flag = "🗑️" if isinstance(days_inactive, int) and days_inactive >= INACTIVE_USERS_RETENTION_DAYS else "▫️"
        lines.append(f"{flag} {i}. {r['first_name'] or '-'} (@{r['username'] or '-'})\nID: {r['user_id']}\nИстекла: {r['expires_at']} ({days_inactive} дн. назад)\n")
    if total > show_n:
        lines.append(f"... и ещё {total - show_n}")

    await update.message.reply_text("\n".join(lines))

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != OWNER_ID:
        await update.message.reply_text("⛔ Доступ запрещён")
        return
    stats = get_stats()
    inactive_count = get_inactive_users_count()
    text = (f"▫️ **Статистика бота**\n\n▫️ Всего пользователей: {stats['total_users']}\n"
            f"⭐ Активных подписок: {stats['active_subs']}\n⛔ Истёкших подписок: {stats['expired_subs']}\n"
            f"🗑️ Неактивных (истекли, не продлены): {inactive_count}")
    await update.message.reply_text(text, parse_mode="Markdown")

async def notify_admins_new_subscription(context: ContextTypes.DEFAULT_TYPE, user, expires_at: str):
    text = (f"▫️**Новая подписка**\n\n👤 Пользователь: {user.first_name or '-'} (@{user.username or '-'})\n"
            f"▫️ ID: `{user.id}`\n▫️ Активна до: {expires_at}")
    try:
        await context.bot.send_message(chat_id=OWNER_ID, text=text, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Ошибка отправки уведомления владельцу: {e}")

# Улучшенная функция разбиения на чанки, не разрывающая HTML-теги
def split_into_chunks(text: str, limit: int = 4096) -> list:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind('</b>', 0, limit)
        if split_at == -1:
            split_at = text.rfind(' ', 0, limit)
        if split_at == -1:
            split_at = limit
        else:
            split_at += 4
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    thinking_message = None
    user = update.effective_user
    if user is None:
        return
    user_id = user.id
    user_message_text = sanitize_text(update.message.text)
    logger.info(f"Получен запрос от user_id {user_id}: {user_message_text}")

    if user_id == OWNER_ID or user_id in EXEMPT_USERS:
        get_or_create_user(user_id, user.username or "", user.first_name or "")
    elif not check_subscription(user_id):
        keyboard = [[InlineKeyboardButton("ДОСТУП", callback_data="subscription")]]
        await update.message.reply_text('🔐 У вас нет активной подписки', reply_markup=InlineKeyboardMarkup(keyboard))
        return

    thinking_message = await update.message.reply_text("▫️ Получаю запрос...")

    try:
        match = re.search(r'([A-Z0-9]{3,8})[\/\-]?([A-Z]{3,4})', user_message_text.upper())
        if match:
            raw_ticker = f"{match.group(1)}/{match.group(2)}"
            is_valid, normalized_ticker = validate_ticker(raw_ticker)
            if not is_valid:
                await context.bot.edit_message_text("▫️ Некорректный тикер. Укажите пару вида BTC/USDT.", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)
                return
            ticker = normalized_ticker
        else:
            ticker = None

        last_pair = get_last_pair(user_id)

        if ticker:
            if ticker != last_pair:
                clear_pair_history(user_id, ticker)
            logger.info(f"🔎 Анализ пары {ticker}")

            if user_id != OWNER_ID and user_id not in EXEMPT_USERS:
                if not check_daily_limit(user_id):
                    await context.bot.edit_message_text("▫️ Вы использовали лимит дневных запросов. Следующие доступны через 24 часа.", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)
                    return
                increment_daily_count(user_id)

            await context.bot.edit_message_text(f"▫️ Сбор данных для {ticker}...", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)

            results = await asyncio.gather(get_combined_analysis(ticker), get_fng_index())
            market_data, fng_data = results

            if not market_data or market_data.get("error"):
                error_text = market_data.get("error") if market_data else "Ошибка"
                await context.bot.edit_message_text(f"Ошибка: `{error_text}`", chat_id=update.effective_chat.id, message_id=thinking_message.message_id, parse_mode=ParseMode.MARKDOWN)
                return

            signals: MultiSignal = market_data["signals"]
            price = market_data["price"]
            daily_trend = market_data.get("daily_trend", "sideways")
            support_1d = market_data.get("support_1d", "N/A")
            resistance_1d = market_data.get("resistance_1d", "N/A")
            trend_emoji = "📈" if daily_trend == "bullish" else "📉" if daily_trend == "bearish" else "➡️"
            fng_value = fng_data.get("value") if fng_data else "—"
            fng_sentiment = fng_data.get("sentiment") if fng_data else "—"

            def format_signal_for_prompt(s: TradeSetup, label: str, include_scale_out: bool = False) -> str:
                if s.direction == "NO_TRADE":
                    return f"{label}: Нет сделки"
                emoji = "🟢" if s.direction == "LONG" else "🔴"
                entry_str = f"{s.entry:.2f}" if s.entry else "—"
                stop_str = f"{s.stop:.2f}" if s.stop else "—"
                tp1_str = f"{s.tp1:.2f}" if s.tp1 else "—"
                tp2_str = f"{s.tp2:.2f}" if s.tp2 else "—"
                stop_loss_percent = f"{abs((s.stop - s.entry) / s.entry * 100):.2f}" if s.entry and s.stop and s.entry != 0 else "—"
                take_profit_percent = f"{abs((s.tp1 - s.entry) / s.entry * 100):.2f}" if s.entry and s.tp1 and s.entry != 0 else "—"
                risk_reward = f"{abs((s.tp1 - s.entry) / (s.entry - s.stop)):.2f}" if s.entry and s.stop and s.tp1 and s.entry != s.stop else "—"
                risk_reward_tp2 = f"{abs((s.tp2 - s.entry) / (s.entry - s.stop)):.2f}" if s.entry and s.stop and s.tp2 and s.entry != s.stop else "—"
                base = f"""{label} {emoji} {s.direction}
• Точка входа: {entry_str}
• Стоп-лосс: {stop_str} ({stop_loss_percent}%)
• Тейк-профит: TP1: {tp1_str} (+{take_profit_percent}%) / TP2: {tp2_str}
• Риск/прибыль: TP1 1:{risk_reward} | TP2 1:{risk_reward_tp2}
Уверенность: {s.confidence*100:.0f}%"""
                if include_scale_out:
                    base += f"""
Тактика фиксации (Scale-Out):
• TP1 ({s.tp1_percent}% объёма): {tp1_str}
• TP2 ({s.tp2_percent}% объёма): {tp2_str}
• Runners ({s.runner_percent}% объёма): {s.hold_note}"""
                return base

            main_signal_str = format_signal_for_prompt(signals.swing, "Основное направление", include_scale_out=True)
            counter_str = ""
            if signals.counter_trend and signals.counter_trend.direction != "NO_TRADE":
                counter_str = format_signal_for_prompt(signals.counter_trend, "Контртренд направление", include_scale_out=False)

            # Сохраняем сигнал в БД, если это LONG/SHORT (основной или контртрендовый)
            if signals.swing.direction in ["LONG", "SHORT"]:
                save_signal_outcome(user_id=user_id, symbol=ticker, direction=signals.swing.direction, entry=signals.swing.entry, stop=signals.swing.stop, tp1=signals.swing.tp1, confidence=signals.swing.confidence)
            if signals.counter_trend and signals.counter_trend.direction in ["LONG", "SHORT"]:
                save_signal_outcome(user_id=user_id, symbol=ticker, direction=signals.counter_trend.direction, entry=signals.counter_trend.entry, stop=signals.counter_trend.stop, tp1=signals.counter_trend.tp1, confidence=signals.counter_trend.confidence)

            # Формируем промпт для модели
            rsi_1h_val = market_data.get("rsi_1h", 50.0)
            rsi_4h_val = market_data.get("rsi_4h", 50.0)
            atr_4h_val = market_data.get("atr_4h", 0.0)
            atr_4h_pct = (atr_4h_val / price * 100) if price else 0.0
            macd_4h_val = market_data.get("macd_4h", 0.0)
            macd_signal_4h_val = market_data.get("macd_signal_4h", 0.0)
            macd_status = "бычий (MACD выше сигнальной линии)" if macd_4h_val > macd_signal_4h_val else "медвежий (MACD ниже сигнальной линии)"
            ema50_4h_val = market_data.get("ema50_4h", 0.0)
            ema200_4h_val = market_data.get("ema200_4h", 0.0)
            if ema50_4h_val and ema200_4h_val:
                if price > ema50_4h_val > ema200_4h_val:
                    ema_status = f"цена выше EMA50 ({ema50_4h_val:.2f}) выше EMA200 ({ema200_4h_val:.2f}) — сильный бычий контекст"
                elif price < ema50_4h_val < ema200_4h_val:
                    ema_status = f"цена ниже EMA50 ({ema50_4h_val:.2f}) ниже EMA200 ({ema200_4h_val:.2f}) — сильный медвежий контекст"
                else:
                    ema_status = f"EMA50 ({ema50_4h_val:.2f}) / EMA200 ({ema200_4h_val:.2f}) без чёткого выравнивания"
            else:
                ema_status = "—"
            oi_val = market_data.get("oi", 0.0)
            funding_val = market_data.get("funding", 0.0)
            btc_dom_val = market_data.get("btc_dominance", 0.0)
            rel_vol_val = market_data.get("relative_volume_1h", 1.0)
            fib = market_data.get("fib_levels") or {}
            if fib:
                fib_str = (f"0: {fib.get('fib_0','—')} | 0.236: {fib.get('fib_236','—')} | 0.382: {fib.get('fib_382','—')} | "
                           f"0.5: {fib.get('fib_500','—')} | 0.618: {fib.get('fib_618','—')} | 0.786: {fib.get('fib_786','—')} | 1.0: {fib.get('fib_100','—')}")
            else:
                fib_str = "—"
            prompt = f"""
Pair: {ticker}
Date: {datetime.utcnow().strftime("%Y-%m-%d")}
Price: {price}
Market Regime: {signals.regime}
Global Trend: {daily_trend} {trend_emoji}
Support (1D): {support_1d}
Resistance (1D): {resistance_1d}
FNG Value: {fng_value} ({fng_sentiment})
RSI (1H/4H): {rsi_1h_val:.0f} / {rsi_4h_val:.0f}
MACD (4H): {macd_status}
ATR (4H): {atr_4h_val:.2f} ({atr_4h_pct:.1f}% от цены)
EMA Status (4H): {ema_status}
Открытый интерес (OI): {oi_val}
Funding Rate: {funding_val}
Доминация BTC: {btc_dom_val:.1f}%
Относительный объём (1H): {rel_vol_val:.2f}x от среднего
Фибоначчи (100 баров 1D): {fib_str}
{f"Сигнал активен уже {signals.signal_age_hours:.1f} ч." if signals.is_active_signal else ""}

{main_signal_str}

{counter_str}

{signals.context}
"""
            append_message(user_id, "user", prompt, pair=ticker)
            chat_context = [{"role": "user", "content": prompt}]

        else:
            # Уточняющий вопрос
            last_analysis = get_last_analysis(user_id)
            if not last_analysis:
                await context.bot.edit_message_text("Сначала напишите тикер вида: BTC/USDT", chat_id=update.effective_chat.id, message_id=thinking_message.message_id, parse_mode=ParseMode.MARKDOWN)
                return
            history = get_history(user_id, limit=20)
            guarded_question = f"""
Ответь на уточняющий вопрос по паре {last_pair or 'текущей'}, используя ТОЛЬКО данные из истории диалога выше (включая последнюю торговую рекомендацию).
НЕ придумывай новые цифры, уровни или индикаторы, которых нет в истории.
НЕ упоминай значения индикаторов (ADX, RSI, MACD и т.п.) и конкретные числа, если их явно не было в истории.
Если информации недостаточно — так и скажи.
Вопрос пользователя:
{user_message_text}
Ответь кратко (5-7 предложений) и по существу.
"""
            chat_context = history + [{"role": "user", "content": guarded_question}]

        await context.bot.edit_message_text("▫️ Проводится анализ...", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)

        bot_response_text = await generate_with_routing(chat_context=chat_context, user_id=user_id, is_premium=(user_id == OWNER_ID or user_id in EXEMPT_USERS or check_subscription(user_id)))

        if not bot_response_text:
            await context.bot.edit_message_text("Превышена нагрузка на API. Повторите через 30 секунд.", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)
            return

        append_message(user_id, "model", bot_response_text, pair=ticker or last_pair)
        if thinking_message:
            try: await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=thinking_message.message_id)
            except BadRequest: pass

        for chunk in split_into_chunks(bot_response_text):
            try:
                await update.message.reply_text(chunk, parse_mode="HTML")
            except BadRequest as e:
                logger.warning(f"HTML parse_mode отклонён Telegram ({e}), отправляю как обычный текст")
                await update.message.reply_text(chunk)

        logger.info(f"[{user_id}] Ответ успешно отправлен.")

    except httpx.TimeoutException:
        if thinking_message:
            try: await context.bot.edit_message_text("▫️ Сервис временно недоступен. Попробуйте через несколько секунд.", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)
            except: pass
        else: await update.message.reply_text("▫️ Сервис временно недоступен. Попробуйте через несколько секунд.")
    except Exception as e:
        logger.error(f"[{user_id}] ❌ Ошибка: {e}", exc_info=True)
        if thinking_message:
            try: await context.bot.edit_message_text("▫️ Произошла временная ошибка. Попробуйте снова через 5–10 секунд.", chat_id=update.effective_chat.id, message_id=thinking_message.message_id)
            except: pass
        else: await update.message.reply_text("▫️ Произошла временная ошибка. Попробуйте снова через 5–10 секунд.")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except BadRequest as e:
        if "Query is too old" in str(e):
            return
        raise
    data = query.data
    chat_id = query.message.chat.id
    user_id = query.from_user.id

    if data == "subscription":
        keyboard = [[InlineKeyboardButton(f"Telegram Stars - {SUB_PRICE_STARS}", callback_data="pay_stars"), InlineKeyboardButton(f"Crypto - ${SUB_PRICE_USDT}", callback_data="pay_crypto")]]
        await context.bot.send_message(chat_id, "<b>Оплата подписки</b>\n1 месяц (30 дней) ▫️ Выберите способ оплаты", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "pay_stars":
        if check_subscription(user_id):
            await context.bot.send_message(chat_id, "▫️ У вас уже есть активная подписка.")
            return
        await context.bot.send_invoice(chat_id=chat_id, title="Подписка на 1 месяц", description="ARIXEN – аналитическая платформа", payload=f"stars_30d:{user_id}", currency="XTR", prices=[LabeledPrice("Подписка на 1 месяц", SUB_PRICE_STARS)], provider_token="")
    elif data == "pay_crypto":
        get_or_create_user(user_id, query.from_user.username or "", query.from_user.first_name or "")
        invoice = create_crypto_invoice(user_id=user_id, amount=SUB_PRICE_USDT)
        create_invoice_record(invoice_id=invoice["invoice_id"], user_id=user_id, amount=SUB_PRICE_USDT, username=query.from_user.username or "", first_name=query.from_user.first_name or "")
        keyboard = [[InlineKeyboardButton("Оплатить", url=invoice["pay_url"])], [InlineKeyboardButton("Проверить оплату", callback_data=f"check_{invoice['invoice_id']}")]]
        await context.bot.send_message(chat_id, f"Подписка на 1 месяц. Сумма: <b>${SUB_PRICE_USDT}</b>\nПосле оплаты нажмите <b>Проверить оплату</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(keyboard))
    elif data.startswith("check_"):
        invoice_id = int(data.split("_")[1])
        invoice = get_invoice(invoice_id)
        if invoice and invoice["status"] == "paid":
            if not is_invoice_paid(invoice_id):
                mark_invoice_paid(invoice_id)
                activate_subscription(user_id=update.effective_user.id, username=update.effective_user.username or "", first_name=update.effective_user.first_name or "", payment_method="crypto", amount=SUB_PRICE_USDT, days=SUB_DAYS)
                expires_at = (datetime.utcnow() + timedelta(days=SUB_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
                await notify_admins_new_subscription(context, update.effective_user, expires_at)
                await context.bot.send_message(chat_id, "<b>Оплата подтверждена.</b>\nПодписка активирована", parse_mode="HTML")
            else:
                await context.bot.send_message(chat_id, "Подписка уже была активирована ранее.", parse_mode="HTML")
        else:
            await context.bot.send_message(chat_id, "▫️ Оплата не найдена.\nЕсли оплатили — подождите 5 минут.")
    elif data == "mains":
        text = """▫️ <b>Инструкция</b>
▫️ <b>Запуск анализа</b> 
Напишите пару: BTC/USDT или любую другую. 
Система обработает данные и отправит развернутый отчет.

▫️ <b>Уточняющие вопросы</b>
После получения отчета можете задать уточняющий вопрос. 
<i>Например: «Почему Long?» или «Какие риски для Short?».</i> 
Пишите вопросы по теме текущего анализа. 
Не пишите вопросы не относящиеся к теме.
Система обрабатывает только текстовые запросы и рыночные тикеры. 
Не отправляйте изображения и ссылки. 

▫️ <b>История и память</b>
Система помнит и хранит историю сессии.
<i>Сохраняйте историю в чате.</i>

▫️ <b>Доступ к анализу по подписке</b> 
Аналитика доступна по активной подписке.. 
Оплата через Telegram Stars или Crypto Bot.
При возникновении проблем пишите в Поддержку.

▫️ <b>Поддержка</b>
Поддержка обрабатывает запросы, связанные исключительно с работой инфраструктуры и транзакциями. 
<i>В Поддержку обращаться только по техническим вопросам.</i>

"""
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_start")]]
        await query.edit_message_text(text, parse_mode="HTML", disable_web_page_preview=True, reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "metod":
        text = """▫️ <b>Логика анализа</b> 
▫ Мощная аналитическая сиcтема развернутая на собственной инфраструктуре.
Предоставляет структурированный, многофакторный анализ на реальных данных в текущем времени. 
Экономит часы рутинной работы давая готовые отчеты для принятия решений. 

▫️ <b>Мультитаймфреймовый анализ</b>
Анализирует положение цены по множеству параметров и оценивает силу тренда на каждом таймфрейме.
Старшие таймфреймы задают глобальный тренд и ключевые уровни. Младшие - точные зоны входа.
Алгоритмическая обработка массивов данных. Выявление реальных зон интереса крупных игроков.
Нейросетевая модель агрегирует и обрабатывает массивы данных в отчет-рекомендацию. 

▫️ Работа строго по подтвержденному тренду. Стратегии Интрадей и Свинг-трейдинг.
При отсутствии условий или в фазе неопределенности (флэт) - система выдает статус «Нет сделки»

▫️ <b>Формирование отчета</b>
Процесс полностью автоматизирован и базируется исключительно на реальных рыночных данных.
Все сигналы консолидируются в комплексную оценку. 
На её основе рассчитываются конкретные уровни входа, стоп-лосса и тейк-профита, соотношение риск/прибыль.   
Уверенность в рекомендации выражается в процентах и определяется долей совпадающих факторов.
Вы получаете готовый план для открытия позиции без необходимости сводить графики вручную.

▫️ <b>Контекстный диалог</b>
Система помнит и хранит историю сессии. Отвечает на уточняющие вопросы.
Выступает в роли ассистента, с которым можно вести полноценный диалог.
"""
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_start")]]
        await query.edit_message_text(text, parse_mode="HTML", disable_web_page_preview=True, reply_markup=InlineKeyboardMarkup(keyboard))
    elif data == "faq":
        if not FAQ_DATA:
            load_faq()
        if not FAQ_DATA:
            await query.edit_message_text("❌ Список вопросов временно недоступен. Попробуйте позже.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Назад", callback_data="back_to_start")]]))
            return
        keyboard = []
        for qid, qinfo in FAQ_DATA.items():
            keyboard.append([InlineKeyboardButton(qinfo["title"], callback_data=f"faq_{qid}")])
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_start")])
        await query.edit_message_text(
            "▫️ **Часто задаваемые вопросы**:",
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )        
    
    elif data.startswith("faq_"):
        qid = data[4:]  # убираем "faq_"
        if not FAQ_DATA:
            load_faq()
        qinfo = FAQ_DATA.get(qid)
        if qinfo:
            file_path = os.path.join(FAQ_DIR, qinfo["file"])
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    answer = f.read()
            except Exception as e:
                logger.error(f"Error reading FAQ file {file_path}: {e}")
                answer = f"❌ Ошибка загрузки ответа. Пожалуйста, сообщите в поддержку."
            keyboard = [[InlineKeyboardButton("🔙 К вопросам", callback_data="faq")]]
            await query.edit_message_text(
                answer,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await query.answer("Вопрос не найден", show_alert=True)

    elif data == "back_to_start":
        user = update.effective_user
        text = f"""Здравствуйте, {user.mention_html()}
<b>ARIXEN</b> - платформа профессионального анализа в криптотрейдинге.
По вашему запросу проводит многоуровневый анализ рыночных данных.
▫️ Анализ ликвидности: Поиск ключевых уровней на основе биржевых стаканов и объемов.
Алгоритмическая обработка потоков данных для выявления реальных зон интереса крупных игроков.
▫️ Нейросетевая модель агрегирует и обрабатывает массивы данных в отчет-рекомендацию.
Вы получаете готовый план для открытия позиции.

▫️ Для запуска анализа: 
• Отправьте запрос по паре BTC/USDT или другой.
• Получите готовый отчет для открытия позиции.

🔐 <i>Доступ предоставляется по подписке. 
▫️ Процесс работы - в разделе «ОТЧЕТЫ».</i>"""
        keyboard = [
            [InlineKeyboardButton("ИНСТРУКЦИЯ", callback_data="mains"), InlineKeyboardButton("ДОСТУП", callback_data="subscription")], 
            [InlineKeyboardButton("ЛОГИКА АНАЛИЗА", callback_data="metod"), InlineKeyboardButton("ОТЧЕТЫ", url="https://t.me/arixen_official")],
            [InlineKeyboardButton("ВОПРОСЫ", callback_data="faq"), InlineKeyboardButton("ПОДДЕРЖКА", url="https://t.me/ruby_tie?direct")],
        ]
        await query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode="HTML", disable_web_page_preview=True)

async def successful_payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    payment = update.message.successful_payment
    user_id = update.effective_user.id
    if payment.currency != "XTR" or not payment.invoice_payload.startswith("stars_30d:") or payment.total_amount != SUB_PRICE_STARS:
        return
    try:
        payload_user_id = int(payment.invoice_payload.split(":")[1])
    except (ValueError, IndexError):
        return
    if payload_user_id != user_id or check_subscription(user_id):
        return
    activate_subscription(user_id=user_id, username=update.effective_user.username or "", first_name=update.effective_user.first_name or "", payment_method="stars", amount=SUB_PRICE_STARS, days=SUB_DAYS)
    expires_at = (datetime.utcnow() + timedelta(days=SUB_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    await notify_admins_new_subscription(context, update.effective_user, expires_at)
    await update.message.reply_text("<b>Оплата Telegram Stars подтверждена!</b>\nДоступ активирован", parse_mode="HTML")

async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.pre_checkout_query.answer(ok=True)

POPULAR_SYMBOLS = ["BTC/USDT", "ETH/USDT"]

async def popular_pairs_prewarmer(interval_seconds: int = 45):
    logger.info("🔄 Прогрев кэша для популярных пар запущен")
    while True:
        for symbol in POPULAR_SYMBOLS:
            try:
                await get_spot_data(symbol)
                await get_futures_data(symbol)
            except Exception as e:
                logger.warning(f"Прогрев кэша не удался для {symbol}: {e}")
            await asyncio.sleep(2)
        await asyncio.sleep(interval_seconds)

def _signal_to_db_dict(signal: Signal) -> dict:
    return {
        "symbol": signal.symbol, "direction": signal.direction, "entry": signal.entry,
        "stop": signal.stop, "tp1": signal.tp1, "tp2": signal.tp2, "confidence": signal.confidence,
        "reason": signal.reason, "timeframe": signal.timeframe, "pd_zone": signal.pd_zone,
        "sweep_detected": int(signal.sweep_detected), "created_at": signal.created_at,
        "updated_at": signal.updated_at, "status": signal.status,
        "entry_support": signal.entry_support, "entry_resistance": signal.entry_resistance,
        "distance_limit": signal.distance_limit, "tp1_reached_at": signal.tp1_reached_at,
        "entry_type": signal.entry_type, "is_counter_trend": int(signal.is_counter_trend),
    }

def _persist_signal_set(signal: Signal) -> None:
    try:
        save_active_signal(_signal_to_db_dict(signal))
    except Exception as e:
        logger.error(f"Не удалось сохранить активный сигнал {signal.symbol} в БД: {e}")

def _persist_signal_delete(symbol: str) -> None:
    try:
        delete_active_signal(symbol)
    except Exception as e:
        logger.error(f"Не удалось удалить активный сигнал {symbol} из БД: {e}")

def _load_active_signals_from_db() -> None:
    restored = []
    try:
        rows = get_all_active_signals()
    except Exception as e:
        logger.error(f"Не удалось прочитать active_signals из БД: {e}")
        rows = []
    for row in rows:
        try:
            restored.append(Signal(
                symbol=row["symbol"], direction=row["direction"], entry=row["entry"],
                stop=row["stop"], tp1=row["tp1"], tp2=row["tp2"], confidence=row["confidence"],
                reason=row["reason"] or "", timeframe=row["timeframe"] or "", pd_zone=row["pd_zone"] or "",
                sweep_detected=bool(row["sweep_detected"]), created_at=row["created_at"] or 0.0,
                updated_at=row["updated_at"] or 0.0, status=row["status"] or "ACTIVE",
                entry_support=row["entry_support"], entry_resistance=row["entry_resistance"],
                distance_limit=row["distance_limit"] or 0.0, tp1_reached_at=row["tp1_reached_at"],
                entry_type=row["entry_type"] or "market", is_counter_trend=bool(row["is_counter_trend"]),
            ))
        except Exception as e:
            logger.error(f"Не удалось восстановить активный сигнал {row.get('symbol')}: {e}")
    signal_storage.load_from(restored)
    signal_storage.set_persistence_hooks(on_set=_persist_signal_set, on_delete=_persist_signal_delete)
    logger.info(f"♻️ Восстановлено {len(restored)} активных сигналов из БД")

async def post_init(application: Application) -> None:
    try:
        await application.bot.delete_webhook(drop_pending_updates=True)
        await asyncio.sleep(5)
    except Exception as e:
        logger.warning(f"Failed to clear webhook: {e}")
    commands = [BotCommand("start", "Старт")]
    await application.bot.set_my_commands(commands)
    _load_active_signals_from_db()
    application.bot_data["cache_task"] = asyncio.create_task(cache_cleanup_task(300))
    application.bot_data["prewarm_task"] = asyncio.create_task(popular_pairs_prewarmer())

async def post_shutdown(application: Application):
    task = application.bot_data.get("cache_task")
    if task:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
    prewarm_task = application.bot_data.get("prewarm_task")
    if prewarm_task:
        prewarm_task.cancel()
        try: await prewarm_task
        except asyncio.CancelledError: pass
    # Закрываем fallback HTTP-клиент
    await close_fallback_http()
    await http_client.aclose()

def main() -> None:
    init_db()
    load_faq()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).connect_timeout(120).read_timeout(120).post_init(post_init).post_shutdown(post_shutdown).build()
    init_error_reporter(admin_ids={OWNER_ID})
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("users", users_command))
    application.add_handler(CommandHandler("inactive", inactive_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment))
    application.add_error_handler(global_error_handler)
    print("Бот запущен...")
    application.run_polling(drop_pending_updates=True)
    print("Бот остановлен.")

if __name__ == '__main__':
    main()