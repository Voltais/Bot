import re
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    pass

def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
    """
    Валидация тикера криптовалютной пары.
    Принимает форматы: BTC/USDT, BTC-USDT, C98/USDT, 1000PEPE/USDT, KAS/USDT и т.д.
    """
    if not ticker or not isinstance(ticker, str):
        return False, None

    ticker = ticker.strip().upper()

    # Расширенный regex: 3-12 символов для базовой валюты, 3-5 для котируемой.
    # Это позволяет работать с длинными тикерами (например, 1000PEPE, 1INCH, C98).
    match = re.match(r'^([A-Z0-9]{3,12})[\/\-]([A-Z]{3,5})$', ticker)
    if match:
        base, quote = match.groups()
        normalized = f"{base}/{quote}"
        logger.info(f"Ticker validated: {ticker} -> {normalized}")
        return True, normalized

    logger.warning(f"Invalid ticker format: {ticker}. Use format like BTC/USDT or BTC-USDT.")
    return False, None


def validate_user_input(text: str) -> Tuple[bool, Optional[str]]:
    """
    Валидация пользовательского ввода
    """
    if not text or not isinstance(text, str):
        return False, "Пустое сообщение"

    if len(text) > 1000:
        return False, "Сообщение слишком длинное (максимум 1000 символов)"

    suspicious_patterns = [
        r'<script',
        r'javascript:',
        r'data:',
        r'vbscript:',
        r'onload=',
        r'onerror=',
        r'onclick='
    ]

    text_lower = text.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, text_lower):
            return False, "Обнаружены подозрительные символы"

    return True, None


def validate_user_id(user_id: any) -> bool:
    """Валидация user_id"""
    try:
        uid = int(user_id)
        return 0 < uid < 10**12
    except (ValueError, TypeError):
        return False


def validate_amount(amount: any) -> Tuple[bool, Optional[float]]:
    """
    Валидация суммы платежа
    """
    try:
        amount_float = float(amount)
        if 0.01 <= amount_float <= 10000:
            return True, amount_float
        return False, None
    except (ValueError, TypeError):
        return False, None


def sanitize_text(text: str) -> str:
    """
    Очистка текста от потенциально опасных символов
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[<>"\']', '', text)

    if len(text) > 1000:
        text = text[:1000] + "..."

    return text.strip()


def validate_invoice_id(invoice_id: any) -> bool:
    """Валидация ID инвойса"""
    try:
        iid = int(invoice_id)
        return 0 < iid < 10**15
    except (ValueError, TypeError):
        return False