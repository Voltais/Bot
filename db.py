import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

DB_PATH = "/opt/tradercryptoai-bot/messages.db"
DAILY_LIMIT = 10

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Пользователи (только подписчики)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                created_at TEXT
            )
        """)

        # Подписки
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                user_id INTEGER PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                payment_method TEXT,
                amount REAL,
                paid_at TEXT,
                expires_at TEXT,
                status TEXT CHECK(status IN ('active','expired'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sub_expires ON subscriptions(expires_at)")

        # Инвойсы
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS invoices (
                invoice_id INTEGER PRIMARY KEY,
                user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                amount REAL,
                status TEXT CHECK(status IN ('pending','paid')),
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inv_user ON invoices(user_id)")

        # Сообщения (история диалогов)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                role TEXT CHECK(role IN ('user','model')),
                content TEXT,
                pair TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_msg_timestamp ON messages(timestamp)")

        # Дневные лимиты запросов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_requests (
                user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                date TEXT,
                count INTEGER DEFAULT 0,
                PRIMARY KEY (user_id, date)
            )
        """)

        # ========== НОВАЯ ТАБЛИЦА ДЛЯ АДАПТИВНОГО ОБУЧЕНИЯ ==========
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                stop_price REAL,
                tp1_price REAL,
                confidence REAL,
                created_at TEXT,
                outcome TEXT CHECK(outcome IN ('win', 'loss', 'pending')),
                actual_profit REAL,
                resolved_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_user ON signal_outcomes(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_created ON signal_outcomes(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sig_outcome ON signal_outcomes(outcome)")

        # ========== АКТИВНЫЕ СИГНАЛЫ ДВИЖКА (персистентность SignalStorage) ==========
        # Общее состояние движка по паре (не привязано к user_id) — раньше жило только
        # в памяти процесса и терялось при каждом рестарте/деплое бота.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS active_signals (
                symbol TEXT PRIMARY KEY,
                direction TEXT,
                entry REAL,
                stop REAL,
                tp1 REAL,
                tp2 REAL,
                confidence REAL,
                reason TEXT,
                timeframe TEXT,
                pd_zone TEXT,
                sweep_detected INTEGER,
                created_at REAL,
                updated_at REAL,
                status TEXT,
                entry_support REAL,
                entry_resistance REAL,
                distance_limit REAL,
                tp1_reached_at REAL,
                entry_type TEXT,
                is_counter_trend INTEGER
            )
        """)

        conn.commit()
    finally:
        conn.close()

# =====================================================
# АКТИВНЫЕ СИГНАЛЫ ДВИЖКА (персистентность SignalStorage)
# =====================================================

def save_active_signal(s: dict) -> None:
    """Сохраняет/обновляет активный сигнал движка (upsert по symbol)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO active_signals
            (symbol, direction, entry, stop, tp1, tp2, confidence, reason, timeframe,
             pd_zone, sweep_detected, created_at, updated_at, status, entry_support,
             entry_resistance, distance_limit, tp1_reached_at, entry_type, is_counter_trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                direction=excluded.direction, entry=excluded.entry, stop=excluded.stop,
                tp1=excluded.tp1, tp2=excluded.tp2, confidence=excluded.confidence,
                reason=excluded.reason, timeframe=excluded.timeframe, pd_zone=excluded.pd_zone,
                sweep_detected=excluded.sweep_detected, updated_at=excluded.updated_at,
                status=excluded.status, entry_support=excluded.entry_support,
                entry_resistance=excluded.entry_resistance, distance_limit=excluded.distance_limit,
                tp1_reached_at=excluded.tp1_reached_at, entry_type=excluded.entry_type,
                is_counter_trend=excluded.is_counter_trend
        """, (
            s["symbol"], s["direction"], s["entry"], s["stop"], s["tp1"], s["tp2"],
            s["confidence"], s["reason"], s["timeframe"], s["pd_zone"], s["sweep_detected"],
            s["created_at"], s["updated_at"], s["status"], s["entry_support"],
            s["entry_resistance"], s["distance_limit"], s["tp1_reached_at"],
            s["entry_type"], s["is_counter_trend"],
        ))
        conn.commit()
    finally:
        conn.close()

def delete_active_signal(symbol: str) -> None:
    """Удаляет активный сигнал (инвалидация/истечение/TP2)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM active_signals WHERE symbol = ?", (symbol,))
        conn.commit()
    finally:
        conn.close()

def get_all_active_signals() -> List[Dict[str, Any]]:
    """Возвращает все сохранённые активные сигналы (для восстановления при старте бота)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM active_signals")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
def get_or_create_user(user_id: int, username: str, first_name: str) -> None:
    """
    Создаёт запись пользователя, если её нет, иначе обновляет username/first_name.
    Вызывается для КАЖДОГО пользователя при первом же обращении к боту
    (владелец, exempt-пользователи, подписчики) — до любой другой записи
    в БД, ссылающейся на users(user_id) по внешнему ключу.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (user_id, username, first_name, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name
        """, (user_id, username, first_name, datetime.utcnow().isoformat()))
        conn.commit()
    finally:
        conn.close()

# =====================================================
# ПОДПИСКИ
# =====================================================
def activate_subscription(
    user_id: int,
    username: str,
    first_name: str,
    payment_method: str,
    amount: float,
    days: int = 30
) -> None:
    """
    Единственная точка создания пользователя.
    Пользователь появляется в БД только после оплаты.
    """
    get_or_create_user(user_id, username, first_name)

    conn = get_connection()
    try:
        cursor = conn.cursor()
        now = datetime.utcnow()
        expires = now + timedelta(days=days)

        cursor.execute("""
            INSERT INTO subscriptions
            (user_id, payment_method, amount, paid_at, expires_at, status)
            VALUES (?, ?, ?, ?, ?, 'active')
            ON CONFLICT(user_id) DO UPDATE SET
                payment_method = excluded.payment_method,
                amount = excluded.amount,
                paid_at = excluded.paid_at,
                expires_at = excluded.expires_at,
                status = 'active'
        """, (user_id, payment_method, amount, now.isoformat(), expires.isoformat()))
        conn.commit()
    finally:
        conn.close()

def deactivate_expired() -> None:
    """Переводит в статус expired все подписки, у которых истёк срок."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE subscriptions
            SET status = 'expired'
            WHERE status = 'active' AND expires_at <= ?
        """, (datetime.utcnow().isoformat(),))
        conn.commit()
    finally:
        conn.close()

def check_subscription(user_id: int) -> bool:
    """Проверяет, активна ли подписка у пользователя."""
    deactivate_expired()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM subscriptions
            WHERE user_id = ? AND status = 'active' AND expires_at > ?
        """, (user_id, datetime.utcnow().isoformat()))
        return cursor.fetchone() is not None
    finally:
        conn.close()

def get_active_users_full() -> List[Dict[str, Any]]:
    """Возвращает список активных подписчиков с их данными (для команды /users)."""
    deactivate_expired()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.user_id, s.expires_at, s.status,
                   u.username, u.first_name
            FROM subscriptions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.status = 'active'
            ORDER BY s.expires_at DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_inactive_users() -> List[Dict[str, Any]]:
    """
    Возвращает неактивных пользователей (подписка истекла и не продлена),
    отсортированных от самых давних к недавним — удобно, чтобы решать,
    кого в первую очередь чистить (для команды /inactive).
    """
    deactivate_expired()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.user_id, s.expires_at, s.payment_method, s.amount,
                   u.username, u.first_name
            FROM subscriptions s
            JOIN users u ON s.user_id = u.user_id
            WHERE s.status = 'expired'
            ORDER BY s.expires_at ASC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_inactive_users_count() -> int:
    """Количество неактивных (истёкших) подписчиков."""
    deactivate_expired()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM subscriptions WHERE status = 'expired'")
        return cursor.fetchone()[0]
    finally:
        conn.close()

# =====================================================
# ИНВОЙСЫ (КРИПТО-ПЛАТЕЖИ)
# =====================================================
def create_invoice_record(
    invoice_id: int,
    user_id: int,
    amount: float,
    username: str,
    first_name: str
) -> None:
    """Сохраняет инвойс и создаёт пользователя заранее (чтобы FK не падал)."""
    get_or_create_user(user_id, username, first_name)
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO invoices (invoice_id, user_id, amount, status, created_at)
            VALUES (?, ?, ?, 'pending', ?)
        """, (invoice_id, user_id, amount, datetime.utcnow().isoformat()))
        conn.commit()
    finally:
        conn.close()

def mark_invoice_paid(invoice_id: int) -> None:
    """Помечает инвойс как оплаченный."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE invoices SET status = 'paid' WHERE invoice_id = ?", (invoice_id,))
        conn.commit()
    finally:
        conn.close()

def is_invoice_paid(invoice_id: int) -> bool:
    """Проверяет, оплачен ли инвойс."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM invoices WHERE invoice_id = ?", (invoice_id,))
        row = cursor.fetchone()
        return row and row["status"] == "paid"
    finally:
        conn.close()

def get_invoice_user(invoice_id: int) -> Optional[int]:
    """Возвращает user_id, связанный с инвойсом."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM invoices WHERE invoice_id = ?", (invoice_id,))
        row = cursor.fetchone()
        return row["user_id"] if row else None
    finally:
        conn.close()

# =====================================================
# ИСТОРИЯ СООБЩЕНИЙ (ТОЛЬКО ДЛЯ ПОДПИСЧИКОВ)
# =====================================================
def append_message(
    user_id: int,
    role: str,
    content: str,
    pair: Optional[str] = None
) -> None:
    """
    Добавляет сообщение в историю пользователя.
    НЕ создаёт пользователя – пользователь должен уже существовать (после оплаты).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (user_id, role, content, pair, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, role, content, pair, datetime.utcnow().isoformat()))

        # Удаляем старые сообщения, оставляя последние 50
        cursor.execute("""
            DELETE FROM messages
            WHERE user_id = ?
            AND id NOT IN (
                SELECT id FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT 50
            )
        """, (user_id, user_id))

        conn.commit()
    finally:
        conn.close()

def get_history(user_id: int, limit: int = 20) -> List[Dict[str, str]]:
    """Возвращает последние сообщения пользователя в хронологическом порядке."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT role, content FROM (
                SELECT id, role, content FROM messages
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) recent
            ORDER BY id ASC
        """, (user_id, limit))
        rows = cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]
    finally:
        conn.close()

def clear_history(user_id: int) -> None:
    """Полностью удаляет историю сообщений пользователя."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()

def get_last_pair(user_id: int) -> Optional[str]:
    """Возвращает последнюю использованную пользователем пару (из сообщений с ролью user)."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT pair FROM messages
            WHERE user_id = ? AND role = 'user' AND pair IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        return row["pair"] if row else None
    finally:
        conn.close()

def get_last_analysis(user_id: int) -> Optional[str]:
    """Возвращает последний ответ модели (торговую рекомендацию) для пользователя."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT content FROM messages
            WHERE user_id = ? AND role = 'model'
            ORDER BY id DESC
            LIMIT 1
        """, (user_id,))
        row = cursor.fetchone()
        return row["content"] if row else None
    finally:
        conn.close()

# =====================================================
# ДНЕВНЫЕ ЛИМИТЫ ЗАПРОСОВ
# =====================================================
def get_daily_requests_count(user_id: int) -> int:
    """Возвращает количество запросов пользователя за текущий день."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT count FROM daily_requests
            WHERE user_id = ? AND date = ?
        """, (user_id, today))
        row = cursor.fetchone()
        return row["count"] if row else 0
    finally:
        conn.close()

def increment_daily_count(user_id: int) -> None:
    """Увеличивает счётчик запросов пользователя на 1."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO daily_requests (user_id, date, count)
            VALUES (?, ?, 1)
            ON CONFLICT(user_id, date)
            DO UPDATE SET count = count + 1
        """, (user_id, today))
        conn.commit()
    finally:
        conn.close()

def check_daily_limit(user_id: int) -> bool:
    """Проверяет, не превышен ли дневной лимит."""
    return get_daily_requests_count(user_id) < DAILY_LIMIT

# =====================================================
# СТАТИСТИКА (ТОЛЬКО ПОДПИСЧИКИ)
# =====================================================
def get_stats() -> Dict[str, int]:
    """Возвращает статистику подписок (количество подписчиков)."""
    deactivate_expired()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM subscriptions")
        total_subs = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM subscriptions WHERE status = 'active'")
        active = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM subscriptions WHERE status = 'expired'")
        expired = cursor.fetchone()[0]
        return {
            "total_users": total_subs,
            "active_subs": active,
            "expired_subs": expired
        }
    finally:
        conn.close()

# =====================================================
# АДАПТИВНОЕ ОБУЧЕНИЕ (СИГНАЛЫ)
# =====================================================
def save_signal_outcome(
    user_id: int,
    symbol: str,
    direction: str,
    entry: float,
    stop: float,
    tp1: float,
    confidence: float
) -> int:
    """Сохраняет информацию о сгенерированном сигнале со статусом 'pending'."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signal_outcomes
            (user_id, symbol, direction, entry_price, stop_price, tp1_price, confidence, created_at, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, symbol, direction, entry, stop, tp1, confidence, datetime.utcnow().isoformat(), 'pending'))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def update_signal_outcome(signal_id: int, outcome: str, profit: float) -> None:
    """Обновляет исход сигнала (win/loss) и фактическую прибыль."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE signal_outcomes
            SET outcome = ?, actual_profit = ?, resolved_at = ?
            WHERE id = ?
        """, (outcome, profit, datetime.utcnow().isoformat(), signal_id))
        conn.commit()
    finally:
        conn.close()

def get_signal_stats(user_id: Optional[int] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Возвращает статистику по сигналам:
    - общее количество,
    - количество побед,
    - winrate,
    - средняя прибыль,
    - и (опционально) список последних сигналов с username.
    Если user_id указан – статистика по пользователю, иначе по всем.
    """
    deactivate_expired()  # необязательно, но для актуальности подписок
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Строим базовый запрос с JOIN на users для получения username
        query = """
            SELECT s.*, u.username, u.first_name
            FROM signal_outcomes s
            JOIN users u ON s.user_id = u.user_id
            WHERE 1=1
        """
        params = []
        if user_id is not None:
            query += " AND s.user_id = ?"
            params.append(user_id)
        if symbol is not None:
            query += " AND s.symbol = ?"
            params.append(symbol)

        # Статистика по исходам
        cursor.execute(f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                   AVG(CASE WHEN outcome = 'win' THEN actual_profit ELSE NULL END) as avg_win_profit,
                   AVG(CASE WHEN outcome = 'loss' THEN actual_profit ELSE NULL END) as avg_loss_profit
            FROM ({query}) t
        """, params)
        row = cursor.fetchone()
        total = row['total'] or 0
        wins = row['wins'] or 0
        losses = row['losses'] or 0
        winrate = (wins / total * 100) if total > 0 else 0

        # Список последних 10 сигналов (для отображения)
        cursor.execute(f"{query} ORDER BY s.created_at DESC LIMIT 10", params)
        recent = [dict(r) for r in cursor.fetchall()]

        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "winrate": round(winrate, 2),
            "avg_win_profit": row['avg_win_profit'] if row['avg_win_profit'] is not None else 0,
            "avg_loss_profit": row['avg_loss_profit'] if row['avg_loss_profit'] is not None else 0,
            "recent": recent
        }
    finally:
        conn.close()

def clear_pair_history(user_id: int, pair: str) -> None:
    """Удаляет сообщения пользователя только для указанной пары."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE user_id = ? AND pair = ?",
            (user_id, pair)
        )
        conn.commit()
    finally:
        conn.close()