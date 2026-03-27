import hashlib
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


TRIAL_DAYS = 30


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _normalize_company(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _email_domain(email: str) -> str:
    parts = (email or "").split("@", 1)
    return parts[1].strip().lower() if len(parts) == 2 else ""


@contextmanager
def get_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    with get_db(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                company_name TEXT NOT NULL,
                company_name_normalized TEXT NOT NULL,
                company_domain TEXT NOT NULL,
                stripe_customer_id TEXT,
                stripe_subscription_id TEXT,
                subscription_status TEXT NOT NULL DEFAULT 'none',
                plan_code TEXT NOT NULL DEFAULT 'none',
                trial_started_at TEXT,
                trial_ends_at TEXT,
                trial_used INTEGER NOT NULL DEFAULT 0,
                is_admin INTEGER NOT NULL DEFAULT 0,
                is_banned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_company_domain ON users(company_domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_company_normalized ON users(company_name_normalized)")
        
        # Create user_logs table for saved digitized logs
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_logs (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                curve_count INTEGER NOT NULL DEFAULT 0,
                depth_start REAL,
                depth_end REAL,
                depth_unit TEXT,
                las_content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_logs_user_id ON user_logs(user_id)")

        # Create admin_settings table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT NOT NULL
            )
            """
        )
        # Insert default settings if they don't exist
        conn.execute("INSERT OR IGNORE INTO admin_settings (setting_key, setting_value) VALUES ('global_banner', '')")
        conn.execute("INSERT OR IGNORE INTO admin_settings (setting_key, setting_value) VALUES ('feature_flag_experimental_ai', '0')")

        # Remember-me tokens table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS remember_tokens (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_remember_tokens_user_id ON remember_tokens(user_id)")


REMEMBER_TOKEN_DAYS = 30


def create_remember_token(db_path: str, user_id: int) -> str:
    """Generate a secure token, store its hash in the DB, and return the raw token."""
    raw = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=REMEMBER_TOKEN_DAYS)).isoformat()
    with get_db(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO remember_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at),
        )
    return raw


def get_user_by_remember_token(db_path: str, raw_token: str) -> Optional[Dict[str, Any]]:
    """Validate a raw token and return the associated user, or None if invalid/expired."""
    if not raw_token:
        return None
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT user_id FROM remember_tokens WHERE token = ? AND expires_at > ?",
            (token_hash, now),
        ).fetchone()
        if not row:
            return None
        user_row = conn.execute("SELECT * FROM users WHERE id = ?", (row["user_id"],)).fetchone()
    return dict(user_row) if user_row else None


def delete_remember_token(db_path: str, raw_token: str) -> None:
    """Delete a token by its raw value."""
    if not raw_token:
        return
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM remember_tokens WHERE token = ?", (token_hash,))


def delete_remember_tokens_for_user(db_path: str, user_id: int) -> None:
    """Delete all remember-me tokens for a user (e.g. on password change or explicit logout-all)."""
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM remember_tokens WHERE user_id = ?", (user_id,))


def save_user_log(db_path: str, log_id: str, user_id: int, name: str, curve_count: int, depth_start: float, depth_end: float, depth_unit: str, las_content: str) -> None:
    now = _utc_now_iso()
    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_logs (
                id, user_id, name, curve_count, depth_start, depth_end, depth_unit, las_content, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                curve_count=excluded.curve_count,
                depth_start=excluded.depth_start,
                depth_end=excluded.depth_end,
                depth_unit=excluded.depth_unit,
                las_content=excluded.las_content,
                updated_at=excluded.updated_at
            """,
            (log_id, user_id, name, curve_count, depth_start, depth_end, depth_unit, las_content, now, now)
        )


def get_user_logs(db_path: str, user_id: int) -> List[Dict[str, Any]]:
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT id, name, curve_count, depth_start, depth_end, depth_unit, created_at, updated_at FROM user_logs WHERE user_id = ? ORDER BY created_at DESC", 
            (user_id,)
        ).fetchall()
    
    result = []
    for r in rows:
        d = dict(r)
        # Format date nicely
        if d.get('created_at'):
            try:
                dt = datetime.fromisoformat(d['created_at'])
                d['created_at_formatted'] = dt.strftime('%b %d, %Y')
            except Exception:
                d['created_at_formatted'] = d['created_at']
        result.append(d)
    return result


def get_user_log(db_path: str, log_id: str, user_id: int) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM user_logs WHERE id = ? AND user_id = ?", (log_id, user_id)).fetchone()
    return dict(row) if row else None


def create_user(
    db_path: str,
    *,
    email: str,
    password_hash: str,
    full_name: str,
    company_name: str,
) -> int:
    now = _utc_now_iso()
    normalized_company = _normalize_company(company_name)
    domain = _email_domain(email)
    with get_db(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO users (
                email, password_hash, full_name, company_name,
                company_name_normalized, company_domain, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email.strip().lower(),
                password_hash,
                full_name.strip(),
                company_name.strip(),
                normalized_company,
                domain,
                now,
                now,
            ),
        )
        return int(cur.lastrowid)


def get_user_by_email(db_path: str, email: str) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    return dict(row) if row else None


def get_user_by_id(db_path: str, user_id: int) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_all_users_for_admin(db_path: str) -> List[Dict[str, Any]]:
    with get_db(db_path) as conn:
        # Also compute total log size for each user
        rows = conn.execute("""
            SELECT u.id, u.email, u.full_name, u.company_name, u.is_admin, u.is_banned, u.subscription_status, u.plan_code, u.trial_ends_at, u.created_at,
                   COUNT(l.id) as log_count,
                   SUM(LENGTH(l.las_content)) as total_log_bytes
            FROM users u
            LEFT JOIN user_logs l ON u.id = l.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """).fetchall()
    return [dict(r) for r in rows]


def get_all_logs_for_admin(db_path: str) -> List[Dict[str, Any]]:
    with get_db(db_path) as conn:
        rows = conn.execute(
            """
            SELECT l.id, l.name, l.curve_count, l.created_at, LENGTH(l.las_content) as size_bytes, u.email as user_email
            FROM user_logs l
            JOIN users u ON l.user_id = u.id
            ORDER BY l.created_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]

def get_admin_stats(db_path: str) -> Dict[str, Any]:
    stats = {
        'mrr': 0,
        'active_paid_users': 0,
        'trial_users': 0,
        'logs_24h': 0,
        'logs_7d': 0,
        'total_curves': 0,
        'total_storage_bytes': 0,
        'most_active_users': []
    }
    
    with get_db(db_path) as conn:
        # MRR & Users
        users = conn.execute("SELECT subscription_status, plan_code FROM users WHERE is_banned = 0").fetchall()
        for u in users:
            status = u['subscription_status']
            plan = u['plan_code']
            if status == 'active':
                stats['active_paid_users'] += 1
                if plan == 'monthly_99':
                    stats['mrr'] += 99
                elif plan == 'annual_999':
                    stats['mrr'] += 83.25 # $999/12
            elif status == 'trialing':
                stats['trial_users'] += 1
                
        # Usage Metrics
        now_ts = datetime.now(timezone.utc).timestamp()
        
        logs = conn.execute("SELECT created_at, curve_count, LENGTH(las_content) as size_bytes, user_id FROM user_logs").fetchall()
        
        user_log_counts = {}
        for l in logs:
            stats['total_curves'] += (l['curve_count'] or 0)
            stats['total_storage_bytes'] += (l['size_bytes'] or 0)
            
            try:
                log_dt = datetime.fromisoformat(l['created_at'].replace('Z', '+00:00'))
                log_ts = log_dt.timestamp()
                diff_hours = (now_ts - log_ts) / 3600
                
                if diff_hours <= 24:
                    stats['logs_24h'] += 1
                if diff_hours <= 24 * 7:
                    stats['logs_7d'] += 1
            except Exception:
                pass
                
            uid = l['user_id']
            user_log_counts[uid] = user_log_counts.get(uid, 0) + 1
            
        # Top 5 most active users
        if user_log_counts:
            top_uids = sorted(user_log_counts.keys(), key=lambda x: user_log_counts[x], reverse=True)[:5]
            placeholders = ','.join(['?'] * len(top_uids))
            top_users = conn.execute(f"SELECT id, email, full_name FROM users WHERE id IN ({placeholders})", top_uids).fetchall()
            user_map = {u['id']: u for u in top_users}
            
            for uid in top_uids:
                if uid in user_map:
                    stats['most_active_users'].append({
                        'email': user_map[uid]['email'],
                        'name': user_map[uid]['full_name'],
                        'log_count': user_log_counts[uid]
                    })
                    
    return stats

def get_admin_settings(db_path: str) -> Dict[str, str]:
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT setting_key, setting_value FROM admin_settings").fetchall()
    return {r['setting_key']: r['setting_value'] for r in rows}

def update_admin_setting(db_path: str, key: str, value: str) -> None:
    with get_db(db_path) as conn:
        conn.execute("INSERT OR REPLACE INTO admin_settings (setting_key, setting_value) VALUES (?, ?)", (key, value))


def get_user_by_customer_id(db_path: str, stripe_customer_id: str) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE stripe_customer_id = ?", (stripe_customer_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_subscription_id(db_path: str, stripe_subscription_id: str) -> Optional[Dict[str, Any]]:
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE stripe_subscription_id = ?", (stripe_subscription_id,)).fetchone()
    return dict(row) if row else None


def admin_update_user_action(db_path: str, user_id: int, action: str) -> bool:
    with get_db(db_path) as conn:
        if action == 'ban':
            conn.execute("UPDATE users SET is_banned = 1 WHERE id = ?", (user_id,))
        elif action == 'unban':
            conn.execute("UPDATE users SET is_banned = 0 WHERE id = ?", (user_id,))
        elif action == 'extend_trial':
            # SQLite datetime modifier
            conn.execute("UPDATE users SET trial_ends_at = datetime(trial_ends_at, '+7 days') WHERE id = ?", (user_id,))
        elif action == 'make_lifetime':
            conn.execute("UPDATE users SET subscription_status = 'active', plan_code = 'lifetime_comped' WHERE id = ?", (user_id,))
        elif action == 'delete':
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.execute("DELETE FROM user_logs WHERE user_id = ?", (user_id,))
        return True


def update_user_fields(db_path: str, user_id: int, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _utc_now_iso()
    keys = list(fields.keys())
    assignments = ", ".join(f"{k} = ?" for k in keys)
    values = [fields[k] for k in keys]
    values.append(user_id)
    with get_db(db_path) as conn:
        conn.execute(f"UPDATE users SET {assignments} WHERE id = ?", values)


def trial_eligibility(db_path: str, user: Dict[str, Any]) -> Dict[str, Any]:
    # Rule: 1 trial per company/email/domain
    if int(user.get("trial_used") or 0) == 1:
        return {"eligible": False, "reason": "A trial has already been used for this account."}

    email = (user.get("email") or "").strip().lower()
    domain = (user.get("company_domain") or "").strip().lower()
    company = _normalize_company(user.get("company_name") or "")

    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT id, email, company_domain, company_name
            FROM users
            WHERE trial_used = 1
              AND id != ?
              AND (
                    email = ?
                 OR company_domain = ?
                 OR company_name_normalized = ?
              )
            LIMIT 1
            """,
            (user["id"], email, domain, company),
        ).fetchone()

    if row:
        return {
            "eligible": False,
            "reason": "A trial has already been used for this company or email domain.",
        }
    return {"eligible": True, "reason": "Eligible for trial"}


def mark_trial_started(db_path: str, user_id: int, trial_ends_at: Optional[str]) -> None:
    update_user_fields(
        db_path,
        user_id,
        trial_used=1,
        trial_started_at=_utc_now_iso(),
        trial_ends_at=trial_ends_at,
    )


def compute_trial_countdown(user: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ends_at = _parse_iso(user.get("trial_ends_at"))
    if not ends_at:
        return None

    now = datetime.now(timezone.utc)
    delta = ends_at - now
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return {"days": 0, "hours": 0, "expired": True}

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    return {"days": days, "hours": hours, "expired": False}


def plan_label(plan_code: str) -> str:
    return {
        "trial": "Free Trial",
        "monthly": "$99 / month",
        "annual": "$999 / year",
        "none": "No active plan",
    }.get(plan_code or "none", "No active plan")


def subscription_access_allowed(user: Dict[str, Any]) -> bool:
    status = (user.get("subscription_status") or "none").lower()
    return status in {"trialing", "active"}


def serialize_invoices(stripe_invoices: List[Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for inv in stripe_invoices:
        amount_paid = (inv.get("amount_paid") or 0) / 100.0
        created_ts = inv.get("created")
        created_at = datetime.fromtimestamp(created_ts, tz=timezone.utc).strftime("%Y-%m-%d") if created_ts else ""
        output.append(
            {
                "id": inv.get("id"),
                "number": inv.get("number") or inv.get("id"),
                "amount_paid": amount_paid,
                "currency": (inv.get("currency") or "usd").upper(),
                "status": inv.get("status") or "",
                "created_at": created_at,
                "hosted_invoice_url": inv.get("hosted_invoice_url"),
                "invoice_pdf": inv.get("invoice_pdf"),
            }
        )
    return output
