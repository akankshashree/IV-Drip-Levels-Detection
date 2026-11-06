from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import sqlite3

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import hashlib
import hmac
import secrets
from pydantic import BaseModel, EmailStr
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "users.db"

SECRET_KEY = os.getenv("IV_DRIP_SECRET_KEY", "change-this-development-secret-before-deployment")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def hash_password(password: str):
    salt = secrets.token_bytes(16)
    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=16384,
        r=8,
        p=1,
        dklen=64
    )
    return f"scrypt${salt.hex()}${password_hash.hex()}"


def verify_password(password: str, stored_hash: str):
    try:
        algorithm, salt_hex, hash_hex = stored_hash.split("$")
        if algorithm != "scrypt":
            return False

        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=16384,
            r=8,
            p=1,
            dklen=64
        )
        return hmac.compare_digest(actual_hash, expected_hash)
    except (ValueError, TypeError):
        return False


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_auth_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'nurse')),
                created_at TEXT NOT NULL
            )
        """)
        cols = [r["name"] for r in connection.execute("PRAGMA table_info(users)").fetchall()]
        if "phone" not in cols:
            connection.execute("ALTER TABLE users ADD COLUMN phone TEXT")


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str
    phone: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


def create_access_token(user_id: int, role: str):
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": expires
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token_and_get_user(token: str) -> dict:
    """Decodes and validates a raw JWT token string, returning the user dict."""
    credentials_error = HTTPException(
        status_code=401,
        detail="Invalid or expired login session",
        headers={"WWW-Authenticate": "Bearer"}
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise credentials_error

    with get_connection() as connection:
        user = connection.execute(
            "SELECT id, name, email, role, phone FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    if user is None:
        raise credentials_error

    return dict(user)


def get_current_user(token: str = Depends(oauth2_scheme)):
    return verify_token_and_get_user(token)


def require_role(required_role: str):
    def role_checker(user: dict = Depends(get_current_user)):
        if user["role"] != required_role:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this resource"
            )
        return user
    return role_checker


@router.post("/register")
def register_user(data: RegisterRequest):
    if len(data.name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Enter a valid name")

    if len(data.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters"
        )

    if data.role not in ("admin", "nurse"):
        raise HTTPException(status_code=400, detail="Invalid role")

    password_hash = hash_password(data.password)

    try:
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (name, email, password_hash, role, phone, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    data.name.strip(),
                    data.email.lower(),
                    password_hash,
                    data.role,
                    data.phone.strip() if data.phone else None,
                    datetime.now(timezone.utc).isoformat()
                )
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="An account with this email already exists"
        )

    return {
        "message": "Account created successfully",
        "user_id": user_id
    }


@router.post("/login")
def login_user(data: LoginRequest):
    with get_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (data.email.lower(),)
        ).fetchone()

    if user is None or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user["id"], user["role"])

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "phone": user["phone"] if "phone" in user.keys() else None
        }
    }


@router.post("/token")
def login_for_swagger(data: OAuth2PasswordRequestForm = Depends()):
    with get_connection() as connection:
        user = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (data.username.lower(),)
        ).fetchone()

    if user is None or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    token = create_access_token(user["id"], user["role"])
    return {
        "access_token": token,
        "token_type": "bearer"
    }


@router.get("/me")
def get_my_profile(user: dict = Depends(get_current_user)):
    return user


@router.get("/admin-only")
def admin_check(user: dict = Depends(require_role("admin"))):
    return {"message": "Admin access granted", "user": user}


@router.get("/nurse-only")
def nurse_check(user: dict = Depends(require_role("nurse"))):
    return {"message": "Nurse access granted", "user": user}
