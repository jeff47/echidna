from __future__ import annotations

import hashlib
import os
import secrets


def scrypt_password_hash(
    password: str,
    *,
    salt: bytes | None = None,
    n: int = 1 << 14,
    r: int = 8,
    p: int = 1,
) -> str:
    if salt is None:
        salt = os.urandom(16)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p)
    return f"scrypt${n}${r}${p}${salt.hex()}${derived.hex()}"


def verify_password_against_hash(password: str, password_hash: str) -> bool:
    algorithm, n_raw, r_raw, p_raw, salt_hex, expected_hex = password_hash.split("$", 5)
    if algorithm != "scrypt":
        raise ValueError("Unsupported password hash algorithm")
    n = int(n_raw)
    r = int(r_raw)
    p = int(p_raw)
    salt = bytes.fromhex(salt_hex)
    expected = bytes.fromhex(expected_hex)
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=len(expected))
    return secrets.compare_digest(derived, expected)
