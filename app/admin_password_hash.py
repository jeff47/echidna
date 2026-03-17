from __future__ import annotations

import argparse
import getpass
import sys

from app.auth import scrypt_password_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate an ECHIDNA admin password hash.")
    parser.add_argument(
        "password",
        nargs="?",
        help="Admin password. Omit to enter it securely via prompt.",
    )
    args = parser.parse_args(argv)

    password = args.password
    if password is None:
        first = getpass.getpass("Admin password: ")
        second = getpass.getpass("Confirm admin password: ")
        if not first:
            print("Password must not be empty.", file=sys.stderr)
            return 1
        if first != second:
            print("Passwords do not match.", file=sys.stderr)
            return 1
        password = first

    print(scrypt_password_hash(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
