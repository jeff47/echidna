from __future__ import annotations

import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib import error, request

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NGINX_CONFIG = PROJECT_ROOT / "nginx" / "echidna.conf"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(not _docker_available(), reason="docker daemon unavailable")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _start_nginx_pair(tmp_path: Path) -> tuple[str, int, str, str]:
    suffix = uuid.uuid4().hex[:8]
    network_name = f"echidna-proxy-test-{suffix}"
    upstream_name = f"echidna-upstream-{suffix}"
    proxy_name = f"echidna-proxy-{suffix}"
    port = _free_port()

    upstream_conf = tmp_path / "upstream.conf"
    upstream_conf.write_text(
        "server { listen 8001; location / { return 204; } }\n",
        encoding="utf-8",
    )

    _docker("network", "create", network_name)
    _docker(
        "run",
        "-d",
        "--rm",
        "--name",
        upstream_name,
        "--network",
        network_name,
        "--network-alias",
        "echidna",
        "-v",
        f"{upstream_conf}:/etc/nginx/conf.d/default.conf:ro",
        "nginx:1.27-alpine",
    )
    _docker(
        "run",
        "-d",
        "--rm",
        "--name",
        proxy_name,
        "--network",
        network_name,
        "-p",
        f"{port}:8080",
        "-v",
        f"{NGINX_CONFIG}:/etc/nginx/conf.d/default.conf:ro",
        "nginx:1.27-alpine",
    )

    for _ in range(30):
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as response:
                if response.status == 204:
                    return network_name, port, upstream_name, proxy_name
        except Exception:
            time.sleep(0.2)
    raise AssertionError("proxy container did not become ready in time")


def _stop_nginx_pair(network_name: str, upstream_name: str, proxy_name: str) -> None:
    for container_name in (proxy_name, upstream_name):
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            text=True,
            capture_output=True,
            check=False,
        )
    subprocess.run(
        ["docker", "network", "rm", network_name],
        text=True,
        capture_output=True,
        check=False,
    )


def _status_code(url: str, *, method: str) -> int:
    req = request.Request(url, data=(b"" if method == "POST" else None), method=method)
    try:
        with request.urlopen(req, timeout=2) as response:
            return int(response.status)
    except error.HTTPError as exc:
        return int(exc.code)


def test_nginx_rate_limits_disambiguate_posts(tmp_path: Path) -> None:
    network_name, port, upstream_name, proxy_name = _start_nginx_pair(tmp_path)
    try:
        statuses = [
            _status_code(f"http://127.0.0.1:{port}/disambiguate", method="POST")
            for _ in range(8)
        ]
    finally:
        _stop_nginx_pair(network_name, upstream_name, proxy_name)

    assert 503 in statuses


def test_nginx_rate_limits_orcid_search_gets(tmp_path: Path) -> None:
    network_name, port, upstream_name, proxy_name = _start_nginx_pair(tmp_path)
    try:
        statuses = [
            _status_code(
                f"http://127.0.0.1:{port}/orcid-search?author_name=Jeffrey%20Rice",
                method="GET",
            )
            for _ in range(31)
        ]
    finally:
        _stop_nginx_pair(network_name, upstream_name, proxy_name)

    assert 503 in statuses
