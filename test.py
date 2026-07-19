#!/usr/bin/env python3
"""OpenAI connectivity diagnostic script.

Usage:
  python3 test.py
  python3 test.py --skip-chat --skip-embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import ssl
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_step(name: str, status: str, detail: str = "") -> None:
    line = f"[{status}] {name}"
    if detail:
        line += f" -> {detail}"
    print(line)


def mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 10:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]} (len={len(value)})"


def env_snapshot() -> Dict[str, str]:
    keys = [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ]
    return {k: os.getenv(k, "") for k in keys}


def check_dns(host: str = "api.openai.com") -> None:
    name = "DNS resolution"
    start = time.perf_counter()
    try:
        addresses = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ips = sorted({entry[4][0] for entry in addresses})
        elapsed = (time.perf_counter() - start) * 1000
        print_step(name, "OK", f"resolved {host} to {len(ips)} IP(s) in {elapsed:.2f} ms")
        for ip in ips[:8]:
            print(f"    - {ip}")
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000
        print_step(name, "FAIL", f"{type(exc).__name__}: {exc} ({elapsed:.2f} ms)")


def check_tls(host: str = "api.openai.com", port: int = 443) -> None:
    name = "TLS handshake"
    start = time.perf_counter()
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                protocol = ssock.version()
                cipher = ssock.cipher()
                elapsed = (time.perf_counter() - start) * 1000
                print_step(
                    name,
                    "OK",
                    f"protocol={protocol}, cipher={cipher[0] if cipher else 'n/a'}, {elapsed:.2f} ms",
                )
                subject = cert.get("subject", [])
                print(f"    - cert subject: {subject}")
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000
        print_step(name, "FAIL", f"{type(exc).__name__}: {exc} ({elapsed:.2f} ms)")


def create_client(api_key: str, base_url: str) -> OpenAI:
    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "timeout": 30.0,
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def run_models_check(client: OpenAI, model: str) -> None:
    name = "Model access check"
    start = time.perf_counter()
    try:
        info = client.models.retrieve(model)
        elapsed = (time.perf_counter() - start) * 1000
        print_step(name, "OK", f"model={info.id}, owned_by={info.owned_by}, {elapsed:.2f} ms")
    except Exception as exc:  # noqa: BLE001
        handle_openai_exception(name, exc, start)


def run_chat_check(client: OpenAI, model: str) -> None:
    name = "Chat completion check"
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a diagnostic assistant."},
                {"role": "user", "content": "Reply with only: pong"},
            ],
            max_tokens=20,
            temperature=0,
        )
        elapsed = (time.perf_counter() - start) * 1000
        content = response.choices[0].message.content if response.choices else "<no choices>"
        usage = getattr(response, "usage", None)
        print_step(name, "OK", f"{elapsed:.2f} ms")
        print(f"    - response: {content!r}")
        if usage is not None:
            print(f"    - usage: {usage}")
    except Exception as exc:  # noqa: BLE001
        handle_openai_exception(name, exc, start)


def run_embeddings_check(client: OpenAI, embedding_model: str) -> None:
    name = "Embeddings check"
    start = time.perf_counter()
    try:
        response = client.embeddings.create(
            model=embedding_model,
            input=["motor controller reverse torque behavior"],
        )
        elapsed = (time.perf_counter() - start) * 1000
        dim = len(response.data[0].embedding) if response.data else 0
        print_step(name, "OK", f"embedding_dim={dim}, vectors={len(response.data)}, {elapsed:.2f} ms")
    except Exception as exc:  # noqa: BLE001
        handle_openai_exception(name, exc, start)


def handle_openai_exception(name: str, exc: Exception, start_time: float) -> None:
    elapsed = (time.perf_counter() - start_time) * 1000
    if isinstance(exc, AuthenticationError):
        print_step(name, "FAIL", f"AuthenticationError: {exc} ({elapsed:.2f} ms)")
    elif isinstance(exc, PermissionDeniedError):
        print_step(name, "FAIL", f"PermissionDeniedError: {exc} ({elapsed:.2f} ms)")
    elif isinstance(exc, RateLimitError):
        print_step(name, "FAIL", f"RateLimitError: {exc} ({elapsed:.2f} ms)")
    elif isinstance(exc, BadRequestError):
        print_step(name, "FAIL", f"BadRequestError: {exc} ({elapsed:.2f} ms)")
    elif isinstance(exc, APITimeoutError):
        print_step(name, "FAIL", f"APITimeoutError: {exc} ({elapsed:.2f} ms)")
    elif isinstance(exc, APIConnectionError):
        print_step(name, "FAIL", f"APIConnectionError: {exc} ({elapsed:.2f} ms)")
    elif isinstance(exc, APIStatusError):
        print_step(
            name,
            "FAIL",
            (
                f"APIStatusError: status={exc.status_code}, "
                f"response={getattr(exc, 'response', None)}, {elapsed:.2f} ms"
            ),
        )
    elif isinstance(exc, APIError):
        print_step(name, "FAIL", f"APIError: {exc} ({elapsed:.2f} ms)")
    else:
        print_step(name, "FAIL", f"{type(exc).__name__}: {exc} ({elapsed:.2f} ms)")

    print("    - traceback:")
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    for line in tb.splitlines():
        print(f"      {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug OpenAI connectivity from this environment.")
    parser.add_argument("--skip-chat", action="store_true", help="Skip chat completion test")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embeddings test")
    parser.add_argument("--skip-model", action="store_true", help="Skip model access test")
    args = parser.parse_args()

    print_header("OpenAI Connection Diagnostic")

    load_dotenv()

    now = datetime.now(timezone.utc).isoformat()
    print(f"Timestamp (UTC): {now}")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")

    print_header("Environment")
    snap = env_snapshot()

    api_key = snap["OPENAI_API_KEY"]
    model = snap["OPENAI_MODEL"] or "gpt-4o-mini"
    embedding_model = snap["OPENAI_EMBEDDING_MODEL"] or "text-embedding-3-small"
    base_url = snap["OPENAI_BASE_URL"]

    print(f"OPENAI_API_KEY: {mask_secret(api_key)}")
    print(f"OPENAI_MODEL: {model}")
    print(f"OPENAI_EMBEDDING_MODEL: {embedding_model}")
    print(f"OPENAI_BASE_URL: {base_url or '<default>'}")
    print(f"HTTP_PROXY: {snap['HTTP_PROXY'] or '<not set>'}")
    print(f"HTTPS_PROXY: {snap['HTTPS_PROXY'] or '<not set>'}")
    print(f"ALL_PROXY: {snap['ALL_PROXY'] or '<not set>'}")
    print(f"NO_PROXY: {snap['NO_PROXY'] or '<not set>'}")

    if not api_key:
        print_step("API key presence", "FAIL", "OPENAI_API_KEY is missing in .env")
        return 2
    print_step("API key presence", "OK", "key loaded from .env")

    print_header("Network Prechecks")
    check_dns("api.openai.com")
    check_tls("api.openai.com", 443)

    print_header("OpenAI SDK Checks")
    try:
        client = create_client(api_key=api_key, base_url=base_url)
        print_step("Client initialization", "OK", "OpenAI client created")
    except Exception as exc:  # noqa: BLE001
        print_step("Client initialization", "FAIL", f"{type(exc).__name__}: {exc}")
        print("Traceback:")
        print(traceback.format_exc())
        return 3

    if not args.skip_model:
        run_models_check(client, model)
    else:
        print_step("Model access check", "SKIP", "disabled by flag")

    if not args.skip_chat:
        run_chat_check(client, model)
    else:
        print_step("Chat completion check", "SKIP", "disabled by flag")

    if not args.skip_embeddings:
        run_embeddings_check(client, embedding_model)
    else:
        print_step("Embeddings check", "SKIP", "disabled by flag")

    print_header("Done")
    print("Diagnostic run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
