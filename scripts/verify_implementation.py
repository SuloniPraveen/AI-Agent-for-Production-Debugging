#!/usr/bin/env python3
"""Smoke-test Phase 0–1 behavior (sanitization, API, DB, RAG plumbing, chat history).

Run from repo root with env loaded (same as the app):

  PYTHONPATH=. uv run python scripts/verify_implementation.py

Optional: set VERIFY_LLM=1 to also POST /chat (calls OpenAI; costs tokens).
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import List

from dotenv import load_dotenv


def _ok(name: str) -> None:
    print(f"PASS  {name}")


def _fail(name: str, detail: str, failures: List[str]) -> None:
    print(f"FAIL  {name}: {detail}")
    failures.append(f"{name}: {detail}")


def main() -> int:
    load_dotenv(".env.development")
    load_dotenv()

    failures: List[str] = []

    # --- Sanitization (no app import yet) ---
    from app.utils.sanitization import (
        sanitize_email,
        sanitize_plaintext_secret,
    )

    if sanitize_plaintext_secret("p&ss<word") != "p&ss<word":
        _fail("sanitize_plaintext_secret", "must not HTML-escape secrets", failures)
    else:
        _ok("sanitize_plaintext_secret preserves & and <")

    try:
        em = sanitize_email("  Verify.User@Example.COM  ")
        if em != "verify.user@example.com":
            _fail("sanitize_email", f"expected lowercase trimmed email, got {em!r}", failures)
        else:
            _ok("sanitize_email normalizes case and whitespace")
    except Exception as e:
        _fail("sanitize_email", str(e), failures)

    # --- Tools registered ---
    from app.core.langgraph.tools import tools as bound_tools

    names = sorted(t.name for t in bound_tools)
    if "search_incident_knowledge" not in names or "search_logs" not in names:
        _fail("tools", f"missing RAG/log tools, have {names}", failures)
    else:
        _ok(f"LangGraph tools include search_logs + search_incident_knowledge ({names})")

    # --- RAG service (embed + SQL; empty table is OK) ---
    async def _rag_once() -> str:
        from app.services.rag import rag_service

        return await rag_service.search("health check query", source_type="any", top_k=2)

    try:
        rag_out = asyncio.run(_rag_once())
        if not rag_out or not isinstance(rag_out, str):
            _fail("rag_service.search", "expected non-empty string", failures)
        else:
            _ok("rag_service.search returns text (empty index or hits)")
    except Exception as e:
        _fail("rag_service.search", str(e), failures)

    # --- Message shaping ---
    from langchain_core.messages import AIMessage

    from app.core.langgraph.graph import LangGraphAgent
    from app.schemas.chat import Message as ChatMessage

    agent = LangGraphAgent()
    out = agent._LangGraphAgent__process_messages(  # noqa: SLF001
        [AIMessage(content=[{"type": "text", "text": "hello from tool path"}])]
    )
    if not out or not isinstance(out[0], ChatMessage):
        _fail("__process_messages", f"expected list[Message], got {out!r}", failures)
    else:
        _ok("__process_messages handles multimodal AIMessage content")

    # --- HTTP API ---
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)

    h = client.get("/health")
    if h.status_code not in (200, 503):
        _fail("/health", f"status {h.status_code}", failures)
    else:
        body = h.json()
        db_comp = (body.get("components") or {}).get("database")
        if h.status_code == 503 and db_comp != "unhealthy":
            _fail("/health", "503 but database component not marked unhealthy", failures)
        else:
            _ok(f"/health -> {h.status_code} (database: {db_comp})")

    r = client.get("/api/v1/health")
    if r.status_code != 200:
        _fail("/api/v1/health", f"status {r.status_code}", failures)
    else:
        _ok("/api/v1/health -> 200")

    if h.status_code == 503:
        print("SKIP  auth/chat checks (database unavailable)")
        return 1 if failures else 0

    email = f"verify_{uuid.uuid4().hex[:10]}@example.com"
    password = "Aa1&x<#Strong"  # special chars including & < (validator allows these)

    reg = client.post("/api/v1/auth/register", json={"email": email, "password": password})
    if reg.status_code != 200:
        _fail("POST /auth/register", f"{reg.status_code} {reg.text}", failures)
    else:
        _ok("POST /auth/register")

    user_token = reg.json()["token"]["access_token"]

    login = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password, "grant_type": "password"},
    )
    if login.status_code != 200:
        _fail("POST /auth/login", f"{login.status_code} {login.text}", failures)
    else:
        _ok("POST /auth/login (password with & and <)")

    sess = client.post("/api/v1/auth/session", headers={"Authorization": f"Bearer {user_token}"})
    if sess.status_code != 200:
        _fail("POST /auth/session", f"{sess.status_code} {sess.text}", failures)
    else:
        _ok("POST /auth/session")

    session_token = sess.json()["token"]["access_token"]

    msg = client.get("/api/v1/chatbot/messages", headers={"Authorization": f"Bearer {session_token}"})
    if msg.status_code != 200:
        _fail("GET /chatbot/messages", f"{msg.status_code} {msg.text}", failures)
    else:
        data = msg.json()
        if data.get("messages") is None:
            _fail("GET /chatbot/messages", "messages is null (should be list)", failures)
        elif not isinstance(data.get("messages"), list):
            _fail("GET /chatbot/messages", "messages must be a list", failures)
        else:
            _ok("GET /chatbot/messages returns messages array (may be empty)")

    if os.environ.get("VERIFY_LLM") == "1":
        chat = client.post(
            "/api/v1/chatbot/chat",
            headers={"Authorization": f"Bearer {session_token}"},
            json={"messages": [{"role": "user", "content": "Reply with exactly: OK"}]},
        )
        if chat.status_code != 200:
            _fail("POST /chatbot/chat", f"{chat.status_code} {chat.text}", failures)
        else:
            cj = chat.json()
            if cj.get("messages") is None:
                _fail("POST /chatbot/chat", "messages is null", failures)
            elif not isinstance(cj.get("citations"), list):
                _fail("POST /chatbot/chat", "citations must be a list", failures)
            else:
                _ok("POST /chatbot/chat -> 200 with messages and citations list")

    else:
        print("SKIP  POST /chatbot/chat (set VERIFY_LLM=1 to call OpenAI)")

    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        return 1
    print("\nAll automated checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
