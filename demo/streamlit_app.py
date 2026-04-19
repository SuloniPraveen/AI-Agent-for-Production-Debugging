"""
Phase 4 demo UI: upload logs, run scripted scenarios, chat with citations.
Requires API with DEMO_API_KEY; Streamlit sends X-Demo-API-Key on every request.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests
import streamlit as st
import yaml

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = os.getenv("API_V1_PREFIX", "/api/v1")
DEMO_KEY = os.getenv("DEMO_API_KEY", "").strip()
DEMO_DATA = Path(os.getenv("DEMO_DATA_DIR", str(Path(__file__).resolve().parents[1] / "demo_data")))
# Read timeout for POST /chatbot/chat (seconds). Agent + tools + OpenAI can be several minutes.
CHAT_TIMEOUT = int(os.getenv("DEMO_CHAT_TIMEOUT", "1800"))

st.set_page_config(page_title="Incident debugging demo", layout="wide")


def _headers() -> dict[str, str]:
    if not DEMO_KEY:
        st.error("DEMO_API_KEY is not set. Add it to your environment (see README Phase 4).")
        st.stop()
    return {"X-Demo-API-Key": DEMO_KEY}


def _upload_logs(file_bytes: bytes, filename: str) -> str | None:
    url = f"{API_BASE}{API_PREFIX}/logs/upload"
    r = requests.post(
        url,
        headers=_headers(),
        files={"file": (filename, file_bytes)},
        timeout=120,
    )
    if r.status_code != 200:
        st.error(f"Upload failed: {r.status_code} {r.text}")
        return None
    return str(r.json().get("batch_id", ""))


def _poll_batch(batch_id: str) -> dict | None:
    url = f"{API_BASE}{API_PREFIX}/logs/batches/{batch_id}"
    for _ in range(120):
        r = requests.get(url, headers=_headers(), timeout=30)
        if r.status_code != 200:
            st.error(f"Status poll failed: {r.status_code} {r.text}")
            return None
        data = r.json()
        if data.get("status") in ("completed", "failed"):
            return data
        time.sleep(1)
    st.warning("Ingestion still running; refresh batch status later.")
    return None


def _chat(payload: dict) -> dict | None:
    url = f"{API_BASE}{API_PREFIX}/chatbot/chat"
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        json=payload,
        timeout=CHAT_TIMEOUT,
    )
    if r.status_code != 200:
        st.error(f"Chat failed: {r.status_code} {r.text}")
        return None
    return r.json()


def _load_scenarios() -> list[dict]:
    path = DEMO_DATA / "scenarios.yaml"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("scenarios") or [])


st.title("Production debugging demo")
st.caption(f"API: `{API_BASE}` · Auth: demo key (no JWT in the browser)")

tab_upload, tab_scenario, tab_chat = st.tabs(["1. Upload sample logs", "2. Run incident scenario", "3. Chat + citations"])

with tab_upload:
    st.markdown(
        "Use a small **incident-style** sample first (DB / 503 / timeouts / Redis), or a larger synthetic file for stress. "
        "Any UTF-8 `.log` or `.txt` works."
    )

    up = st.file_uploader("Log file", type=["log", "txt"])
    if st.button("Upload and ingest", disabled=up is None):
        body = up.getvalue()
        fn = up.name or "upload.log"
        with st.spinner("Uploading…"):
            bid = _upload_logs(body, fn)
        if bid:
            st.session_state["last_log_batch_id"] = bid
            st.success(f"Batch **{bid}** accepted.")
            with st.spinner("Waiting for ingestion…"):
                status = _poll_batch(bid)
            if status:
                st.json(status)

with tab_scenario:
    scenarios = _load_scenarios()
    if not scenarios:
        st.warning("No `demo_data/scenarios.yaml` found.")
    else:
        labels = [f"{s.get('title', s['id'])} ({s['id']})" for s in scenarios]
        idx = st.selectbox("Scenario", range(len(scenarios)), format_func=lambda i: labels[i])
        sc = scenarios[idx]
        st.markdown(sc.get("suggested_prompt", ""))
        themes = sc.get("expected_themes") or []
        if themes:
            st.caption("Scripted themes to look for in answers (human checklist): " + ", ".join(themes))
        if st.button("Use this prompt in Chat tab"):
            st.session_state.user_msg = (sc.get("suggested_prompt") or "").strip()
            st.success("Open **Chat + citations** — the prompt is filled there.")

with tab_chat:
    if "user_msg" not in st.session_state:
        st.session_state.user_msg = (
            "We deployed this morning and checkout returns 503. What should we check using the logs?"
        )
    st.caption(
        "Log search uses **your most recent completed upload** unless you just uploaded a file in tab 1 "
        "(that batch is pinned for this browser session)."
    )
    st.text_area("Your message", height=120, key="user_msg")
    if st.button("Send"):
        msgs = [{"role": "user", "content": st.session_state.user_msg.strip()}]
        if not msgs[0]["content"]:
            st.warning("Enter a message.")
        else:
            payload: dict = {"messages": msgs}
            bid = st.session_state.get("last_log_batch_id")
            if bid:
                payload["focus_log_batch_id"] = bid
            with st.spinner("Agent running…"):
                out = _chat(payload)
            if out:
                for m in out.get("messages") or []:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    with st.chat_message(role):
                        st.markdown(content)
                cites = out.get("citations") or []
                st.subheader("Citations (log chunks)")
                if not cites:
                    st.caption("No log citations returned for this turn (try after ingesting logs and asking a log-grounded question).")
                else:
                    for c in cites:
                        with st.expander(
                            f"chunk {c.get('chunk_id')} · {c.get('service') or '?'} · {c.get('level') or '?'}"
                        ):
                            st.code(c.get("snippet") or "", language="text")
                            st.json(
                                {
                                    k: c.get(k)
                                    for k in ("batch_id", "timestamp", "line_start", "line_end")
                                    if c.get(k) is not None
                                }
                            )
