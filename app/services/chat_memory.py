"""
app/services/chat_memory.py
============================
Chat session persistence with multi-document tracking.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# ─────────────────────────────────────────────────────────────────────────────
# Storage paths
# ─────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
SESSIONS_DIR = DATA_DIR / "sessions"

DATA_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

# Maximum sessions shown in the sidebar — prevents slowdown from test data
_MAX_SIDEBAR_SESSIONS = 20


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _session_meta_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# JSON-backed chat history
# ─────────────────────────────────────────────────────────────────────────────

class JsonChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._path = _session_path(session_id)
        self._messages: list[BaseMessage] = []
        self._load()

    def _load(self):
        if self._path.exists():
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._messages = [
                HumanMessage(content=m["content"]) if m["role"] == "human"
                else AIMessage(content=m["content"])
                for m in data.get("messages", [])
            ]
        else:
            self._messages = []

    def _save(self):
        data = {
            "session_id": self.session_id,
            "updated_at": datetime.utcnow().isoformat(),
            "messages": [
                {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
                for m in self._messages
            ]
        }
        self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def messages(self) -> list[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)
        self._save()

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        self._messages = []
        self._save()


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

_session_cache: dict[str, JsonChatMessageHistory] = {}


def get_session_history(session_id: str) -> JsonChatMessageHistory:
    if session_id not in _session_cache:
        _session_cache[session_id] = JsonChatMessageHistory(session_id)
    return _session_cache[session_id]


def save_session(session_id: str):
    """Force-save a session (messages auto-save, but this updates metadata)."""
    meta_path = _session_meta_path(session_id)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["updated_at"] = datetime.utcnow().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def delete_session(session_id: str) -> bool:
    """Delete session history, metadata, and cache."""
    removed = False
    for path in [_session_path(session_id), _session_meta_path(session_id)]:
        if path.exists():
            path.unlink()
            removed = True
    _session_cache.pop(session_id, None)
    return removed


def get_session_messages(session_id: str) -> list[dict]:
    path = _session_path(session_id)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("messages", [])


def list_sessions() -> list[dict]:
    """
    Return the most recent _MAX_SIDEBAR_SESSIONS sessions for the sidebar.

    Performance fix: reads only the _meta.json file per session (tiny, ~200 bytes)
    instead of the full message file. The message file is only read when the user
    actually opens a session (get_session_messages). This means list_sessions is
    O(1) per session regardless of how many messages a session has.

    Also sorts by meta file mtime (OS-level, no file read needed) to find the
    most recent sessions before reading anything, so even 1000 sessions on disk
    only results in _MAX_SIDEBAR_SESSIONS file reads.
    """
    # Step 1: collect all meta files, sort by OS mtime (no file read)
    meta_files = sorted(
        SESSIONS_DIR.glob("*_meta.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    sessions = []
    for meta_path in meta_files:
        if len(sessions) >= _MAX_SIDEBAR_SESSIONS:
            break
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            sid = meta_path.stem.replace("_meta", "")

            # Count messages only if the session file exists — use cached size
            # rather than reading the whole file
            session_path = _session_path(sid)
            message_count = 0
            if session_path.exists():
                # Fast count: read file once, count "role" keys without parsing
                # full message objects
                raw = session_path.read_text(encoding="utf-8")
                message_count = raw.count('"role"')

            sessions.append({
                "session_id":   sid,
                "title":        meta.get("title", sid[:8]),
                "session_type": meta.get("session_type", "document"),
                "message_count": message_count,
                "updated_at":   meta.get("updated_at", ""),
                "file_count":   len(meta.get("files", [])),
            })
        except Exception as e:
            print(f"Skipping corrupt session file {meta_path}: {e}")

    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Session metadata (title, files, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def set_session_metadata(session_id: str, file_name: str = "", title: str = "",
    session_type: str = "document", research_question: str = "",
    extra_metadata: dict = None):

    path = _session_meta_path(session_id)
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))

    data["session_id"] = session_id
    if file_name:
        data["file_name"] = file_name
    if title:
        data["title"] = title
    if session_type:
        data["session_type"] = session_type
    if research_question:
        data["research_question"] = research_question
    if "files" not in data:
        data["files"] = [file_name] if file_name else []

    if extra_metadata:
        data.update(extra_metadata)

    data["updated_at"] = datetime.utcnow().isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_session_metadata(session_id: str) -> dict:
    path = _session_meta_path(session_id)
    if not path.exists():
        return {"session_id": session_id, "files": []}
    return json.loads(path.read_text(encoding="utf-8"))


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-DOCUMENT TRACKING
# ═════════════════════════════════════════════════════════════════════════════

def add_session_file(session_id: str, file_name: str):
    """Track a file as part of a session."""
    path = _session_meta_path(session_id)
    data = get_session_metadata(session_id)

    files = data.get("files", [])
    if file_name and file_name not in files:
        files.append(file_name)

    data["files"] = files
    data["updated_at"] = datetime.utcnow().isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_session_files(session_id: str) -> list[str]:
    """Get all files associated with a session."""
    return get_session_metadata(session_id).get("files", [])


def remove_session_file(session_id: str, file_name: str):
    """Remove a file from a session's tracking."""
    path = _session_meta_path(session_id)
    data = get_session_metadata(session_id)

    files = data.get("files", [])
    if file_name in files:
        files.remove(file_name)

    data["files"] = files
    data["updated_at"] = datetime.utcnow().isoformat()
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")