"""Session manager for conversation state."""

from __future__ import annotations

import abc
from datetime import datetime
from typing import TYPE_CHECKING

from models.session import Session

if TYPE_CHECKING:
    pass


# ── Persistence adapter interface ──────────────────────────────────────


class SessionPersistenceAdapter(abc.ABC):
    """Abstract persistence backend for sessions.

    Implement this interface to add Redis, SQLite, or any other storage.
    The in-memory manager will call hooks on save/load/delete.
    """

    @abc.abstractmethod
    def save(self, session: Session) -> None:
        """Persist a session (called after create/update)."""
        ...

    @abc.abstractmethod
    def load(self, session_id: str) -> Session | None:
        """Load a session from storage."""
        ...

    @abc.abstractmethod
    def delete(self, session_id: str) -> bool:
        """Remove a session from storage."""
        ...

    @abc.abstractmethod
    def list_session_ids(self, user_id: str | None = None) -> list[str]:
        """List all session IDs, optionally filtered by user."""
        ...

    @abc.abstractmethod
    def cleanup_expired(self, max_age_seconds: int) -> list[str]:
        """Remove expired sessions and return deleted IDs."""
        ...


# ── In-memory adapter (default) ────────────────────────────────────────


class MemorySessionAdapter(SessionPersistenceAdapter):
    """In-memory session store. Replace with Redis/DB in production."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def save(self, session: Session) -> None:
        self._sessions[session.session_id] = session

    def load(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_session_ids(self, user_id: str | None = None) -> list[str]:
        if user_id:
            return [sid for sid, s in self._sessions.items() if s.user_id == user_id]
        return list(self._sessions.keys())

    def cleanup_expired(self, max_age_seconds: int) -> list[str]:
        now = datetime.now()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.updated_at).total_seconds() > max_age_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        return expired


# ── Session Manager ────────────────────────────────────────────────────


class SessionManager:
    """Manages conversation sessions with pluggable persistence."""

    def __init__(
        self,
        adapter: SessionPersistenceAdapter | None = None,
        default_ttl_seconds: int = 7200,  # 2 hours — matches config default
    ):
        self._adapter = adapter or MemorySessionAdapter()
        self.default_ttl_seconds = default_ttl_seconds

    def create(self, user_id: str) -> Session:
        session = Session(user_id=user_id)
        self._adapter.save(session)
        return session

    def get(self, session_id: str) -> Session | None:
        return self._adapter.load(session_id)

    def delete(self, session_id: str) -> bool:
        return self._adapter.delete(session_id)

    def list_sessions(
        self, user_id: str | None = None
    ) -> list[tuple[str, Session]]:
        """List sessions as (session_id, session) pairs."""
        result = []
        for sid in self._adapter.list_session_ids(user_id):
            s = self._adapter.load(sid)
            if s:
                result.append((sid, s))
        return result

    def cleanup_expired(self, max_age_seconds: int | None = None):
        """Remove sessions older than max_age_seconds.

        Defaults to the manager's default_ttl_seconds.
        """
        if max_age_seconds is None:
            max_age_seconds = self.default_ttl_seconds
        return self._adapter.cleanup_expired(max_age_seconds)

    @property
    def _sessions(self) -> dict[str, Session]:
        """Backward-compatible property for code that accesses internal store directly."""
        if isinstance(self._adapter, MemorySessionAdapter):
            return self._adapter._sessions
        raise AttributeError("Use list_sessions() instead of _sessions with external adapters")
