"""Session manager for conversation state."""

from datetime import datetime

from models.session import Session


class SessionManager:
    """In-memory session store. Replace with Redis/DB in production."""

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def create(self, user_id: str) -> Session:
        session = Session(user_id=user_id)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired(self, max_age_seconds: int = 3600 * 24):
        """Remove sessions older than max_age_seconds."""
        now = datetime.now()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.updated_at).total_seconds() > max_age_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
