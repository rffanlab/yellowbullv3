"""YellowBull Agent — entry point."""

import sys

from config.settings import load_settings
from core.logging_setup import setup_logging


def main():
    settings = load_settings()
    setup_logging("DEBUG" if settings.server.debug else "INFO")

    import uvicorn

    uvicorn.run(
        "api.server:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
