"""YellowBull Agent — entry point."""


import argparse
import asyncio

from config.settings import load_settings
from core.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="YellowBull Agent")
    parser.add_argument("--cli", action="store_true", help="Start interactive CLI REPL")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (YAML)")
    args = parser.parse_args()

    settings = load_settings(args.config)
    setup_logging("DEBUG" if settings.server.debug else "INFO")

    if args.cli:
        from cli.repl import run_cli

        asyncio.run(run_cli(config_path=args.config))
        return

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
