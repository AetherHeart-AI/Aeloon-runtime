"""Human-readable benchmark progress on stderr."""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "aeloon.benchmarks"


class _DynamicStderrHandler(logging.Handler):
    """Resolve stderr at emit time so redirects and test capture keep working."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(self.format(record), file=sys.stderr, flush=True)
        except Exception:
            self.handleError(record)


def configure_progress() -> None:
    """Enable INFO progress without affecting the JSON written to stdout."""

    logger = logging.getLogger(LOGGER_NAME)
    if not any(isinstance(handler, _DynamicStderrHandler) for handler in logger.handlers):
        handler = _DynamicStderrHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s INFO %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def info(message: str, *args: object) -> None:
    logging.getLogger(LOGGER_NAME).info(message, *args)
