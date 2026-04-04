from __future__ import annotations

import logging
import sys
from pathlib import Path


class _PhaseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        phase = getattr(record, "phase", "SYSTEM")
        record.phase = str(phase).upper()
        return True


class _TqdmCompatibleStreamHandler(logging.StreamHandler):
    """Route log lines through tqdm when available so progress bars stay intact."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            from tqdm.auto import tqdm

            tqdm.write(message, file=self.stream)
            self.flush()
        except Exception:
            super().emit(record)


def setup_logger(log_dir: str | Path, name: str = "rhpe_boneage") -> logging.Logger:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.filters.clear()
    logger.propagate = False
    logger.addFilter(_PhaseFilter())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(phase)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = _TqdmCompatibleStreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(directory / "run.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
