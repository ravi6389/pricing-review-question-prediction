"""
Logging setup for the pipeline.

WHY THIS EXISTS: the pipeline runs unattended for hours on a GPU box, and
`print` gives no levels, no timestamps and no file record — when a stage
fails at 2am the only evidence is whatever scrolled past. Every stage now
writes to both the console and a per-run log file, so a failed run can be
diagnosed after the fact.

Levels used in this codebase:
  DEBUG    per-item detail (each document, each generation)
  INFO     stage boundaries, counts, progress checkpoints
  WARNING  something recoverable that changes the result — a dropped
           training argument, a tokenizer that makes losses incomparable
  ERROR    a stage could not complete

Usage at the top of a module:

    from logging_config import get_logger
    log = get_logger(__name__)

and once, in the entry point:

    setup_logging(results_dir=Path("results"))
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
DATE_FORMAT = "%H:%M:%S"

_CONFIGURED = False


def setup_logging(
    results_dir: Optional[Path] = None,
    level: Optional[str] = None,
    log_name: str = "pipeline.log",
) -> logging.Logger:
    """
    Configure root logging once. Safe to call repeatedly — later calls are
    ignored rather than stacking duplicate handlers, which is the usual way
    log lines end up printed three times.

    Level resolution order: explicit argument, then the LOG_LEVEL environment
    variable, then INFO. The environment variable means a Kaggle run can be
    put into DEBUG without editing code.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logging.getLogger("pipeline")

    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    root.setLevel(getattr(logging, resolved, logging.INFO))

    for h in list(root.handlers):          # drop anything a library installed
        root.removeHandler(h)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if results_dir is not None:
        try:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(results_dir / log_name, encoding="utf-8")
            fh.setFormatter(formatter)
            fh.setLevel(logging.DEBUG)     # file keeps more than the console
            root.addHandler(fh)
        except OSError as exc:
            # A read-only or missing results dir must not take the run down;
            # console logging is enough to keep going.
            root.warning("file logging disabled (%s)", exc)

    # These libraries are chatty at INFO and drown the pipeline's own output.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "filelock",
                  "transformers.tokenization_utils_base"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger("pipeline")


def get_logger(name: str) -> logging.Logger:
    """
    Module-level logger. Falls back to a console-only configuration if the
    entry point never called setup_logging — a module imported directly in a
    notebook should still log rather than go silent.

    Each stage is run as a script by the runner, so __name__ is "__main__" and
    every line would otherwise be labelled identically. The stem of the
    calling file is used instead, which is what makes a combined log readable.
    """
    if name == "__main__":
        import inspect
        frame = inspect.stack()[1]
        name = Path(frame.filename).stem

    if not _CONFIGURED and not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format=LOG_FORMAT,
            datefmt=DATE_FORMAT,
            stream=sys.stdout,
        )
    return logging.getLogger(name.split(".")[-1])
