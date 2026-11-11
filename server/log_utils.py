"""
Logging utilities for tts-troupe.

Routes per-component logs into date-partitioned directories:
  logs/<YYYY-MM-DD>/<component>.log
with the date computed per record, so long-running operations that cross
midnight land in the correct day file. A unified live stream also goes to
console stdout. Worker stderr relayed through ``worker.<model>`` loggers
resolves to the model own component file.
"""
import datetime
import logging
import os
import sys
from typing import Dict, Optional, Tuple

# Repository root: server/log_utils.py -> server/ -> repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

# Ordered prefix -> component mapping (first match wins).
_COMPONENT_PREFIXES = (
    ("cuda", "cuda_utils"),
    ("model_manager", "model_manager"),
    ("proxy_adapter", "proxy_adapter"),
    ("model_worker", "model_worker"),
    ("voice_cache", "voice_cache"),
    ("transcribe_voices", "transcribe_voices"),
    ("faster_whisper", "transcribe_voices"),
    ("ctranslate2", "transcribe_voices"),
    ("config", "config"),
    ("audio_utils", "audio_utils"),
)


class ComponentLogRouter(logging.Handler):
    """
    Dispatches log records to per-component files organized by date:
      logs/<YYYY-MM-DD>/<component>.log
    The date is computed at emit time (never frozen at handler construction),
    so records from operations that cross midnight land in the correct day
    file. Handler cache is keyed by (date, component); on rollover the
    previous day handles are closed and reopened lazily on demand.
    """

    def __init__(self, logs_dir: str = LOGS_DIR, level=logging.NOTSET):
        super().__init__(level=level)
        self.logs_dir = logs_dir
        self._handlers: Dict[Tuple[str, str], logging.FileHandler] = {}
        self._current_date: Optional[str] = None

    def _determine_component(self, logger_name: str) -> str:
        name = logger_name.strip()
        for prefix, component in _COMPONENT_PREFIXES:
            if prefix in name:
                return component
        if "server" in name or "uvicorn" in name or "fastapi" in name:
            return "server"
        parts = name.split(".")
        return parts[-1] if parts else "server"

    def _get_file_handler(self, date: str, component: str) -> logging.FileHandler:
        key = (date, component)
        handler = self._handlers.get(key)
        if handler is None:
            target_dir = os.path.join(self.logs_dir, date)
            os.makedirs(target_dir, exist_ok=True)
            file_path = os.path.join(target_dir, f"{component}.log")
            handler = logging.FileHandler(file_path, encoding="utf-8")
            if self.formatter:
                handler.setFormatter(self.formatter)
            self._handlers[key] = handler
            # Date rollover: retire the previous day open handles.
            if self._current_date is not None and self._current_date != date:
                stale = [k for k in self._handlers if k[0] != date]
                for k in stale:
                    try:
                        self._handlers.pop(k).close()
                    except Exception:
                        pass
            self._current_date = date
        return handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            component = self._determine_component(record.name)
            handler = self._get_file_handler(date, component)
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        for handler in self._handlers.values():
            handler.flush()

    def close(self) -> None:
        for handler in self._handlers.values():
            handler.close()
        self._handlers.clear()
        self._current_date = None
        super().close()


def setup_server_logging(level=logging.INFO) -> None:
    """
    Configure global logging for the server application:
    - Console output to stdout (for start_tts_server.bat).
    - Per-component file routing to logs/<YYYY-MM-DD>/<component>.log, with
      the date determined per record so midnight rollovers are correct.
      Worker stderr relayed through ``worker.<model>`` loggers lands in the
      model own component file.
    - Suppresses routine HTTP request polling noise from httpx/httpcore.
    """
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    router_handler = ComponentLogRouter(logs_dir=LOGS_DIR, level=level)
    router_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(router_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
