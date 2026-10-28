"""
Persistent safety event log.

Records PPE violations and unknown-person sightings as JSON lines so that
supervisors can review afterwards when (and for whom) equipment was missing.
Only text metadata is stored - no images or face data.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("event_log")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVENT_LOG_PATH = _PROJECT_ROOT / "data" / "events" / "safety_events.jsonl"

# Oldest entries are dropped beyond this many, so the file cannot grow forever
MAX_EVENTS = 5000


class SafetyEventLog:
    """Thread-safe append-only JSON-lines store of safety events."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path or os.getenv("PPE_EVENT_LOG") or DEFAULT_EVENT_LOG_PATH)
        self._lock = threading.Lock()
        self._events: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.is_file():
            return []
        events: List[Dict[str, Any]] = []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping corrupt line in %s", self.path)
        except OSError as err:
            logger.error("Failed to read event log %s: %s", self.path, err)
        return events[-MAX_EVENTS:]

    def _rewrite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event) + "\n")

    def append(self, event: Dict[str, Any]) -> None:
        """Store one event in memory and on disk."""
        with self._lock:
            self._events.append(event)
            try:
                if len(self._events) > MAX_EVENTS:
                    self._events = self._events[-MAX_EVENTS:]
                    self._rewrite()
                else:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(event) + "\n")
            except OSError as err:
                logger.error("Failed to write event log %s: %s", self.path, err)

    def list(self, limit: int = 200, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return the most recent events, newest first."""
        with self._lock:
            events = self._events
            if event_type:
                events = [e for e in events if e.get("type") == event_type]
            return list(reversed(events[-limit:])) if limit > 0 else []

    def count(self) -> int:
        with self._lock:
            return len(self._events)

    def clear(self) -> int:
        """Delete all events. Returns how many were removed."""
        with self._lock:
            removed = len(self._events)
            self._events = []
            try:
                self._rewrite()
            except OSError as err:
                logger.error("Failed to clear event log %s: %s", self.path, err)
            return removed
