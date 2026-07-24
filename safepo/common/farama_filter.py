"""Filter Farama Adroit dense-v1 import notice (spam under spawn workers)."""

from __future__ import annotations

import sys


def silence_farama_adroit_spam() -> None:
    """Drop AdroitHand*Dense-v1 Farama notice lines from stdout/stderr."""
    if getattr(sys.stdout, "_safepo_farama_filtered", False):
        return

    class _Filter:
        __slots__ = ("_stream",)
        _safepo_farama_filtered = True

        def __init__(self, stream):
            self._stream = stream

        def write(self, data):
            text = data if isinstance(data, str) else str(data)
            if "AdroitHandRelocateDense-v1" in text:
                return 0
            return self._stream.write(data)

        def flush(self):
            return self._stream.flush()

        def fileno(self):
            return self._stream.fileno()

        def isatty(self):
            return self._stream.isatty()

        @property
        def encoding(self):
            return getattr(self._stream, "encoding", "utf-8")

        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _Filter(sys.stdout)  # type: ignore[assignment]
    sys.stderr = _Filter(sys.stderr)  # type: ignore[assignment]
