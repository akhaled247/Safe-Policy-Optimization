"""Filter Farama Adroit dense-v1 import notice (spam under spawn workers)."""

from __future__ import annotations

import sys

_ADROIT_MARKERS = (
    "AdroitHandRelocateDense-v1",
    "AdroitHandHammerDense-v1",
    "AdroitHandDoorDense-v1",
    "gymnasium-robotics",
)


def _is_adroit_spam(text: str) -> bool:
    return any(marker in text for marker in _ADROIT_MARKERS)


def silence_farama_adroit_spam() -> None:
    """Drop whole Adroit/Farama notice lines from stdout/stderr (no blank lines)."""
    if getattr(sys.stdout, "_safepo_farama_filtered", False):
        return

    class _Filter:
        __slots__ = ("_stream", "_buf")
        _safepo_farama_filtered = True

        def __init__(self, stream):
            self._stream = stream
            self._buf = ""

        def write(self, data):
            if isinstance(data, bytes):
                text = data.decode(getattr(self._stream, "encoding", "utf-8"), errors="replace")
            else:
                text = data if isinstance(data, str) else str(data)

            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line and not _is_adroit_spam(line):
                    self._stream.write(line + "\n")
            return len(data)

        def flush(self):
            if self._buf and not _is_adroit_spam(self._buf):
                self._stream.write(self._buf)
            self._buf = ""
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
