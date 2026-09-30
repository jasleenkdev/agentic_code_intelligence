"""Citation verification gate (D-005, NFR-5): every file:line is re-read from disk before it is
returned. A citation survives only if the file is inside the repo, the range exists, and — when
the caller knows what should be there — at least one expected token appears in the range."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

MAX_SNIPPET_LINES = 60


@dataclass
class Snippet:
    file_path: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None
    reason: str = ""
    highlight_lines: list[int] = field(default_factory=list)
    verified: bool = True
    truncated: bool = False

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["location"] = self.location
        return d


@lru_cache(maxsize=512)
def _read_lines(path_str: str, mtime: float) -> tuple[str, ...]:
    return tuple(Path(path_str).read_text(encoding="utf-8", errors="replace").split("\n"))


def file_lines(file_path: str, repo_root: Path | None = None) -> tuple[str, ...] | None:
    root = (repo_root or config.REPO_ROOT).resolve()
    target = (root / file_path.lstrip("/")).resolve()
    if root not in target.parents or not target.is_file():
        return None
    return _read_lines(str(target), target.stat().st_mtime)


def verify(
    file_path: str,
    start_line: int,
    end_line: int,
    expect: list[str] | None = None,
    symbol: str | None = None,
    reason: str = "",
    highlight: list[int] | None = None,
    repo_root: Path | None = None,
) -> Snippet | None:
    """Return a verified Snippet with the exact text from disk, or None if it fails."""
    lines = file_lines(file_path, repo_root)
    if lines is None:
        logger.info("citation dropped (no such file): %s", file_path)
        return None
    try:
        start, end = int(start_line), int(end_line)
    except (TypeError, ValueError):
        return None
    if start < 1 or end < start or start > len(lines):
        logger.info("citation dropped (bad range): %s:%s-%s", file_path, start_line, end_line)
        return None
    end = min(end, len(lines))
    body = lines[start - 1 : end]
    if expect:
        tokens = [t for t in expect if t]
        if tokens and not any(t in "\n".join(body) for t in tokens):
            logger.info(
                "citation dropped (expected %s not in %s:%d-%d)", tokens, file_path, start, end
            )
            return None
    truncated = False
    if len(body) > MAX_SNIPPET_LINES:
        # Keep the window around the first highlighted line when there is one.
        anchor = (highlight or [start])[0]
        s = max(start, min(anchor - 10, end - MAX_SNIPPET_LINES + 1))
        body = lines[s - 1 : s - 1 + MAX_SNIPPET_LINES]
        start, end, truncated = s, s + len(body) - 1, True
    return Snippet(
        file_path=file_path,
        start_line=start,
        end_line=end,
        text="\n".join(body),
        symbol=symbol,
        reason=reason,
        highlight_lines=[h for h in (highlight or []) if start <= h <= end],
        truncated=truncated,
    )
