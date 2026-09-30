"""Enumerate indexable JavaScript files and record why everything else was skipped."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)


@dataclass
class WalkResult:
    files: list[Path]
    skipped: list[dict]


def _is_minified(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lines = text.count("\n") + 1
    return len(text) / lines > config.MINIFIED_AVG_LINE_LEN


def walk_repo(repo_root: Path = config.REPO_ROOT) -> WalkResult:
    """Return JS files under the configured include dirs, plus a skip record for each exclusion.

    Only files with a JS extension are considered; excluded-directory contents are skipped
    silently per directory (one skip record per directory) to keep the manifest readable.
    """
    files: list[Path] = []
    skipped: list[dict] = []
    for include in config.INCLUDE_DIRS:
        base = repo_root / include
        if not base.is_dir():
            skipped.append({"path": include, "reason": "include dir missing"})
            continue
        stack = [base]
        while stack:
            current = stack.pop()
            for entry in sorted(current.iterdir()):
                rel = entry.relative_to(repo_root).as_posix()
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if entry.name in config.EXCLUDE_DIR_NAMES:
                        skipped.append({"path": rel + "/", "reason": "excluded directory"})
                    else:
                        stack.append(entry)
                    continue
                if entry.suffix not in config.JS_EXTENSIONS:
                    continue
                if entry.name.endswith(".min.js"):
                    skipped.append({"path": rel, "reason": "minified (.min.js)"})
                elif entry.stat().st_size > config.MAX_FILE_BYTES:
                    skipped.append({"path": rel, "reason": "too large"})
                elif _is_minified(entry):
                    skipped.append({"path": rel, "reason": "minified (long lines)"})
                else:
                    files.append(entry)
    files.sort()
    logger.info("walker: %d files to index, %d skip records", len(files), len(skipped))
    return WalkResult(files=files, skipped=skipped)
