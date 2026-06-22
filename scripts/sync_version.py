#!/usr/bin/env python
"""Single-source-of-truth version management for the WC 2026 Predictor.

VERSION (repo root) is the canonical version. This script keeps every other
place the version is written in sync with it.

Usage:
    python scripts/sync_version.py            # rewrite all targets to match VERSION
    python scripts/sync_version.py --check    # exit non-zero if any target drifts
    python scripts/sync_version.py --set X.Y  # bump VERSION to X.Y, then rewrite

Design notes:
  * Files are read/written as raw bytes (utf-8) so existing line endings are
    preserved exactly -- we never reflow a whole file just to touch one token.
  * Each target carries its own regex with named groups (pre / ver / post); only
    the `ver` token is rewritten, so surrounding text (e.g. the credits date
    " - June 2026") is untouched.
  * A missing file or unmatched pattern is a warning, never a crash -- one bad
    target can't take the whole script down.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"

# A semantic-ish version: MAJOR.MINOR with an optional .PATCH.
VERSION_RE = re.compile(r"^\d+\.\d+(?:\.\d+)?$")
# The version token as it appears embedded in target files (optional 'v' prefix).
TOKEN_RE = r"v?\d+\.\d+(?:\.\d+)?"


class Target:
    """One place the version is written.

    `pattern` must expose named groups `pre`, `ver`, `post`. Only `ver` is
    replaced; `prefix` ("" or "v") is prepended to the bare canonical version to
    form the desired token. `required=False` marks a file that may be absent
    (e.g. the gitignored local CLAUDE.md) -- absence is silent, not an error.
    """

    def __init__(self, relpath: str, pattern: re.Pattern, prefix: str = "",
                 required: bool = True, label: str | None = None):
        self.relpath = relpath
        self.path = ROOT / relpath
        self.pattern = pattern
        self.prefix = prefix
        self.required = required
        self.label = label or relpath

    def desired_token(self, version: str) -> str:
        return f"{self.prefix}{version}"


TARGETS = [
    # CITATION.cff -> version: "X.Y"  (keep the quotes, no 'v' prefix)
    Target(
        "CITATION.cff",
        re.compile(r'(?P<pre>^version:\s*")(?P<ver>' + TOKEN_RE + r')(?P<post>"\s*$)', re.M),
        prefix="",
        label='CITATION.cff (version: field)',
    ),
    # app/streamlit_app.py -> <p class="credits-version">vX.Y - June 2026</p>
    # Match only the version token right after the credits-version marker; the
    # date that follows is preserved.
    Target(
        "app/streamlit_app.py",
        re.compile(r'(?P<pre>class="credits-version">)(?P<ver>' + TOKEN_RE + r')(?P<post>)'),
        prefix="v",
        label="app/streamlit_app.py (credits badge)",
    ),
    # CLAUDE.md (optional, gitignored local file) -> **Version: vX.Y**
    Target(
        "CLAUDE.md",
        re.compile(r'(?P<pre>\*\*Version:\s*)(?P<ver>' + TOKEN_RE + r')(?P<post>\*\*)'),
        prefix="v",
        required=False,
        label="CLAUDE.md (Version: header)",
    ),
    # pyproject.toml -> version = "X.Y"  (project table, no 'v' prefix)
    Target(
        "pyproject.toml",
        re.compile(r'(?P<pre>^version\s*=\s*")(?P<ver>' + TOKEN_RE + r')(?P<post>"\s*$)', re.M),
        prefix="",
        label="pyproject.toml (project.version)",
    ),
]


def read_version() -> str:
    if not VERSION_FILE.exists():
        print(f"ERROR: canonical version file not found: {VERSION_FILE}", file=sys.stderr)
        sys.exit(2)
    version = VERSION_FILE.read_text(encoding="utf-8").strip()
    if not VERSION_RE.match(version):
        print(f"ERROR: VERSION content {version!r} is not a valid X.Y[.Z] version",
              file=sys.stderr)
        sys.exit(2)
    return version


def _read_text(path: Path) -> str:
    # Bytes -> str so we can rewrite without touching line endings on save.
    return path.read_bytes().decode("utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_bytes(text.encode("utf-8"))


def process_target(target: Target, version: str, *, check: bool) -> tuple[bool, bool]:
    """Return (ok, changed).

    ok=False means this target is out of sync (check mode) and should fail the
    run. changed=True means rewrite mode actually edited the file. Missing
    optional files / unmatched patterns warn and count as ok.
    """
    if not target.path.exists():
        if target.required:
            print(f"WARNING: target file not found, skipping: {target.relpath}")
        return True, False

    try:
        text = _read_text(target.path)
    except OSError as exc:
        print(f"WARNING: could not read {target.relpath}: {exc}")
        return True, False

    match = target.pattern.search(text)
    if not match:
        if target.required:
            print(f"WARNING: version pattern not found in {target.relpath}; skipping")
        return True, False

    current = match.group("ver")
    desired = target.desired_token(version)

    if current == desired:
        return True, False

    if check:
        print(f"  DRIFT  {target.label}: found {current!r}, expected {desired!r}")
        return False, False

    new_text = (
        text[: match.start("ver")] + desired + text[match.end("ver"):]
    )
    _write_text(target.path, new_text)
    print(f"  UPDATED {target.label}: {current!r} -> {desired!r}")
    return True, True


def update_date_released(version: str) -> None:
    """On a release bump, set CITATION.cff's date-released to today (UTC)."""
    path = ROOT / "CITATION.cff"
    if not path.exists():
        print("WARNING: CITATION.cff not found; cannot update date-released")
        return
    text = _read_text(path)
    today = datetime.date.today().isoformat()
    pat = re.compile(r'(?P<pre>^date-released:\s*)(?P<date>\d{4}-\d{2}-\d{2})(?P<post>\s*$)', re.M)
    m = pat.search(text)
    if not m:
        print("WARNING: date-released field not found in CITATION.cff; leaving as-is")
        return
    if m.group("date") == today:
        return
    new_text = text[: m.start("date")] + today + text[m.end("date"):]
    _write_text(path, new_text)
    print(f"  UPDATED CITATION.cff (date-released): {m.group('date')} -> {today}")


def run_check(version: str) -> int:
    print(f"sync_version --check: canonical VERSION = {version}")
    all_ok = True
    for target in TARGETS:
        ok, _ = process_target(target, version, check=True)
        all_ok = all_ok and ok
    if all_ok:
        print("All version strings are in sync with VERSION.")
        return 0
    print("\nVersion drift detected. Fix with:  python scripts/sync_version.py")
    return 1


def run_rewrite(version: str) -> int:
    print(f"sync_version: rewriting all targets to VERSION = {version}")
    changed_any = False
    for target in TARGETS:
        _, changed = process_target(target, version, check=False)
        changed_any = changed_any or changed
    if not changed_any:
        print("Nothing to do -- everything already matches VERSION.")
    return 0


def run_set(new_version: str) -> int:
    new_version = new_version.lstrip("v").strip()
    if not VERSION_RE.match(new_version):
        print(f"ERROR: --set value {new_version!r} is not a valid X.Y[.Z] version",
              file=sys.stderr)
        return 2
    old_version = read_version()
    _write_text(VERSION_FILE, new_version + "\n")
    print(f"VERSION bumped: {old_version} -> {new_version}")
    rc = run_rewrite(new_version)
    update_date_released(new_version)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true",
                       help="exit non-zero if any target is out of sync (no writes)")
    group.add_argument("--set", metavar="X.Y", dest="set_version",
                       help="bump VERSION to X.Y, then rewrite all targets")
    args = parser.parse_args(argv)

    if args.set_version:
        return run_set(args.set_version)

    version = read_version()
    if args.check:
        return run_check(version)
    return run_rewrite(version)


if __name__ == "__main__":
    raise SystemExit(main())
