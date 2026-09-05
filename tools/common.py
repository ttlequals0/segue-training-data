"""Shared helpers for segue tools."""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"
EXAMPLES_DIR = REPO_ROOT / "data" / "examples"
HOLDOUT_FILE = REPO_ROOT / "data" / "holdout.txt"
# Local-only (gitignored): feed slugs the extractor must skip entirely.
EXCLUDED_FEEDS_FILE = REPO_ROOT / ".local" / "excluded_feeds.txt"
SCHEMA_FILE = REPO_ROOT / "schema" / "example.schema.json"

EXTRACTOR_VERSION = "0.2.0"


def setup_minuspod_path(explicit: str | None = None) -> Path:
    """Put MinusPod's src/ on sys.path so ad_detector imports resolve."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("SEGUE_MINUSPOD_SRC")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO_ROOT.parent / "MinusPod" / "src")
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "ad_detector").is_dir():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    raise SystemExit(
        "MinusPod src/ not found. Pass --minuspod-src or set SEGUE_MINUSPOD_SRC.")


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def store_prompt(text: str) -> str:
    """Write a prompt to prompts/<hash>.txt if new; return its hash ref."""
    ref = sha256_text(text)
    path = PROMPTS_DIR / f"{ref.split(':', 1)[1]}.txt"
    if not path.exists():
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return ref


def load_prompt(ref: str) -> str:
    path = PROMPTS_DIR / f"{ref.split(':', 1)[1]}.txt"
    text = path.read_text(encoding="utf-8")
    if sha256_text(text) != ref:
        raise ValueError(f"prompt store corrupted: {path} does not match its hash")
    return text


def load_holdout() -> set[tuple[str, str]]:
    """Return {(slug, episode_id)} pairs that must never be extracted."""
    pairs = set()
    for line in HOLDOUT_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        slug, _, ep = line.partition("/")
        pairs.add((slug, ep))
    return pairs


def load_excluded_feeds() -> set[str]:
    if not EXCLUDED_FEEDS_FILE.exists():
        return set()
    return {line.strip() for line in EXCLUDED_FEEDS_FILE.read_text().splitlines()
            if line.strip() and not line.startswith("#")}


def iter_examples():
    """Yield (path, line_number, example_dict) across data/examples."""
    for path in sorted(EXAMPLES_DIR.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    yield path, n, json.loads(line)
