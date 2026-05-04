"""Wikipedia ingestion.

Pulls plain-text extracts from the Wikipedia REST API for each entity in
``config.PEOPLE`` and ``config.PLACES``, persists them as ``data/raw/*.json``
files, and records ingestion bookkeeping in the SQLite database
``data/wikirag.sqlite`` so we can tell at-a-glance which entities are present.

We deliberately use ``requests`` against the public ``api.php`` endpoint rather
than the ``wikipedia`` PyPI package so the ingestion logic is transparent
(per the assignment's "language-native where possible" guidance).
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

from . import config


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A single ingested Wikipedia article."""

    entity: str          # canonical entity name (e.g. "Albert Einstein")
    type: str            # "person" or "place"
    title: str           # Wikipedia article title actually fetched
    url: str             # full https://en.wikipedia.org/... link
    text: str            # plain-text extract (full article)

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "type": self.type,
            "title": self.title,
            "url": self.url,
            "text": self.text,
        }

    def filename(self) -> str:
        """Stable filename for the raw JSON dump."""
        slug = (
            self.entity.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("'", "")
        )
        return f"{self.type}__{slug}.json"


# ---------------------------------------------------------------------------
# Wikipedia fetching
# ---------------------------------------------------------------------------


def _fetch_extract(title: str) -> tuple[str, str, str] | None:
    """Fetch the plain-text extract for a Wikipedia article.

    Returns ``(canonical_title, url, plain_text)`` or ``None`` if the article
    could not be located. Uses the standard MediaWiki ``api.php`` endpoint with
    ``prop=extracts`` and ``explaintext=1`` so we get clean text without HTML.
    """

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
    }
    headers = {"User-Agent": config.USER_AGENT}
    resp = requests.get(
        config.WIKIPEDIA_API, params=params, headers=headers, timeout=30
    )
    resp.raise_for_status()
    payload = resp.json()
    pages = payload.get("query", {}).get("pages", {})
    if not pages:
        return None
    # ``pages`` is keyed by pageid; -1 means "missing"
    for pageid, page in pages.items():
        if pageid == "-1" or "missing" in page:
            return None
        text = page.get("extract", "").strip()
        if not text:
            return None
        return page.get("title", title), page.get("fullurl", ""), text
    return None


# ---------------------------------------------------------------------------
# SQLite bookkeeping
# ---------------------------------------------------------------------------


def _open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            entity      TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            title       TEXT NOT NULL,
            url         TEXT NOT NULL,
            char_count  INTEGER NOT NULL,
            ingested_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _record_document(conn: sqlite3.Connection, doc: Document) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO documents
            (entity, type, title, url, char_count, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doc.entity, doc.type, doc.title, doc.url, len(doc.text), time.time()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_entities(
    entities: Iterable[str],
    entity_type: str,
    *,
    skip_existing: bool = True,
    sleep_seconds: float = 0.2,
) -> list[Document]:
    """Ingest every entity in ``entities`` as ``entity_type`` ("person" / "place").

    Skips entities whose raw JSON file already exists unless
    ``skip_existing=False``. A small sleep between requests is polite to the
    Wikipedia API.
    """

    if entity_type not in {"person", "place"}:
        raise ValueError(f"entity_type must be 'person' or 'place', got {entity_type!r}")

    conn = _open_db()
    out: list[Document] = []

    for entity in entities:
        # Determine target file and skip if already present
        slug = entity.lower().replace(" ", "_").replace("/", "_").replace("'", "")
        target = config.RAW_DIR / f"{entity_type}__{slug}.json"
        if skip_existing and target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            doc = Document(**data)
            print(f"  [skip] {entity_type}: {entity} (already present)")
            out.append(doc)
            continue

        print(f"  [fetch] {entity_type}: {entity}")
        try:
            result = _fetch_extract(entity)
        except requests.RequestException as e:
            print(f"    ! network error for {entity}: {e}")
            continue

        if result is None:
            print(f"    ! no Wikipedia article found for {entity!r}; skipping")
            continue

        title, url, text = result
        doc = Document(
            entity=entity,
            type=entity_type,
            title=title,
            url=url,
            text=text,
        )

        target.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _record_document(conn, doc)
        out.append(doc)
        time.sleep(sleep_seconds)

    conn.close()
    return out


def load_all_documents() -> list[Document]:
    """Load every ingested document from ``data/raw/`` into memory."""
    docs: list[Document] = []
    for path in sorted(Path(config.RAW_DIR).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs.append(Document(**data))
    return docs


def ingest_all(skip_existing: bool = True) -> list[Document]:
    """Ingest the full configured corpus (people + places)."""
    print("Ingesting people...")
    people_docs = ingest_entities(
        config.PEOPLE, "person", skip_existing=skip_existing
    )
    print("\nIngesting places...")
    place_docs = ingest_entities(
        config.PLACES, "place", skip_existing=skip_existing
    )
    print(
        f"\nDone. {len(people_docs)} people, {len(place_docs)} places ingested."
    )
    return people_docs + place_docs


if __name__ == "__main__":  # pragma: no cover - manual entry point
    ingest_all()
