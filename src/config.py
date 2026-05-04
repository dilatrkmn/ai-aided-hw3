"""Project-wide configuration: entity lists, paths, model names, hyperparameters.

The required minimum set from the assignment is included plus a handful of
extras to comfortably exceed the "20 people / 20 places" floor.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"
SQLITE_PATH = DATA_DIR / "wikirag.sqlite"

for _d in (DATA_DIR, RAW_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Entities to ingest
# ---------------------------------------------------------------------------
# The first ten of each list are the assignment's required minimum set.
# We add a handful of extras so the corpus exceeds the "20 of each" floor and
# can support more interesting mixed / comparison queries.

PEOPLE: list[str] = [
    # Required
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Extras (bringing the count to >= 20)
    "Isaac Newton",
    "Charles Darwin",
    "Mahatma Gandhi",
    "Vincent van Gogh",
    "Pablo Picasso",
    "Mustafa Kemal Atatürk",
    "Mozart",
    "Ludwig van Beethoven",
    "Stephen Hawking",
    "Alan Turing",
]

PLACES: list[str] = [
    # Required
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",  # Wikipedia title for "Pyramids of Giza"
    "Mount Everest",
    # Extras
    "Stonehenge",
    "Petra",
    "Acropolis of Athens",
    "Sagrada Família",
    "Burj Khalifa",
    "Mount Fuji",
    "Niagara Falls",
    "Cappadocia",
    "Angkor Wat",
    "Sydney Opera House",
]

# ---------------------------------------------------------------------------
# Models / hyperparameters
# ---------------------------------------------------------------------------

# Local sentence-transformers embedding model. 384-dim, fast on CPU.
EMBEDDING_MODEL_NAME = os.getenv(
    "WIKIRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Default Ollama generation model. Override via env or Streamlit sidebar.
LLM_MODEL_NAME = os.getenv("WIKIRAG_LLM_MODEL", "llama3.2:3b")

# Where Ollama exposes its HTTP API.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Chunking
CHUNK_SIZE_WORDS = 220        # ~ 1k characters; comfortably fits embedder ctx
CHUNK_OVERLAP_WORDS = 40      # ~18% overlap so concepts that span boundaries
                              # remain retrievable from either chunk

# Retrieval
TOP_K = 5                     # chunks per query (per type, when both)
TOP_K_PER_TYPE_WHEN_BOTH = 3  # for mixed queries we take 3 person + 3 place

# Chroma collection name
COLLECTION_NAME = "wiki_rag"

# Wikipedia REST endpoint for plain-text extracts
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "BLG483E-HW3-LocalWikipediaRAG/1.0 (educational use)"
