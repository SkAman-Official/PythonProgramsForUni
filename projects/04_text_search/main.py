"""Search local UTF-8 documents with a positional TF-IDF index."""

from __future__ import annotations

import argparse
import logging
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """Normalize Unicode and return case-insensitive alphanumeric tokens."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


@dataclass(frozen=True)
class Hit:
    """Associate a document identifier with its cosine similarity."""

    document: str
    score: float


class SearchIndex:
    """Maintain document vectors and token positions for a small corpus."""

    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = dict(documents)
        self.positions: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self.counts: dict[str, Counter[str]] = {}
        for name, text in self.documents.items():
            tokens = tokenize(text)
            self.counts[name] = Counter(tokens)
            for position, token in enumerate(tokens):
                self.positions[token].setdefault(name, []).append(position)
        self.idf = {
            token: math.log((1 + len(documents)) / (1 + len(postings))) + 1
            for token, postings in self.positions.items()
        }
        self.vectors = {
            name: self._vector(counts) for name, counts in self.counts.items()
        }
        self.norms = {
            name: math.sqrt(sum(weight * weight for weight in vector.values()))
            for name, vector in self.vectors.items()
        }

    def _vector(self, counts: Counter[str]) -> dict[str, float]:
        return {
            token: (1 + math.log(count)) * self.idf[token]
            for token, count in counts.items()
            if token in self.idf
        }

    def search(self, query: str, limit: int = 5) -> list[Hit]:
        """Rank documents by cosine similarity, breaking ties by name."""
        if limit < 0:
            raise ValueError("Limit must be nonnegative")
        vector = self._vector(Counter(tokenize(query)))
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if not norm or not limit:
            return []
        candidates = set()
        for token in vector:
            candidates.update(self.positions[token])
        hits = []
        for name in candidates:
            score = sum(
                weight * self.vectors[name].get(token, 0)
                for token, weight in vector.items()
            )
            if self.norms[name]:
                hits.append(Hit(name, score / (norm * self.norms[name])))
        return sorted(hits, key=lambda hit: (-hit.score, hit.document))[:limit]

    def phrase(self, query: str) -> list[str]:
        """Find documents containing the exact normalized token sequence."""
        tokens = tokenize(query)
        if not tokens or any(token not in self.positions for token in tokens):
            return []
        candidates = set(self.positions[tokens[0]])
        for token in tokens[1:]:
            candidates.intersection_update(self.positions[token])
        matches = []
        for name in sorted(candidates):
            starts = set(self.positions[tokens[0]][name])
            for offset, token in enumerate(tokens[1:], 1):
                starts.intersection_update(
                    position - offset for position in self.positions[token][name]
                )
            if starts:
                matches.append(name)
        return matches

    def excerpt(self, name: str, query: str, radius: int = 4) -> str:
        """Return normalized context surrounding the first matching token."""
        if radius < 0:
            raise ValueError("Radius must be nonnegative")
        tokens = tokenize(self.documents[name])
        query_tokens = set(tokenize(query))
        location = next(
            (i for i, token in enumerate(tokens) if token in query_tokens), 0
        )
        return " ".join(tokens[max(0, location - radius) : location + radius + 1])


def load_documents(directory: Path) -> dict[str, str]:
    """Read direct .txt children of a directory in deterministic order."""
    if not directory.is_dir():
        raise ValueError("Expected a document directory")
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(directory.glob("*.txt"))
        if path.is_file() and not path.is_symlink()
    }


def demo_documents() -> dict[str, str]:
    """Return a small corpus with overlapping topics."""
    return {
        "graphs": "Graph search finds paths. Graph algorithms use nodes and edges.",
        "learning": "Machine learning fits models. Learning needs careful evaluation.",
        "hybrid": "Graph search and machine learning can share representations.",
    }


def main() -> None:
    """Search an optional local corpus or the built-in example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--query", default="graph search")
    parser.add_argument("--phrase", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    documents = load_documents(args.directory) if args.directory else demo_documents()
    index = SearchIndex(documents)
    if args.phrase:
        LOGGER.info("Phrase matches: %s", ", ".join(index.phrase(args.query)))
    else:
        for hit in index.search(args.query, args.limit):
            LOGGER.info(
                "%s %.3f | %s",
                hit.document,
                hit.score,
                index.excerpt(hit.document, args.query),
            )


if __name__ == "__main__":
    main()
