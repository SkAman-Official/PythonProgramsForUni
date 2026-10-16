# Positional text search

I wanted to retrieve local notes. I normalize Unicode, keep token positions and build sparse TF-IDF document vectors. Queries use cosine similarity, while phrase matching intersects positions at the required offsets. Equal scores are ordered by document name.

From the repository root, after the main setup:

```bash
python projects/04_text_search/main.py
pytest projects/04_text_search/tests -q
```

Example output:

```text
graphs 0.488 | graph search finds paths graph
hybrid 0.443 | graph search and machine learning
```

The default query ranks the graph note ahead of the mixed-topic note. I test Unicode case folding, repeated-word phrases, unknown terms, empty documents and stable ties. I can search my own direct UTF-8 text files with `--directory notes --query "graph search"`; adding `--phrase` requires consecutive tokens. This is an in-memory index for small collections. It has no stemming, language detection or incremental updates, and snippets contain normalized tokens rather than original formatting.
