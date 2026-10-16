import pytest


def test_ranking_and_bounds(project):
    index = project.SearchIndex({"a": "graph search", "b": "graph nodes edges"})
    hits = index.search("graph search")
    assert hits[0].document == "a"
    assert hits[0].score == pytest.approx(1)
    assert all(0 < hit.score <= 1.0000001 for hit in hits)


def test_unicode_and_punctuation(project):
    assert project.tokenize("Straße, CAFÉ! ＡＢＣ") == ["strasse", "café", "abc"]


def test_phrase_positions_and_repetition(project):
    index = project.SearchIndex({"a": "red red blue", "b": "red blue red"})
    assert index.phrase("red red") == ["a"]
    assert index.phrase("red blue") == ["a", "b"]
    assert index.phrase("blue red") == ["b"]


def test_empty_unknown_and_limits(project):
    index = project.SearchIndex({"empty": "", "a": "data"})
    assert index.search("unknown") == index.search("") == []
    assert index.search("data", 0) == []
    assert index.phrase("missing") == []
    with pytest.raises(ValueError):
        index.search("data", -1)


def test_file_loading_and_ties(project, tmp_path):
    for name in ("b.txt", "a.txt"):
        (tmp_path / name).write_text("same tokens")
    index = project.SearchIndex(project.load_documents(tmp_path))
    assert [hit.document for hit in index.search("same")] == ["a.txt", "b.txt"]
    assert index.excerpt("a.txt", "tokens", 0) == "tokens"
