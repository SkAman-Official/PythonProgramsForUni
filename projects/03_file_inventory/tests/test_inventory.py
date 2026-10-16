import json

import pytest


def test_changes(project, tmp_path):
    (tmp_path / "old").write_text("a")
    (tmp_path / "changed").write_text("abc")
    before = project.scan(tmp_path)
    (tmp_path / "old").unlink()
    (tmp_path / "new").write_bytes(b"")
    (tmp_path / "changed").write_text("xyz")
    diff = project.compare(before, project.scan(tmp_path))
    assert (diff.added, diff.removed, diff.modified) == (
        ("new",),
        ("old",),
        ("changed",),
    )


def test_json_round_trip(project, tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "é.txt").write_text("hello")
    inventory = project.scan(root)
    destination = tmp_path / "snapshot.json"
    project.save_inventory(inventory, destination)
    assert project.load_inventory(destination) == inventory
    assert not list(tmp_path.glob(".inventory-*"))


def test_skip_symlinks(project, tmp_path):
    (tmp_path / "file").write_bytes(b"123")
    (tmp_path / "link").symlink_to(tmp_path / "file")
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert set(project.scan(tmp_path)) == {"file"}


def test_invalid_snapshot_and_empty(project, tmp_path):
    assert project.scan(tmp_path) == {}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "files": {"../escape": {}}}))
    with pytest.raises(ValueError, match="Unsafe"):
        project.load_inventory(path)
    with pytest.raises(ValueError):
        project.scan(path)


def test_known_digest(project, tmp_path):
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert project.digest_file(path).sha256 == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_scan_reports_unreadable_directory(project, tmp_path, monkeypatch):
    def failing_walk(root, *, followlinks, onerror):
        onerror(PermissionError("unreadable child"))
        return iter(())

    monkeypatch.setattr(project.os, "walk", failing_walk)
    with pytest.raises(PermissionError, match="unreadable"):
        project.scan(tmp_path)
