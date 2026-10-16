import json
import struct

import pytest


@pytest.mark.parametrize("text", ["", "aaaaa", "banana", "Straße 東京 🌱", "01" * 100])
def test_round_trip(project, text):
    assert project.decode(project.encode(text)) == text


def test_prefix_free_and_stable(project):
    codes = project.build_codes("abcddd")
    assert len(codes) == 4
    assert all(
        not b.startswith(a) for a in codes.values() for b in codes.values() if a != b
    )
    assert project.encode("abcddd") == project.encode("abcddd")


def test_compression_on_repetition(project):
    text = "ab" * 1000
    assert len(project.encode(text)) < len(text.encode())


def test_malformed_and_truncated(project):
    for archive in (b"", b"NOPE1234", project.encode("banana")[:-1]):
        with pytest.raises(ValueError):
            project.decode(archive)
    with pytest.raises(ValueError, match="prefix"):
        project.validate_codes({"a": "0", "b": "01"})


def test_invalid_padding(project):
    archive = bytearray(project.encode("a"))
    archive[-1] |= 1
    with pytest.raises(ValueError, match="padding"):
        project.decode(bytes(archive))


def test_unknown_prefix(project):
    header = json.dumps({"codes": {"a": "0"}, "bits": 1, "length": 1}).encode()
    archive = b"HUF1" + struct.pack(">I", len(header)) + header + b"\x80"
    with pytest.raises(ValueError, match="prefix"):
        project.decode(archive)
