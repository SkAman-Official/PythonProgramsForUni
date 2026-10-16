"""Compress Unicode strings with deterministic Huffman prefix codes."""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)
MAGIC = b"HUF1"
MAX_HEADER = 8 * 1024 * 1024


@dataclass(frozen=True)
class Node:
    """Represent a leaf symbol or a pair of child nodes."""

    symbol: str | None = None
    left: Node | None = None
    right: Node | None = None


def build_codes(text: str) -> dict[str, str]:
    """Construct a stable codebook, including the single-symbol case."""
    heap: list[tuple[int, int, Node]] = []
    for sequence, (symbol, count) in enumerate(sorted(Counter(text).items())):
        heap.append((count, sequence, Node(symbol)))
    heapq.heapify(heap)
    sequence = len(heap)
    while len(heap) > 1:
        left_count, _, left = heapq.heappop(heap)
        right_count, _, right = heapq.heappop(heap)
        heapq.heappush(
            heap, (left_count + right_count, sequence, Node(left=left, right=right))
        )
        sequence += 1
    codes = {}
    stack = [(heap[0][2], "")] if heap else []
    while stack:
        node, prefix = stack.pop()
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
        else:
            stack.append((node.right, prefix + "1"))
            stack.append((node.left, prefix + "0"))
    return codes


def encode(text: str) -> bytes:
    """Pack a JSON codebook and bit payload into a versioned archive."""
    codes = build_codes(text)
    bits = "".join(codes[symbol] for symbol in text)
    header = json.dumps(
        {"codes": codes, "bits": len(bits), "length": len(text)},
        ensure_ascii=True,
        sort_keys=True,
    ).encode("ascii")
    if len(header) > MAX_HEADER:
        raise ValueError("Codebook exceeds supported header size")
    padded = bits + "0" * (-len(bits) % 8)
    payload = bytes(int(padded[i : i + 8], 2) for i in range(0, len(padded), 8))
    return MAGIC + struct.pack(">I", len(header)) + header + payload


def validate_codes(codes: dict[str, str]) -> dict:
    """Build a decoding trie and reject duplicate or prefix-conflicting codes."""
    root: dict = {}
    for symbol, code in codes.items():
        if len(symbol) != 1 or not isinstance(code, str) or not code:
            raise ValueError("Invalid codebook entry")
        if any(bit not in "01" for bit in code):
            raise ValueError("Codes must contain binary digits")
        node = root
        for bit in code:
            if "symbol" in node:
                raise ValueError("Codebook is not prefix-free")
            node = node.setdefault(bit, {})
        if node:
            raise ValueError("Codebook is not prefix-free")
        node["symbol"] = symbol
    return root


def decode(archive: bytes) -> str:
    """Validate an archive and reconstruct exactly the declared text length."""
    if len(archive) < 8 or archive[:4] != MAGIC:
        raise ValueError("Invalid archive signature")
    size = struct.unpack(">I", archive[4:8])[0]
    if size > MAX_HEADER or size > len(archive) - 8:
        raise ValueError("Invalid header size")
    try:
        header = json.loads(archive[8 : 8 + size])
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid archive header") from error
    if not isinstance(header, dict) or not isinstance(header.get("codes"), dict):
        raise ValueError("Invalid codebook")
    bit_count, length = header.get("bits"), header.get("length")
    if type(bit_count) is not int or type(length) is not int:
        raise ValueError("Lengths must be integers")
    payload = archive[8 + size :]
    if bit_count < 0 or length < 0 or len(payload) != (bit_count + 7) // 8:
        raise ValueError("Payload length mismatch")
    root = validate_codes(header["codes"])
    if bit_count % 8 and payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise ValueError("Nonzero padding")
    node = root
    output = []
    for position in range(bit_count):
        bit = str((payload[position // 8] >> (7 - position % 8)) & 1)
        if bit not in node:
            raise ValueError("Unknown encoded prefix")
        node = node[bit]
        if "symbol" in node:
            output.append(node["symbol"])
            node = root
    if node is not root or len(output) != length:
        raise ValueError("Truncated or inconsistent payload")
    return "".join(output)


def main() -> None:
    """Run a round trip or encode/decode an explicitly named file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.input:
        if args.output is None or args.input.resolve() == args.output.resolve():
            parser.error("Choose a separate --output path")
        if args.decode:
            args.output.write_text(decode(args.input.read_bytes()), encoding="utf-8")
        else:
            args.output.write_bytes(encode(args.input.read_text(encoding="utf-8")))
    else:
        text = "banana bandana " * 100
        archive = encode(text)
        assert decode(archive) == text
        LOGGER.info(
            "Round trip: %d characters; %d archive bytes", len(text), len(archive)
        )


if __name__ == "__main__":
    main()
