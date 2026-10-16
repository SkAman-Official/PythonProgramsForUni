# Huffman codec

I wanted to connect prefix trees with a binary file format I could inspect. I count Unicode symbols, merge the lightest nodes and pack their code bits into bytes. The archive stores a version marker and JSON codebook, so decoding does not need the original text.

From the repository root, after the main setup:

```bash
python projects/05_huffman_codec/main.py
pytest projects/05_huffman_codec/tests -q
```

Example output:

```text
Round trip: 1500 characters; 508 archive bytes
```

The repeated sample compresses to 508 bytes including its header. I test empty text, one-symbol input, Unicode, prefix conflicts, truncation and invalid padding. To encode a file I add `--input notes.txt --output notes.huf`; decoding adds `--decode` and chooses a separate text output. I keep the whole message in memory, and small inputs may grow because of metadata. This format has structural validation but no checksum, so a valid-looking bit change may go undetected.
