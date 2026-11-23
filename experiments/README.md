# Experiments

Early prototypes kept to document how the chunking strategy evolved. Run from the project root.

| Script | What it explores |
|--------|------------------|
| `01_fixed_size_chunking.py` | Naive fixed-size (1000-char) chunks: simple, but cuts sentences and ideas mid-way. |
| `02_recursive_chunking.py` | `RecursiveCharacterTextSplitter` with 200-char overlap: splits on paragraphs/sentences first and keeps context across chunk boundaries. This is the approach used in the final pipeline. |
