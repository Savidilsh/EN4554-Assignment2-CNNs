"""Copy the markdown cells of the percent-format source into an already-executed notebook,
keeping all code cells and their outputs. The cell structure must be identical.

Usage: python tools/sync_markdown.py a02_cnn.py a02_cnn.ipynb
"""
import sys

import nbformat

from build_notebook import parse_cells


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        new = parse_cells(f.read())
    nb = nbformat.read(dst, as_version=4)
    assert len(new) == len(nb.cells), f"cell count differs: {len(new)} vs {len(nb.cells)}"
    changed = 0
    for a, b in zip(new, nb.cells):
        assert a.cell_type == b.cell_type, "cell types differ"
        if b.cell_type == "markdown" and a.source != b.source:
            b.source = a.source
            changed += 1
        elif b.cell_type == "code" and a.source != b.source:
            print("warning: code cell differs from source (left unchanged):", b.source[:60].replace("\n", " "))
    nbformat.write(nb, dst)
    print(f"updated {changed} markdown cells in {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
