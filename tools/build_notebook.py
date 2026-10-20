"""Convert a percent-format script (``# %%`` / ``# %% [markdown]`` cells) into a Jupyter notebook.

Usage: python tools/build_notebook.py a02_cnn.py a02_cnn.ipynb
"""
import sys

import nbformat


def parse_cells(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in lines]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines)))
        elif lines:
            cells.append(nbformat.v4.new_code_cell("\n".join(lines)))

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()
    return cells


def main(src, dst):
    with open(src, encoding="utf-8") as f:
        cells = parse_cells(f.read())
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["accelerator"] = "GPU"
    nbformat.write(nb, dst)
    print(f"wrote {dst} ({len(cells)} cells)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
