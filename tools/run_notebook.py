"""Execute a notebook in place, saving it after every cell (and every 60 s during long cells),
so that code and outputs can be inspected while a multi-hour run is in progress.

Usage: python tools/run_notebook.py a02_cnn.ipynb [kernel_name]
"""
import copy
import os
import sys
import threading

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def save(nb, path):
    tmp = path + ".tmp"
    nbformat.write(copy.deepcopy(nb), tmp)
    os.replace(tmp, path)


def main(path, kernel_name="python3"):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:  # start from a clean notebook
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    client = NotebookClient(nb, timeout=None, kernel_name=kernel_name,
                            resources={"metadata": {"path": os.path.dirname(os.path.abspath(path))}})
    stop = threading.Event()

    def autosave():
        while not stop.wait(60):
            try:
                save(nb, path)
            except Exception:  # outputs may change while copying; try again next minute
                pass

    threading.Thread(target=autosave, daemon=True).start()
    code = 0
    try:
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                if cell.cell_type != "code":
                    continue
                print(f"executing cell {i}", flush=True)
                client.execute_cell(cell, i)
                save(nb, path)
    except CellExecutionError as e:
        print(f"cell failed: {e}", flush=True)
        code = 1
    finally:
        stop.set()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        save(nb, path)
    sys.exit(code)


if __name__ == "__main__":
    main(*sys.argv[1:3])
