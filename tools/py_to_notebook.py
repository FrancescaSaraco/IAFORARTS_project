"""
Converte uno script Python "a celle" in un Jupyter Notebook.

Convenzione (la stessa di VS Code / Jupytext):
    # %%               -> inizio di una cella di codice
    # %% [markdown]    -> inizio di una cella di testo (righe commentate con '# ')

Così lo script .py e il notebook .ipynb hanno sempre lo stesso contenuto.
Uso:  python tools/py_to_notebook.py 01_eda.py notebooks/01_eda.ipynb
"""

import sys
from pathlib import Path

import nbformat


def converti(py_path: str, ipynb_path: str) -> None:
    righe = Path(py_path).read_text(encoding="utf-8").splitlines()
    celle, tipo, buffer = [], None, []

    def chiudi_cella():
        testo = "\n".join(buffer).strip("\n")
        if not testo:
            return
        if tipo == "markdown":
            md = "\n".join(r[2:] if r.startswith("# ") else r.lstrip("#") for r in testo.splitlines())
            celle.append(nbformat.v4.new_markdown_cell(md))
        else:
            celle.append(nbformat.v4.new_code_cell(testo))

    for riga in righe:
        if riga.startswith("# %%"):
            chiudi_cella()
            buffer = []
            tipo = "markdown" if "[markdown]" in riga else "code"
            # il titolo della cella di codice diventa un commento in testa
            titolo = riga[4:].strip()
            if tipo == "code" and titolo:
                buffer.append(f"# {titolo}")
        else:
            buffer.append(riga)
    chiudi_cella()

    nb = nbformat.v4.new_notebook(cells=celle)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    Path(ipynb_path).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, ipynb_path)
    print(f"Notebook creato: {ipynb_path} ({len(celle)} celle)")


if __name__ == "__main__":
    converti(sys.argv[1], sys.argv[2])

