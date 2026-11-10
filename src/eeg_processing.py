"""
NeuroCalm - funzioni di supporto per il dataset EEG (emotions.csv).

Per ora contiene SOLO gli strumenti usati nella fase di EDA:
    - caricamento del CSV e pulizia dei nomi di colonna
    - "anatomia" delle colonne (famiglia di feature, metà finestra a/b)
    - ricerca delle colonne duplicate (identiche)

La pulizia vera e propria e la feature selection le aggiungeremo
insieme nel passo successivo.
"""

import os
import re

import numpy as np
import pandas as pd

# Ordine fisso delle classi: NEGATIVE è la classe "ansia/disagio" che ci interessa
CLASS_ORDER = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
CLASS_COLORS = {"NEGATIVE": "#e74c3c", "NEUTRAL": "#7f8c8d", "POSITIVE": "#2ecc71"}


def load_dataset(csv_path: str = "data/emotions.csv") -> pd.DataFrame:
    """Carica emotions.csv e pulisce i nomi di colonna.

    La prima colonna nel file si chiama '# mean_0_a' (il cancelletto è un
    residuo dell'export): lo togliamo per avere nomi uniformi.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.replace("#", "").strip() for c in df.columns]
    return df


def feature_family(col: str) -> str:
    """Restituisce la 'famiglia' di una colonna.

    Esempi: 'mean_d_3_a' -> 'mean_d', 'fft_12_b' -> 'fft',
            'entropy0_a' -> 'entropy', 'covmat_5_a' -> 'covmat'.
    """
    if col.startswith("entropy"):
        return "entropy"
    return re.sub(r"_\d+_[ab]2?$", "", col)


def window_half(col: str) -> str:
    """Le feature sono calcolate su due metà della finestra temporale: 'a' e 'b'."""
    return "a" if re.search(r"_a2?$", col) else "b"


def column_anatomy(columns) -> pd.DataFrame:
    """Tabella con famiglia e metà-finestra di ogni colonna (utile per i conteggi)."""
    cols = [c for c in columns if c != "label"]
    return pd.DataFrame({
        "colonna": cols,
        "famiglia": [feature_family(c) for c in cols],
        "meta": [window_half(c) for c in cols],
    })


def find_duplicate_columns(X: pd.DataFrame) -> list:
    """Trova le colonne che sono copie IDENTICHE di una colonna precedente.

    Ritorna una lista di coppie (colonna_duplicata, colonna_originale).
    Usiamo un hash dei valori: due colonne con lo stesso hash sono identiche.
    """
    seen = {}
    pairs = []
    for col in X.columns:
        key = hash(np.ascontiguousarray(X[col].values).tobytes())
        if key in seen:
            pairs.append((col, seen[key]))
        else:
            seen[key] = col
    return pairs

