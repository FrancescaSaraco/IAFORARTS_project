# %% [markdown]
# # NeuroCalm — Step 1a: Analisi Esplorativa (EDA) di `emotions.csv`
#
# **Obiettivo di questa fase:** capire *com'è fatto* il dataset prima di toccarlo.
# Non rimuoviamo ancora nulla: raccogliamo le prove che useremo nella fase
# successiva (pulizia + feature selection) per decidere *cosa* togliere e *perché*.
#
# **Il dataset in breve** (J. J. Bird et al., caschetto EEG Muse):
# - ogni riga è una **finestra temporale** di segnale EEG già trasformata in feature statistiche;
# - etichetta `label` ∈ {NEGATIVE, NEUTRAL, POSITIVE} (lo stimolo emotivo era un video);
# - le feature sono calcolate separatamente sulle due **metà della finestra** (suffisso `_a` e `_b`).

# %% Setup: percorsi, librerie, stile dei grafici
import os
import sys
from pathlib import Path

# Funziona sia lanciato dalla radice del progetto sia dal notebook in notebooks/
ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # da terminale: niente finestre, salviamo solo i PNG

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.eeg_processing import (
    CLASS_COLORS,
    CLASS_ORDER,
    column_anatomy,
    find_duplicate_columns,
    load_dataset,
)

FIG_DIR = Path("reports/figures/eda")
FIG_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def mostra(fig, nome_file):
    """Salva il grafico in reports/figures/eda e lo mostra (solo nel notebook)."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / nome_file, dpi=150)
    print(f"[grafico salvato] {FIG_DIR / nome_file}")
    if IN_NOTEBOOK:
        plt.show()
    else:
        plt.close(fig)


# %% [markdown]
# ## 1. Caricamento e dimensioni
# Il file pesa ~49 MB su disco e ~44 MB in memoria: **non è un dataset pesante**,
# pandas lo carica in un paio di secondi. Il problema non è il peso ma la *forma*:
# ci sono più colonne che righe.

# %% 1. Caricamento
df = load_dataset("data/emotions.csv")
X = df.drop(columns="label")
y = df["label"]

print(f"Righe (finestre EEG): {df.shape[0]}")
print(f"Feature:              {X.shape[1]}")
print(f"Memoria occupata:     {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Tipi di dato:         {X.dtypes.value_counts().to_dict()}")
print(f"Rapporto feature/righe: {X.shape[1] / X.shape[0]:.2f}  (>1 = rischio overfitting)")

# %% [markdown]
# ## 2. Qualità dei dati: valori mancanti, infiniti, duplicati

# %% 2. Qualità
print(f"Valori NaN:                 {X.isna().sum().sum()}")
print(f"Valori infiniti:            {np.isinf(X.values).sum()}")
print(f"Righe duplicate:            {df.duplicated().sum()}")
print(f"Colonne costanti:           {(X.nunique() <= 1).sum()}")

# %% [markdown]
# ## 3. Distribuzione delle classi
# Se le classi fossero sbilanciate dovremmo usare pesi o ricampionamento.

# %% 3. Classi
conteggi = y.value_counts().reindex(CLASS_ORDER)
print(conteggi.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(conteggi.index, conteggi.values,
       color=[CLASS_COLORS[c] for c in conteggi.index], edgecolor="black")
for i, v in enumerate(conteggi.values):
    ax.text(i, v / 2, f"{v}\n({v / len(y):.1%})", ha="center", color="white", fontweight="bold")
ax.set_title("Distribuzione delle classi emotive")
ax.set_ylabel("Numero di finestre")
mostra(fig, "01_distribuzione_classi.png")

# %% [markdown]
# ## 4. Anatomia delle colonne
# I nomi delle colonne seguono lo schema `famiglia_indice_metà`.
# Raggruppiamoli per famiglia per capire da dove vengono le 2.548 feature.
#
# | Famiglia | Significato |
# |---|---|
# | `mean`, `stddev`, `max`, `min` | statistiche del segnale per ciascuno dei 5 canali |
# | `moments` | momenti statistici di ordine superiore (asimmetria, curtosi) |
# | `*_d`, `*_q` | differenze tra sotto-finestre (quarti di finestra): come cambia il segnale nel tempo |
# | `covmat` | matrice di covarianza tra i segnali |
# | `eigen` | autovalori della matrice di covarianza |
# | `logm` | logaritmo di matrice della covarianza (parte triangolare) |
# | `entropy` | entropia del segnale per canale |
# | `correlate` | correlazioni tra segnali |
# | `fft` | coefficienti della trasformata di Fourier |

# %% 4. Anatomia
anat = column_anatomy(X.columns)
tab = anat.pivot_table(index="famiglia", columns="meta", values="colonna", aggfunc="count")
tab["totale"] = tab.sum(axis=1)
tab = tab.sort_values("totale", ascending=False)
print(tab.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
tab["totale"].sort_values().plot.barh(ax=ax, color="#3498db", edgecolor="black")
ax.set_title("Numero di colonne per famiglia di feature")
ax.set_xlabel("Numero di colonne")
mostra(fig, "02_colonne_per_famiglia.png")
print(f"\nLe sole FFT sono {tab.loc['fft', 'totale']} colonne su {X.shape[1]} "
      f"({tab.loc['fft', 'totale'] / X.shape[1]:.0%} del dataset).")

# %% [markdown]
# ## 5. Colonne duplicate (ridondanza "perfetta")
# Prima di parlare di correlazione, cerchiamo colonne **identiche** al 100%:
# sono informazione ripetuta, eliminarle non perde nulla.

# %% 5. Colonne duplicate
coppie_dup = find_duplicate_columns(X)
dup_df = pd.DataFrame(coppie_dup, columns=["duplicata", "originale"])
dup_df["famiglia"] = column_anatomy(dup_df["duplicata"])["famiglia"].values

print(f"Colonne identiche a un'altra: {len(dup_df)} su {X.shape[1]}")
print(dup_df["famiglia"].value_counts().to_string())
print("\nEsempi:")
print(dup_df.groupby("famiglia").head(3).to_string(index=False))

# %% [markdown]
# **Perché esistono?**
# - **`covmat` (132 duplicati):** una matrice di covarianza è *simmetrica*
#   (cov(i,j) = cov(j,i)). Ogni metà ha una matrice 12×12 = 144 valori, di cui solo
#   78 distinti: gli altri 66 sono la copia speculare. 66 × 2 metà = 132.
# - **`fft` (600 duplicati):** la trasformata di Fourier di un segnale reale è
#   *simmetrica* (coniugata). Lo vediamo meglio nella sezione seguente.

# %% [markdown]
# ## 6. Il caso FFT: si possono leggere le onde cerebrali θ, α, β, γ?
# Era l'idea iniziale: usare le colonne `fft_*` come uno spettro di frequenza e
# ricavarne la potenza nelle bande θ (4–8 Hz), α (8–13 Hz), β (13–30 Hz), γ (>30 Hz).
# Controlliamo se i dati lo permettono.

# %% 6. Struttura delle FFT
fft_a = X[[f"fft_{i}_a" for i in range(750)]].values
print(f"Valori FFT negativi: {(fft_a < 0).mean():.1%}")

gruppi = fft_a.reshape(len(X), 150, 5)  # 750 colonne = 150 gruppi da 5
for p in range(5):
    print(f"posizione {p} nel gruppo -> media {gruppi[:, :, p].mean():8.2f}   "
          f"std {gruppi[:, :, p].std():7.2f}")

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(range(40), fft_a[0, :40], "o-", color="#34495e")
for k in range(0, 40, 5):
    ax.axvspan(k - 0.5, k + 4.5, color="#3498db" if (k // 5) % 2 else "#ecf0f1", alpha=0.25)
ax.set_title("Prime 40 colonne fft_*_a di una finestra: lo schema si ripete a gruppi di 5")
ax.set_xlabel("indice colonna fft")
ax.set_ylabel("valore")
mostra(fig, "03_struttura_fft.png")

# %% [markdown]
# **Cosa ci dicono questi numeri:**
# 1. Circa metà dei valori è **negativa** → non sono potenze/ampiezze spettrali
#    (una potenza è sempre ≥ 0): sono la parte reale di coefficienti complessi.
# 2. Le 750 colonne sono **150 blocchi da 5**, e in ogni blocco la posizione 3 è
#    identica alla 2 e la 4 alla 1: è la firma di una FFT a 5 punti, non di uno
#    spettro su 750 frequenze.
#
# ⚠️ **Conclusione:** le colonne `fft` **non** sono uno spettro 0–110 Hz da cui
# ritagliare le bande θ/α/β/γ. Il dataset non contiene il segnale grezzo né le
# potenze di banda, quindi i grafici delle onde cerebrali non si possono ricavare
# onestamente da questo file. (Va deciso come gestirlo: vedi conclusioni.)

# %% [markdown]
# ## 7. Scale delle feature
# Le reti neurali soffrono se le feature hanno ordini di grandezza diversissimi:
# guardiamo la deviazione standard di ogni colonna, famiglia per famiglia
# (in scala logaritmica).

# %% 7. Scale
std_col = X.std()
scale = pd.DataFrame({"famiglia": column_anatomy(X.columns)["famiglia"].values,
                      "log10_std": np.log10(std_col.values + 1e-12)})
ordine = scale.groupby("famiglia")["log10_std"].median().sort_values().index

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=scale, x="famiglia", y="log10_std", order=ordine, ax=ax, color="#85c1e9")
ax.set_title("Deviazione standard delle colonne per famiglia (log10)")
ax.set_ylabel("log10(deviazione standard)")
ax.tick_params(axis="x", rotation=45)
mostra(fig, "04_scale_feature.png")
print(f"Std minima: {std_col.min():.3g}   Std massima: {std_col.max():.3g}   "
      f"-> rapporto ~10^{np.log10(std_col.max() / std_col.min()):.0f}")

# Chi sono le colonne "giganti"? I momenti di ordine alto esplodono con pochi valori estremi
print("\nLe 6 colonne con deviazione standard più alta:")
print(std_col.sort_values(ascending=False).head(6).to_string())
col_max = std_col.idxmax()
print(f"\nQuantili di {col_max}:")
print(X[col_max].quantile([0.5, 0.9, 0.99, 1.0]).to_string())

# Outlier: quante righe hanno almeno un valore a più di 10 deviazioni standard dalla media?
z = (X - X.mean()) / X.std()
print(f"\nRighe con almeno un |z| > 10: {(z.abs() > 10).any(axis=1).sum()} su {len(X)}")

# %% [markdown]
# ## 8. Le feature distinguono le emozioni? Statistiche per canale
# Per le famiglie "semplici" l'indice corrisponde al canale del caschetto (0–4).
# Il Muse ha 4 elettrodi (TP9, AF7, AF8, TP10) più un canale ausiliario: la
# corrispondenza esatta indice→elettrodo non è documentata nel CSV, quindi
# parliamo di *canale 0…4*.

# %% 8. Distribuzioni per classe
famiglie_base = ["mean", "stddev", "entropy"]
fig, axes = plt.subplots(3, 5, figsize=(17, 9))
for r, fam in enumerate(famiglie_base):
    for ch in range(5):
        col = f"{fam}{ch}_a" if fam == "entropy" else f"{fam}_{ch}_a"
        sns.boxplot(data=df, x="label", y=col, order=CLASS_ORDER, palette=CLASS_COLORS,
                    hue="label", legend=False, showfliers=False, ax=axes[r, ch])
        axes[r, ch].set_title(f"{fam} – canale {ch}", fontsize=10)
        axes[r, ch].set_xlabel("")
        axes[r, ch].set_ylabel("")
        axes[r, ch].tick_params(axis="x", labelsize=8)
fig.suptitle("Media, deviazione standard ed entropia del segnale per canale e classe (metà a)")
mostra(fig, "05_statistiche_per_canale.png")

print(df.groupby("label")[[f"mean_{c}_a" for c in range(5)]].median().round(1).to_string())

# L'entropia ha pochissimi valori distinti: controlliamo in quali classi "si accende"
print("\n% di finestre con entropia > 1, per classe:")
for ch in range(5):
    col = f"entropy{ch}_a"
    quota = (df[col] > 1).groupby(df["label"]).mean().reindex(CLASS_ORDER)
    print(f"  {col}: {df[col].nunique():>2} valori distinti  ->  "
          + "   ".join(f"{k}: {v:.0%}" for k, v in quota.items()))

# %% [markdown]
# **Cosa osserviamo:**
# - La classe **NEUTRAL è molto "compatta"** (scatole strettissime): segnale stabile,
#   poca variabilità. **NEGATIVE è la più variabile** (stddev alta sui canali 0, 2, 3).
# - **Canale 2:** la media del segnale in NEGATIVE è circa −235 contro ~+30 delle altre
#   classi. Una differenza di *offset* così grande è più tipica di un cambio di
#   contatto dell'elettrodo/sessione di registrazione che di un'emozione.
# - **Entropia:** ha solo 6–26 valori distinti e `entropy0` è > 1 **solo** in NEUTRAL.
#   Funziona quasi come un "interruttore" che riconosce la sessione neutra.
#   `entropy4` è praticamente costante (zero) → inutile.
#
# ⚠️ Queste feature permettono al modello di riconoscere *la sessione di registrazione*
# invece dell'emozione: l'accuratezza sembrerà altissima ma non sarà tutta "merito
# dell'EEG emotivo". Ne terremo conto nella pulizia e lo diremo nella relazione.

# %% [markdown]
# ## 9. Ridondanza: quanto sono correlate le feature?
# Togliamo prima le colonne identiche (sezione 5), poi calcoliamo la correlazione
# di Pearson fra **tutte** le coppie di colonne rimaste.

# %% 9a. Correlazione globale
X_nodup = X.drop(columns=dup_df["duplicata"])
corr = X_nodup.corr().abs().values
tri = corr[np.triu_indices_from(corr, k=1)]
tri = tri[~np.isnan(tri)]

print(f"Colonne dopo aver tolto i duplicati esatti: {X_nodup.shape[1]}")
print(f"Coppie di colonne analizzate: {len(tri):,}")
for soglia in [0.8, 0.9, 0.95, 0.99]:
    print(f"  coppie con |r| > {soglia}: {(tri > soglia).sum():>8,}  ({(tri > soglia).mean():.2%})")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(tri, bins=100, color="#8e44ad", alpha=0.8)
ax.set_yscale("log")
ax.set_title("Distribuzione della correlazione |r| fra tutte le coppie di feature")
ax.set_xlabel("|r| di Pearson")
ax.set_ylabel("numero di coppie (scala log)")
mostra(fig, "06_distribuzione_correlazioni.png")

# %% 9b. Metà 'a' contro metà 'b' della stessa finestra
nomi_a = [c for c in X_nodup.columns if c.endswith("_a")]
coppie_ab = [(c, c[:-1] + "b") for c in nomi_a if c[:-1] + "b" in X_nodup.columns]
r_ab = pd.Series({a: X_nodup[a].corr(X_nodup[b]) for a, b in coppie_ab}).dropna()
print(f"Coppie a/b confrontate: {len(r_ab)}   mediana r = {r_ab.median():.2f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(r_ab, bins=50, color="#16a085", alpha=0.85)
ax.set_title("Correlazione tra la stessa feature calcolata sulla metà 'a' e sulla metà 'b'")
ax.set_xlabel("r di Pearson")
ax.set_ylabel("numero di feature")
mostra(fig, "07_correlazione_meta_a_b.png")

# %% 9c. Heatmap leggibile: statistiche di base dei 5 canali
cols_base = [f"{f}_{c}_a" for f in ["mean", "stddev", "max", "min"] for c in range(5)]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X[cols_base].corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax,
            cbar_kws={"label": "r di Pearson"})
ax.set_title("Correlazioni fra statistiche di base dei 5 canali (metà a)")
mostra(fig, "08_heatmap_statistiche_base.png")

# %% [markdown]
# ## 10. Uno sguardo d'insieme: PCA (solo visualizzazione)
# Comprimiamo le ~1.800 feature non duplicate in 2 componenti per vedere se le classi
# formano gruppi. **Nota:** qui standardizziamo sull'intero dataset solo per *guardare*;
# quando addestreremo il modello lo scaler andrà calcolato solo sul training set.

# %% 10. PCA
Z = StandardScaler().fit_transform(X_nodup)
# np.errstate: numpy 2.0 su Mac (Apple Accelerate) emette falsi warning nelle
# moltiplicazioni di matrici; i risultati sono corretti (verificato: tutti finiti)
with np.errstate(all="ignore"):
    pca = PCA(n_components=10, svd_solver="full").fit(Z)
    P = pca.transform(Z)
assert np.isfinite(P).all()
var = pca.explained_variance_ratio_

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for cls in CLASS_ORDER:
    m = (y == cls).values
    ax1.scatter(P[m, 0], P[m, 1], s=10, alpha=0.5, color=CLASS_COLORS[cls], label=cls)
ax1.set_xlabel(f"PC1 ({var[0]:.1%})")
ax1.set_ylabel(f"PC2 ({var[1]:.1%})")
ax1.set_title("Proiezione PCA delle finestre EEG")
ax1.legend()
ax2.bar(range(1, 11), var * 100, color="#2980b9", edgecolor="black")
ax2.plot(range(1, 11), np.cumsum(var) * 100, "o-", color="#c0392b", label="cumulata")
ax2.set_xlabel("componente")
ax2.set_ylabel("% varianza spiegata")
ax2.set_title("Varianza spiegata dalle prime 10 componenti")
ax2.legend()
mostra(fig, "09_pca.png")
print(f"Varianza spiegata da PC1+PC2: {var[:2].sum():.1%}   prime 10: {var.sum():.1%}")

# %% [markdown]
# **Lettura del grafico PCA:**
# - NEUTRAL forma una nuvola piccola e compatta; POSITIVE in parte si sovrappone a lei;
#   NEGATIVE è molto sparsa.
# - Molti punti formano **"scie" continue**: sono finestre consecutive nel tempo
#   (le finestre si sovrappongono, quindi righe vicine sono quasi uguali).
#   Le righe del CSV sono mescolate, ma **non sono indipendenti tra loro**.

# %% [markdown]
# ## 11. Conclusioni dell'EDA
#
# | # | Evidenza | Conseguenza per la fase di pulizia |
# |---|---|---|
# | 1 | Nessun NaN/inf, nessuna riga duplicata, classi bilanciate (~33% ciascuna) | Nessun imputing né ribilanciamento necessario |
# | 2 | **732 colonne identiche** (covmat simmetrica + FFT coniugata) | Si eliminano senza perdere informazione: 2548 → 1816 |
# | 3 | FFT = 59% delle colonne, ma **non è uno spettro di frequenza** | Non si possono ricavare θ/α/β/γ da qui; valutare se tenerle come feature "astratte" |
# | 4 | Scale diversissime (std da 0.26 a ~10¹⁷); `moments_15…19` con valori fino a 10¹⁸ | Serve trasformazione robusta (es. log/clipping) + standardizzazione |
# | 5 | Metà `a` e `b` della finestra molto correlate (mediana r ≈ 0.76) | Candidato a riduzione (media a/b o tenerne una) |
# | 6 | ~1.200 coppie con \|r\| > 0.95 dopo aver tolto i duplicati | Filtro di correlazione |
# | 7 | `entropy*` quasi binaria, `entropy4` costante, offset del canale 2 | Feature "di sessione": valutare rimozione e discuterle nella relazione |
# | 8 | Finestre sovrapposte → righe non indipendenti (scie nella PCA) | Lo split casuale train/test darà accuratezze ottimistiche: da dichiarare |
#
# **Decisione aperta:** come mostrare le onde cerebrali θ, α, β, γ, visto che questo
# dataset non le contiene.
