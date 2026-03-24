"""Rebuild notebooks/03_content_based_and_hybrid.ipynb with full Person 3 checklist."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks" / "03_content_based_and_hybrid.ipynb"

CELL_IMPORTS = r'''from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DEFAULT_RELEVANCE_THRESHOLD = 4.0
DEFAULT_K = 10
'''

CELL_CONFIG = r'''# Resolve repo root: run from repo root or from notebooks/
_cwd = Path.cwd()
if (_cwd / "data" / "processed").exists():
    REPO_ROOT = _cwd
elif (_cwd.parent / "data" / "processed").exists():
    REPO_ROOT = _cwd.parent
else:
    REPO_ROOT = _cwd

PROC_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_DIR = REPO_ROOT / "models" / "person3_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
DEMO_DIR = REPO_ROOT / "data" / "person3_demo"

# Use bundled tiny demo if full preprocessing outputs are missing (reproducible notebook run)
if not (PROC_DIR / "train.csv").exists() and DEMO_DIR.exists():
    PROC_DIR = DEMO_DIR
    print("NOTE: Using bundled demo data in data/person3_demo/ (replace with Person 1 processed CSVs for real results).")

K = 10
# Default hybrid settings (also used as midpoints in sweeps)
ALPHA_CF = 0.6
MIN_HISTORY_FOR_CF = 5

MAX_FEATURES = 30000
NGRAM_RANGE = (1, 2)

# Sweeps (hybrid uses one fitted content+CF model; only α / threshold change at recommend time)
ALPHA_VALUES = [0.3, 0.5, 0.6, 0.7, 0.9]
SWITCHING_THRESHOLDS = [3, 5, 10, 15]

print("REPO_ROOT:", REPO_ROOT.resolve())
print("PROC_DIR:", PROC_DIR.resolve())
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
'''

CELL_META_MD = """## Review item metadata

Person 1 builds **`content_text`** from **title**, **description**, **features**, **brand**, and **categories** into one searchable profile.

**Which fields matter for our models?**

| Signal | Role |
| --- | --- |
| Title + description + features | Core vocabulary for TF-IDF / BoW similarity between games |
| Brand / publisher | Often captured in `content_text`; helps series/franchise similarity |
| Categories (main/sub) | Structure the catalog; useful for sanity checks and plots |

We explore distributions below; the **recommendation models** consume the pre-merged `content_text` column (plus collaborative signals in hybrids).
"""

CELL_META_EXPLORE = r"""_meta_path = PROC_DIR / "metadata_clean.csv"
if not _meta_path.exists():
    raise FileNotFoundError(f"Expected {_meta_path}")

meta_explore = pd.read_csv(_meta_path)
print("Shape:", meta_explore.shape)
print("Columns:", list(meta_explore.columns))

# Example rows (readable)
show_cols = [c for c in ["item_id", "title", "brand", "main_category", "sub_category"] if c in meta_explore.columns]
if show_cols:
    display(meta_explore[show_cols].head(8))
else:
    display(meta_explore.head(8))

if "content_text" in meta_explore.columns:
    raw_len = meta_explore["content_text"].fillna("").astype(str).str.len()
    print(
        "\ncontent_text length — min / median / max:",
        int(raw_len.min()),
        float(raw_len.median()),
        int(raw_len.max()),
    )
    print(f"Share empty content_text: {(raw_len == 0).mean():.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.clip(raw_len, 0, np.percentile(raw_len, 99)), bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution of content_text length (clipped at 99th pctl)")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Items")

    if "main_category" in meta_explore.columns:
        top_cat = meta_explore["main_category"].fillna("unknown").value_counts().head(12)
        top_cat.plot(kind="barh", ax=axes[1], color="coral")
        axes[1].set_title("Top main_category values")
    else:
        axes[1].text(0.1, 0.5, "No main_category column in this export", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_metadata_overview.png", bbox_inches="tight")
    plt.show()
else:
    print("No content_text column found.")
"""

# Keep implementation cell from existing file - read it
def main() -> None:
    old = json.loads(NB.read_text(encoding="utf-8"))
    impl_cell = None
    for c in old["cells"]:
        src = "".join(c.get("source", []))
        if "class ContentHybridRecommender" in src and "def load_data_nb" in src:
            impl_cell = c
            break
    if impl_cell is None:
        raise SystemExit("Could not find implementation cell")

    md_impl = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Implementation\n\n"
            "**Pipeline:** `_clean_text` → optional **WordNet lemmatization** → **TF-IDF** or **BoW** → **cosine** item similarity (`linear_kernel`).\n\n"
            "**Hybrids:** weighted blend of normalized content vs item–item CF scores; switching sends low-history users to content-first.\n\n"
            "Metrics, recommender class, and helpers:"
        ],
    }

    md_fit = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Fit, compare, evaluate, save\n\n"
            "**Validation protocol:** same as Person 2 — top-K **unseen** train items per user; **relevant** if rating ≥ 4 in `val.csv`.\n\n"
            "Sections below: (1) content TF-IDF vs BoW vs lemmatized TF-IDF, (2) hybrid **α** sweep, (3) **switching** threshold sweep, (4) exports + **titles** on sample recommendations."
        ],
    }

    cell_load = r"""data = load_data_nb(PROC_DIR)

# Titles for readable recommendations (from raw metadata before cleaning in bundle)
_meta_raw = pd.read_csv(PROC_DIR / "metadata_clean.csv")
item_titles = {}
if "title" in _meta_raw.columns:
    item_titles = _meta_raw.set_index(_meta_raw["item_id"].astype(str))["title"].to_dict()
else:
    item_titles = {str(r["item_id"]): str(r.get("item_id", "")) for _, r in _meta_raw.iterrows()}

def title_of(iid: str) -> str:
    return str(item_titles.get(str(iid), iid))

print("Users in train:", data.train["user_id"].nunique(), "| Items in metadata:", len(item_titles))
"""

    cell_content_compare = r"""# --- 1) Content-only: TF-IDF vs BoW vs lemmatized TF-IDF ---
content_rows = []

def eval_content(name: str, r: ContentHybridRecommenderNB) -> None:
    row = r.evaluate(
        model_name=name,
        recommend_fn=lambda u, k, rr=r: rr.recommend_content(user_id=u, k=k),
        k=K,
    )
    content_rows.append(row)

r_tfidf = ContentHybridRecommenderNB(data=data)
r_tfidf.fit_content(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, vectorizer="tfidf", use_lemmatization=False)
eval_content("content_tfidf", r_tfidf)

r_bow = ContentHybridRecommenderNB(data=data)
r_bow.fit_content(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, vectorizer="bow", use_lemmatization=False)
eval_content("content_bow", r_bow)

r_lem = ContentHybridRecommenderNB(data=data)
r_lem.fit_content(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, vectorizer="tfidf", use_lemmatization=True)
eval_content("content_tfidf_lemmatized", r_lem)

df_content = pd.DataFrame(content_rows)
print("\\n=== Content representation comparison (validation) ===")
display(df_content.sort_values("ndcg_at_k", ascending=False))

fig, ax = plt.subplots(figsize=(8, 4))
plot_df = df_content.melt(
    id_vars=["model"],
    value_vars=["ndcg_at_k", "map_at_k"],
    var_name="metric",
    value_name="value",
)
sns.barplot(data=plot_df, x="model", y="value", hue="metric", ax=ax)
ax.set_title("Content-only models: NDCG@K and MAP@K")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "fig_content_tfidf_vs_bow_vs_lemma.png", bbox_inches="tight")
plt.show()

best_content = df_content.loc[df_content["ndcg_at_k"].idxmax(), "model"]
print(f"\\nBest content-only by NDCG@K: {best_content}")
"""

    cell_hybrid_base = r"""# --- 2) Hybrid: use standard TF-IDF + item-item CF (same as Person 2 spirit) ---
recommender = ContentHybridRecommenderNB(data=data)
recommender.fit_content(
    max_features=MAX_FEATURES,
    ngram_range=NGRAM_RANGE,
    vectorizer="tfidf",
    use_lemmatization=False,
)
recommender.fit_item_cf()

# α sweep (no refit; same matrices)
alpha_rows = []
for a in ALPHA_VALUES:
    alpha_rows.append(
        recommender.evaluate(
            model_name=f"hybrid_weighted_alpha_{a}",
            recommend_fn=lambda u, k, rr=recommender, aa=a: rr.recommend_weighted_hybrid(
                user_id=u, k=k, alpha_cf=aa
            ),
            k=K,
        )
    )
df_alpha = pd.DataFrame(alpha_rows)
print("=== Weighted hybrid: α sweep ===")
display(df_alpha[["model", "ndcg_at_k", "map_at_k", "precision_at_k"]])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(df_alpha["model"].str.replace("hybrid_weighted_alpha_", "").astype(float), df_alpha["ndcg_at_k"], "o-", label="NDCG@K")
ax.plot(df_alpha["model"].str.replace("hybrid_weighted_alpha_", "").astype(float), df_alpha["map_at_k"], "s--", label="MAP@K")
ax.set_xlabel("α (CF weight)")
ax.set_ylabel("Score")
ax.set_title("Weighted hybrid vs α (content weight = 1−α)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_hybrid_alpha_sweep.png", bbox_inches="tight")
plt.show()

best_alpha_row = df_alpha.loc[df_alpha["ndcg_at_k"].idxmax()]
ALPHA_CF = float(best_alpha_row["model"].replace("hybrid_weighted_alpha_", ""))
print(f"Best α by NDCG@K: {ALPHA_CF}")
"""

    cell_switch = r"""# --- 3) Switching threshold sweep (uses best α from previous cell) ---
sw_rows = []
for th in SWITCHING_THRESHOLDS:
    sw_rows.append(
        recommender.evaluate(
            model_name=f"hybrid_switching_th_{th}",
            recommend_fn=lambda u, k, rr=recommender, t=th: rr.recommend_switching_hybrid(
                user_id=u, k=k, min_history_for_cf=t, alpha_cf=ALPHA_CF
            ),
            k=K,
        )
    )
df_sw = pd.DataFrame(sw_rows)
print("=== Switching hybrid: min history threshold ===")
display(df_sw[["model", "ndcg_at_k", "map_at_k"]])

fig, ax = plt.subplots(figsize=(7, 4))
th_x = [int(m.split("_th_")[1]) for m in df_sw["model"]]
ax.bar(range(len(th_x)), df_sw["ndcg_at_k"], color="seagreen", alpha=0.85)
ax.set_xticks(range(len(th_x)))
ax.set_xticklabels(th_x)
ax.set_xlabel("Min train interactions for CF-heavy hybrid")
ax.set_ylabel("NDCG@K")
ax.set_title(f"Switching hybrid NDCG@K (α={ALPHA_CF})")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_switching_threshold_sweep.png", bbox_inches="tight")
plt.show()

best_th = int(df_sw.loc[df_sw["ndcg_at_k"].idxmax(), "model"].split("_th_")[1])
MIN_HISTORY_FOR_CF = best_th
print(f"Best switching threshold by NDCG@K: {MIN_HISTORY_FOR_CF}")
"""

    cell_finalize = r"""# --- 4) Merge results, save CSVs, human-readable sample recs ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Final "production" metrics using tuned α and threshold from sweeps
final_rows = [
    recommender.evaluate(
        model_name="hybrid_weighted_tuned",
        recommend_fn=lambda u, k, rr=recommender: rr.recommend_weighted_hybrid(user_id=u, k=k, alpha_cf=ALPHA_CF),
        k=K,
    ),
    recommender.evaluate(
        model_name="hybrid_switching_tuned",
        recommend_fn=lambda u, k, rr=recommender: rr.recommend_switching_hybrid(
            user_id=u, k=k, min_history_for_cf=MIN_HISTORY_FOR_CF, alpha_cf=ALPHA_CF
        ),
        k=K,
    ),
]

results_full = pd.concat(
    [df_content, df_alpha, df_sw, pd.DataFrame(final_rows)],
    ignore_index=True,
)
results_full.to_csv(OUTPUT_DIR / "person3_model_results.csv", index=False)
df_content.to_csv(OUTPUT_DIR / "person3_content_comparison.csv", index=False)
df_alpha.to_csv(OUTPUT_DIR / "person3_alpha_sweep.csv", index=False)
df_sw.to_csv(OUTPUT_DIR / "person3_switching_sweep.csv", index=False)
pd.DataFrame(final_rows).to_csv(OUTPUT_DIR / "person3_final_tuned_metrics.csv", index=False)

# Sample recommendations with titles
sample_users = recommender.eval_users[: min(10, len(recommender.eval_users))]
readable = []
for u in sample_users:
    ids = recommender.recommend_weighted_hybrid(u, K, ALPHA_CF)
    readable.append(
        {
            "user_id": u,
            "recommendations": " | ".join(f"{title_of(i)} ({i})" for i in ids),
        }
    )
sample_df = pd.DataFrame(readable)
sample_df.to_csv(OUTPUT_DIR / "person3_sample_recommendations_titles.csv", index=False)
display(sample_df)

# Summary chart: key models
summary_models = [
    "content_tfidf",
    "content_bow",
    "content_tfidf_lemmatized",
    "hybrid_weighted_tuned",
    "hybrid_switching_tuned",
]
key_df = results_full[results_full["model"].isin(summary_models)].copy()
if key_df.empty:
    key_df = results_full.tail(8)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(key_df))
ax.bar(x, key_df["ndcg_at_k"], color="slateblue", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(key_df["model"], rotation=25, ha="right")
ax.set_ylabel("NDCG@K")
ax.set_title("Key Person 3 models (validation)")
plt.tight_layout()
fig.savefig(FIG_DIR / "fig_key_models_ndcg.png", bbox_inches="tight")
plt.show()

print("Saved:", OUTPUT_DIR / "person3_model_results.csv")
print("Figures:", FIG_DIR)
"""

    md_interpret = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Interpretation (for report)\n\n"
            "- **TF-IDF vs BoW:** TF-IDF usually wins when rare discriminative terms matter; BoW can work on very short text.\n"
            "- **Lemmatization:** Can help vocabulary match; not always better if it removes useful game-specific tokens.\n"
            "- **α:** Higher α trusts collaborative co-interaction more; lower α leans on metadata when CF is noisy.\n"
            "- **Switching threshold:** Lower → more users get CF-heavy hybrid sooner; higher → more content/popularity for cold/sparse users.\n"
            "- **Demo data:** Bundled `data/person3_demo/` is only for runnable outputs; **replace with Person 1 `data/processed/`** for your real report numbers."
        ],
    }

    new_cells = [
        old["cells"][0],
        {"cell_type": "code", "metadata": {}, "source": [CELL_IMPORTS + "\n"], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [CELL_CONFIG + "\n"], "outputs": [], "execution_count": None},
        {"cell_type": "markdown", "metadata": {}, "source": [CELL_META_MD]},
        {
            "cell_type": "code",
            "metadata": {},
            "source": [CELL_META_EXPLORE + "\n"],
            "outputs": [],
            "execution_count": None,
        },
        md_impl,
        impl_cell,
        md_fit,
        {"cell_type": "code", "metadata": {}, "source": [cell_load + "\n"], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [cell_content_compare + "\n"], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [cell_hybrid_base + "\n"], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [cell_switch + "\n"], "outputs": [], "execution_count": None},
        {"cell_type": "code", "metadata": {}, "source": [cell_finalize + "\n"], "outputs": [], "execution_count": None},
        md_interpret,
    ]

    nb = old
    nb["cells"] = new_cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("Wrote", NB)


if __name__ == "__main__":
    main()
