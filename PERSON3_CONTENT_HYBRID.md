# Person 3: Content-Based + Hybrid Recommenders

This deliverable is implemented in:

- `notebooks/03_content_based_and_hybrid.ipynb`

It expects Person 1's processed files in `data/processed/`:

- `train.csv`
- `val.csv`
- `metadata_clean.csv` (must include `item_id` and `content_text`)

## What is implemented

1. **Metadata review**
   - Notebook cell loads `metadata_clean.csv`, lists columns, shows samples, and summarizes `content_text` length / empties.

2. **Content-based recommender (TF-IDF or bag-of-words)**
   - Text cleaning: lowercase, punctuation removal, stopword removal
   - Optional **WordNet lemmatization** (NLTK) for a stronger lexical baseline
   - Feature engineering: **TF-IDF** (default) or **CountVectorizer** BoW, unigrams + bigrams
   - Recommendation logic: average similarity from items the user saw in train to candidate items

3. **Hybrid recommender (weighted)**
   - Builds an item-item collaborative similarity from train interactions
   - Normalizes CF and content scores to [0, 1]
   - Combines with weighted average (`ALPHA_CF`, default 0.6 in the notebook)

4. **Hybrid recommender (switching)**
   - If user has enough history (`MIN_HISTORY_FOR_CF`, default 5), uses CF-heavy weighted hybrid
   - Otherwise falls back to content-based

5. **Cold-start strategy**
   - New users (no history): fallback to popularity-based unseen items
   - Users with sparse history: switching hybrid routes to content-based
   - New items: content-based can still recommend if metadata exists in `metadata_clean.csv`

6. **Evaluation + artifacts**
   - Metrics: `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Coverage`
   - Saves under `models/person3_outputs/`:
     - `person3_model_results.csv`
     - `person3_sample_recommendations.csv`

## Run

1. Open Jupyter from the **repository root** (or from `notebooks/`; the notebook resolves `data/processed` automatically).
2. Run all cells in `notebooks/03_content_based_and_hybrid.ipynb`.

Tune in the config cell:

- `K`, `ALPHA_CF`, `MIN_HISTORY_FOR_CF`
- `CONTENT_VECTORIZER`: `"tfidf"` or `"bow"` (primary pipeline; hybrids use this content space)
- `USE_LEMMATIZATION`: apply lemmas before vectorization (TF-IDF or BoW)
- `MAX_FEATURES`, `NGRAM_RANGE`
- `RUN_CONTENT_BOW_BASELINE` / `RUN_CONTENT_LEMMA_BASELINE`: optional extra content-only rows in the results table for comparison (more compute)

## Notes for team integration

- This notebook does **not** use `test.csv`, matching your project rule.
- Person 4 can use `person3_model_results.csv` for comparison tables.
- If Person 2 exports stronger CF scores (e.g., SVD), you can replace the item–item CF block in the implementation cell with those scores and keep the same hybrid scoring interface.
