# Person 3: Content-Based + Hybrid Recommenders

This deliverable implements the full Person 3 scope in:

- `src/models/content_hybrid.py`

It expects Person 1's processed files in `data/processed/`:

- `train.csv`
- `val.csv`
- `metadata_clean.csv` (must include `item_id` and `content_text`)

## What is implemented

1. **Content-based recommender (TF-IDF)**
   - Text cleaning: lowercase, punctuation removal, stopword removal
   - Feature engineering: TF-IDF with unigrams + bigrams
   - Recommendation logic: average similarity from user's seen items to candidate items

2. **Hybrid recommender (weighted)**
   - Builds an item-item collaborative similarity from train interactions
   - Normalizes CF and content scores to [0, 1]
   - Combines with weighted average (`alpha_cf`, default 0.6)

3. **Hybrid recommender (switching)**
   - If user has enough history (`min_history_for_cf`, default 5), uses CF-heavy weighted hybrid
   - Otherwise falls back to content-based

4. **Cold-start strategy**
   - New users (no history): fallback to popularity-based unseen items
   - Users with sparse history: switching hybrid routes to content-based
   - New items: content-based can still recommend if metadata exists in `metadata_clean.csv`

5. **Evaluation + artifacts**
   - Metrics: `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `Coverage`
   - Saves:
     - `models/person3_outputs/person3_model_results.csv`
     - `models/person3_outputs/person3_sample_recommendations.csv`

## Run

From repo root:

```bash
python src/models/content_hybrid.py --processed-dir data/processed --output-dir models/person3_outputs
```

Optional parameters:

- `--k 10`
- `--alpha-cf 0.6`
- `--min-history-for-cf 5`

## Notes for team integration

- This script does **not** use `test.csv`, matching your project rule.
- Person 4 can use `person3_model_results.csv` for comparison tables.
- If Person 2 exports stronger CF scores (e.g., SVD user-item predictions), you can replace the in-script item-CF block with those scores and keep the same hybrid interface.
