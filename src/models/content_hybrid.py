from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler


DEFAULT_RELEVANCE_THRESHOLD = 4.0
DEFAULT_K = 10


def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def precision_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_set = set(recommended[:k])
    return len(rec_set.intersection(relevant)) / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    rec_set = set(recommended[:k])
    return len(rec_set.intersection(relevant)) / len(relevant)


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    recs = recommended[:k]
    discounts = np.log2(np.arange(2, len(recs) + 2))
    gains = np.array([1.0 if item in relevant else 0.0 for item in recs], dtype=float)
    dcg = np.sum(gains / discounts)

    ideal_len = min(len(relevant), k)
    ideal_discounts = np.log2(np.arange(2, ideal_len + 2))
    idcg = np.sum(np.ones(ideal_len) / ideal_discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def ap_at_k(recommended: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = 0
    sum_precision = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / min(len(relevant), k)


@dataclass
class DataBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    metadata: pd.DataFrame


class ContentHybridRecommender:
    def __init__(
        self,
        data: DataBundle,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        self.train = data.train.copy()
        self.val = data.val.copy()
        self.metadata = data.metadata.copy()
        self.relevance_threshold = relevance_threshold

        self._validate_required_columns()
        self._prepare_ids_as_str()

        self.item_ids: np.ndarray = self.metadata["item_id"].values
        self.item_id_to_idx: Dict[str, int] = {iid: i for i, iid in enumerate(self.item_ids)}
        self.user_seen_train: Dict[str, Set[str]] = (
            self.train.groupby("user_id")["item_id"].apply(set).to_dict()
        )
        self.user_train_count: Dict[str, int] = self.train.groupby("user_id").size().to_dict()
        self.user_val_relevant: Dict[str, Set[str]] = (
            self.val[self.val["rating"] >= self.relevance_threshold]
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )
        self.eval_users: List[str] = sorted(
            [u for u, rel in self.user_val_relevant.items() if rel and u in self.user_seen_train]
        )

        self.popular_items: List[str] = (
            self.train.groupby("item_id")
            .size()
            .sort_values(ascending=False)
            .index.astype(str)
            .tolist()
        )

        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.item_tfidf: csr_matrix | None = None
        self.content_sim: np.ndarray | None = None
        self.item_cf_sim: np.ndarray | None = None

    def _validate_required_columns(self) -> None:
        train_req = {"user_id", "item_id", "rating"}
        val_req = {"user_id", "item_id", "rating"}
        meta_req = {"item_id"}
        if "content_text" not in self.metadata.columns:
            missing_meta = meta_req.union({"content_text"}) - set(self.metadata.columns)
            raise ValueError(
                f"metadata_clean.csv missing required columns: {sorted(missing_meta)}. "
                "Expected at least ['item_id', 'content_text']."
            )

        missing_train = train_req - set(self.train.columns)
        missing_val = val_req - set(self.val.columns)
        if missing_train:
            raise ValueError(f"train.csv missing required columns: {sorted(missing_train)}")
        if missing_val:
            raise ValueError(f"val.csv missing required columns: {sorted(missing_val)}")

    def _prepare_ids_as_str(self) -> None:
        for df in (self.train, self.val, self.metadata):
            df["user_id"] = df["user_id"].astype(str) if "user_id" in df.columns else None
            df["item_id"] = df["item_id"].astype(str)
        self.metadata["content_text"] = (
            self.metadata["content_text"].fillna("").astype(str).map(_clean_text)
        )

    def fit_content(self, max_features: int = 30000, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.item_tfidf = self.tfidf_vectorizer.fit_transform(self.metadata["content_text"])
        self.content_sim = linear_kernel(self.item_tfidf, self.item_tfidf)

    def fit_item_cf(self) -> None:
        users = self.train["user_id"].astype("category")
        items = self.train["item_id"].astype("category")
        rows = users.cat.codes.values
        cols = items.cat.codes.values
        vals = np.ones(len(self.train), dtype=np.float32)

        user_item = csr_matrix((vals, (rows, cols)), shape=(users.cat.categories.size, items.cat.categories.size))
        item_user = user_item.T
        cf_sim = linear_kernel(item_user, item_user)

        item_order = items.cat.categories.astype(str).tolist()
        cf_index = {iid: idx for idx, iid in enumerate(item_order)}

        full_sim = np.zeros((len(self.item_ids), len(self.item_ids)), dtype=np.float32)
        for i, iid_i in enumerate(self.item_ids):
            if iid_i not in cf_index:
                continue
            row_i = cf_index[iid_i]
            for j, iid_j in enumerate(self.item_ids):
                if iid_j not in cf_index:
                    continue
                full_sim[i, j] = cf_sim[row_i, cf_index[iid_j]]
        self.item_cf_sim = full_sim

    def _top_popular_unseen(self, seen: Set[str], k: int) -> List[str]:
        return [item for item in self.popular_items if item not in seen][:k]

    def recommend_content(self, user_id: str, k: int = DEFAULT_K) -> List[str]:
        if self.content_sim is None:
            raise RuntimeError("fit_content() must be called before recommend_content().")
        seen = self.user_seen_train.get(user_id, set())
        if not seen:
            return self._top_popular_unseen(seen, k)

        seen_idx = [self.item_id_to_idx[i] for i in seen if i in self.item_id_to_idx]
        if not seen_idx:
            return self._top_popular_unseen(seen, k)

        content_scores = self.content_sim[seen_idx].mean(axis=0)
        ranked_idx = np.argsort(content_scores)[::-1]
        recs: List[str] = []
        for idx in ranked_idx:
            iid = self.item_ids[idx]
            if iid in seen:
                continue
            recs.append(iid)
            if len(recs) >= k:
                break
        if len(recs) < k:
            recs.extend(self._top_popular_unseen(seen.union(set(recs)), k - len(recs)))
        return recs

    def recommend_weighted_hybrid(
        self,
        user_id: str,
        k: int = DEFAULT_K,
        alpha_cf: float = 0.6,
    ) -> List[str]:
        if self.content_sim is None or self.item_cf_sim is None:
            raise RuntimeError("fit_content() and fit_item_cf() must be called before hybrid recommendations.")
        seen = self.user_seen_train.get(user_id, set())
        if not seen:
            return self._top_popular_unseen(seen, k)

        seen_idx = [self.item_id_to_idx[i] for i in seen if i in self.item_id_to_idx]
        if not seen_idx:
            return self._top_popular_unseen(seen, k)

        content_scores = self.content_sim[seen_idx].mean(axis=0).reshape(-1, 1)
        cf_scores = self.item_cf_sim[seen_idx].mean(axis=0).reshape(-1, 1)

        scaler = MinMaxScaler()
        content_norm = scaler.fit_transform(content_scores).ravel()
        cf_norm = scaler.fit_transform(cf_scores).ravel()

        hybrid_scores = alpha_cf * cf_norm + (1.0 - alpha_cf) * content_norm
        ranked_idx = np.argsort(hybrid_scores)[::-1]

        recs: List[str] = []
        for idx in ranked_idx:
            iid = self.item_ids[idx]
            if iid in seen:
                continue
            recs.append(iid)
            if len(recs) >= k:
                break
        if len(recs) < k:
            recs.extend(self._top_popular_unseen(seen.union(set(recs)), k - len(recs)))
        return recs

    def recommend_switching_hybrid(
        self,
        user_id: str,
        k: int = DEFAULT_K,
        min_history_for_cf: int = 5,
        alpha_cf: float = 0.6,
    ) -> List[str]:
        history_len = self.user_train_count.get(user_id, 0)
        if history_len >= min_history_for_cf:
            return self.recommend_weighted_hybrid(user_id=user_id, k=k, alpha_cf=alpha_cf)
        return self.recommend_content(user_id=user_id, k=k)

    def evaluate(self, model_name: str, recommend_fn, k: int = DEFAULT_K) -> Dict[str, float]:
        precisions, recalls, ndcgs, maps = [], [], [], []
        all_recommended = set()

        for user_id in self.eval_users:
            relevant = self.user_val_relevant.get(user_id, set())
            recs = recommend_fn(user_id, k)
            all_recommended.update(recs)
            precisions.append(precision_at_k(recs, relevant, k))
            recalls.append(recall_at_k(recs, relevant, k))
            ndcgs.append(ndcg_at_k(recs, relevant, k))
            maps.append(ap_at_k(recs, relevant, k))

        coverage = len(all_recommended) / len(self.item_ids) if len(self.item_ids) else 0.0
        return {
            "model": model_name,
            "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
            "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
            "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "map_at_k": float(np.mean(maps)) if maps else 0.0,
            "coverage": coverage,
            "eval_users": len(self.eval_users),
        }


def load_data(processed_dir: Path) -> DataBundle:
    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    metadata_path = processed_dir / "metadata_clean.csv"
    missing = [p.name for p in (train_path, val_path, metadata_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required files in {processed_dir}: {missing}. "
            "Run Person 1 preprocessing first, then rerun this script."
        )
    return DataBundle(
        train=pd.read_csv(train_path),
        val=pd.read_csv(val_path),
        metadata=pd.read_csv(metadata_path),
    )


def save_outputs(
    output_dir: Path,
    results: pd.DataFrame,
    sample_recs: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "person3_model_results.csv", index=False)
    sample_recs.to_csv(output_dir / "person3_sample_recommendations.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Person 3: Content-based and Hybrid Recommender Pipeline")
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=str(Path("data") / "processed"),
        help="Path to processed data directory containing train.csv, val.csv, metadata_clean.csv",
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-K recommendations")
    parser.add_argument("--alpha-cf", type=float, default=0.6, help="CF weight in weighted hybrid")
    parser.add_argument(
        "--min-history-for-cf",
        type=int,
        default=5,
        help="Switching hybrid threshold: min user interactions to use CF-heavy hybrid",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path("models") / "person3_outputs"),
        help="Directory for result artifacts",
    )
    args = parser.parse_args()

    data = load_data(Path(args.processed_dir))
    recommender = ContentHybridRecommender(data=data)

    recommender.fit_content()
    recommender.fit_item_cf()

    results = [
        recommender.evaluate(
            model_name="content_tfidf",
            recommend_fn=lambda u, k: recommender.recommend_content(user_id=u, k=k),
            k=args.k,
        ),
        recommender.evaluate(
            model_name="hybrid_weighted",
            recommend_fn=lambda u, k: recommender.recommend_weighted_hybrid(
                user_id=u, k=k, alpha_cf=args.alpha_cf
            ),
            k=args.k,
        ),
        recommender.evaluate(
            model_name="hybrid_switching",
            recommend_fn=lambda u, k: recommender.recommend_switching_hybrid(
                user_id=u,
                k=k,
                min_history_for_cf=args.min_history_for_cf,
                alpha_cf=args.alpha_cf,
            ),
            k=args.k,
        ),
    ]
    results_df = pd.DataFrame(results)

    sample_users = recommender.eval_users[:10]
    sample_rows = []
    for u in sample_users:
        sample_rows.append(
            {
                "user_id": u,
                "content_recs": recommender.recommend_content(u, args.k),
                "weighted_hybrid_recs": recommender.recommend_weighted_hybrid(u, args.k, args.alpha_cf),
                "switching_hybrid_recs": recommender.recommend_switching_hybrid(
                    u, args.k, args.min_history_for_cf, args.alpha_cf
                ),
            }
        )
    sample_df = pd.DataFrame(sample_rows)

    save_outputs(Path(args.output_dir), results_df, sample_df)

    print("Person 3 pipeline completed.")
    print(results_df.sort_values("ndcg_at_k", ascending=False).to_string(index=False))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
