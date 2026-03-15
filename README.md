# Amazon Video Games Recommendation System

A group data science project building an end-to-end recommendation system using the Amazon Video Games Reviews dataset. We explore and compare non-personalized, collaborative filtering (memory-based & model-based), content-based (TF-IDF, BERT, NER), hybrid, and context-aware recommenders — evaluated with regression, classification, and ranking metrics.

---

## Dataset

**Amazon Video Games Reviews** — sourced from Kaggle  
- `Video_Games.json` — 2,565,349 reviews (user ID, product ID, rating, review text, timestamp)  
- `meta_Video_Games.json` — 84,819 products (title, description, category, brand, price, co-purchase links)

> The dataset files are not included in this repository due to their size. Download from [Kaggle](https://www.kaggle.com/datasets/gabrielfreddi/amazon-reviews-de-vdeo-games) and place them in the `data/raw/` folder.

---

## Project Structure

```
recsys-amazon-videogames/
│
├── data/
│   ├── raw/                  # Original dataset files (not tracked by git)
│   └── processed/            # Cleaned & filtered data
│
├── notebooks/
│   ├── 01_eda.ipynb                          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb                # Data cleaning & filtering
│   ├── 03_nonpersonalized.ipynb              # Random, popular, demographic
│   ├── 04_collaborative_filtering.ipynb      # Memory-based & model-based CF
│   ├── 05_content_based.ipynb                # TF-IDF, BERT, NER
│   ├── 06_hybrid.ipynb                       # Hybrid recommenders
│   ├── 07_context_aware.ipynb                # Context-aware & temporal
│   ├── 08_evaluation.ipynb                   # Full evaluation pipeline
│   └── 09_bandit_algorithms.ipynb            # Exploration-exploitation
│
├── src/
│   ├── data/
│   │   └── preprocessing.py   # Data loading & cleaning functions
│   ├── models/
│   │   ├── nonpersonalized.py
│   │   ├── collaborative.py
│   │   ├── content_based.py
│   │   ├── hybrid.py
│   │   └── context_aware.py
│   └── evaluation/
│       └── metrics.py         # All evaluation metrics
│
├── models/                    # Saved model artefacts
│
├── presentation/              # Slides for CEO presentation
│
├── requirements.txt
└── README.md
```

---

## Recommender Approaches Implemented

| Approach | Methods |
|---|---|
| Non-personalized | Random, Popularity-based, Demographic filtering |
| Collaborative Filtering | User-KNN, Item-KNN, SVD, ALS, NMF |
| Content-Based | BoW, TF-IDF, Lemmatization, BERT embeddings, NER |
| Hybrid | Weighted, Switching, Mixed |
| Context-Aware | Temporal weighting, Platform-based context |
| Bandit Algorithms | ε-greedy, UCB |

---

## Evaluation Metrics

- **Regression**: RMSE, MAE
- **Classification**: Precision, Recall, F1
- **Ranking**: NDCG, MAP, Precision@K, Recall@K
- **Best practices**: Cross-validation, CVTT, early stopping

---

## Setup

```bash
git clone https://github.com/sofiasanjose/recsys-amazon-videogames.git
cd recsys-amazon-videogames
pip install -r requirements.txt
```

### Kaggle API Key (required for dataset download)

The dataset is downloaded automatically via `kagglehub` when you run the notebook. You need a Kaggle account and API token:

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. This downloads a `kaggle.json` file
3. Move it to `~/.kaggle/kaggle.json`
4. Run the notebook — the dataset downloads and caches automatically on first run

---

## Team

Group project — MSc Data Science
