# Service Desk Request Classification & Routing

AI/NLP pipeline for automatically classifying and routing multilingual (Russian/Kazakh) Service Desk requests at Otbasy Bank. Built during a Summer 2026 ML/NLP internship.

---

## Problem

The Service Desk receives ~190,000 requests per year across four quarterly exports. Requests are written in Russian and Kazakh, vary widely in completeness and quality, and are currently sorted and routed to support teams manually — a slow, repetitive process.

This project automates two decisions that previously required a human to read each ticket:

1. **Is this request processable?** (vs. gibberish/incomplete/unclear)
2. **Which of the 18 support groups should handle it?**

---

## Dataset

- ~190,000 requests merged from 5 quarterly `.xls` exports (`1 кв`, `2 кв`, `3 кв`, `4 кв декабрь`, `4 кв октябрь-ноябрь`)
- Key fields: `Тип заявки`, `Программное обеспечение`, `Наименование`, `Описание`, `Комментарий исполнителя`, `Группа поддержки`
- PII removed via regex prior to any modeling: IIN/ЖСН numbers, raw 12-digit identifiers, Kazakhstan IBAN/account numbers
- Rows dropped only when `Описание` is completely empty

---

## Pipeline

```
Raw Excel (5 files)
   │
   ▼
Preprocessing  →  PII removal, full_text construction ("query: ..." prefix)
   │
   ▼
Embeddings     →  multilingual-e5-base (768-dim, L2-normalized, cached to .npy)
   │
   ▼
Topic Modeling →  UMAP (10-dim) → HDBSCAN → BERTopic
                   → 191 topic clusters + keywords per request
   │
   ▼
Labeling       →  Regex over Комментарий исполнителя
                   ("не понятно", "уточните", "вложите", etc.) → ground truth
   │
   ▼
Binary Model   →  SMOTE + LinearSVC → Processable (1) vs Unprocessable (0)
   │
   ▼
Routing        →  BERTopic cluster → dominant Группа поддержки
                   + KNN (k=8, cosine) fallback for topic -1 (noise)
```

---

## Methodology

| Component | Choice | Why |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-base` | Handles Russian + Kazakh in a shared semantic space; `query:` prefix per model spec |
| Dimensionality reduction | UMAP (10 components, cosine) | Preserves non-linear neighborhood structure for clustering |
| Clustering | HDBSCAN (min_cluster_size=150) | No predefined cluster count; native outlier detection (-1) |
| Topic modeling | BERTopic | c-TF-IDF keywords per cluster, used as `Keywords` column |
| Labeling | Regex on executor comments | No ground truth existed; executor feedback is the most reliable proxy for "was this request understandable" |
| Imbalance handling | SMOTE (train split only) | 97/3 class split; synthesizes minority-class embeddings by interpolation |
| Binary classifier | LinearSVC (`class_weight='balanced'`, hinge loss) | Scales to 190K rows × 768 dims (RBF kernel is infeasible at this size); standard for linear classification on top of transformer embeddings |
| Routing | Dominant-group mapping + KNN (k=8, cosine) | BERTopic clusters align with support groups (mean dominant share 0.86); KNN routes the 29K noise points by semantic similarity instead of a single default group |

---

## Results

### Binary Classification (Processable vs. Unprocessable)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Gibberish (0) | 0.06 | 0.72 | 0.11 | 1,260 |
| Processable (1) | 0.99 | 0.74 | 0.85 | 55,621 |
| **Weighted avg** | **0.97** | **0.74** | **0.83** | 56,881 |

Label distribution: **185,402 processable (97%)** vs. **4,201 unprocessable (3%)**.

### Routing (18 Support Groups)

| Method | Precision | Recall | F1-score |
|---|---|---|---|
| Cluster mapping only | 0.81 | 0.80 | 0.79 |
| Cluster mapping + KNN (final) | 0.78 | 0.79 | 0.78 |

BERTopic clusters had a mean dominant-group share of **0.86** (177/191 topics > 50% pure), meaning unsupervised topic clusters align very strongly with real support-group assignments.

---

## Known Limitations

- Severe class imbalance (97/3) caps precision on the minority "unprocessable" class even after SMOTE
- Labels are derived from executor comment text, not human annotation — noisy by construction
- 4 of the 18 support groups (10–295 requests each) are too small to form their own BERTopic clusters

---

## Tech Stack

`pandas` · `sentence-transformers` (multilingual-e5-base) · `UMAP` · `HDBSCAN` · `BERTopic` · `scikit-learn` (LinearSVC, KNN, train_test_split, classification_report) · `imbalanced-learn` (SMOTE) · `matplotlib`

---

## Future Work

- Train a supervised multi-class router (replace cluster-mapping heuristic)
- Collect additional labeled "unprocessable" examples to improve minority-class precision
- Deploy as a real-time inference endpoint for incoming tickets
- Periodic retraining as new quarterly data arrives

---

## How to Run

1. Place quarterly `.xls` files in `/content/`
2. Run preprocessing cells → produces `full_text`
3. Run embedding cell → caches `embeddings.npy` (re-run only if data changes)
4. Run BERTopic cell → produces `topics`, `Keywords`
5. Run labeling cell → produces `label`
6. Run SMOTE + LinearSVC cell → binary classifier
7. Run routing cells → `predicted_group` per request
