# 🌪️ NLP with Disaster Tweets

**Competition:** <https://www.kaggle.com/competitions/nlp-getting-started>
**Kaggle notebook:** <https://www.kaggle.com/code/KAGGLE_USERNAME/disaster-tweets-eda-baselines-and-roberta-ensemble>

Predict whether a tweet describes a real disaster (`1`) or not (`0`). ~7.6k labelled tweets, ~3.3k test tweets.
Evaluation metric: **F1-score** of the positive class.

## Pipeline

```
train.csv / test.csv
      │
      ▼
 EDA ──► Preprocessing ──► 5 shared stratified folds
          • html unescape, URL → "http", @user → "@user", strip "#"
          • prepend keyword:  "<keyword> | <clean text>"
          • majority-vote fix for duplicated tweets with conflicting labels
          • meta features (length, #urls, #mentions, upper/digit ratio, ...)
      │
      ├──► A. TF-IDF word(1-2) + char_wb(2-5) → Logistic Regression
      ├──► B. TF-IDF word(1-2) + meta features → LightGBM
      └──► C. roberta-base fine-tuned (3 epochs, lr 2e-5, fp16, max_len 80)
      │
      ▼
 Weighted blend + decision threshold, both tuned on out-of-fold F1
      │
      ▼
 submission.csv
```

## Key EDA findings

- **Mild class imbalance (57/43)** → F1 depends on the threshold; tuning it on OOF beats the default 0.5.
- **`keyword` is highly predictive** (e.g. *wreckage*, *derailment* ≈ always disasters; *body bags*, *ruin* ≈ figurative).
  `location` is ~33% missing and noisy, so only a `has_location` flag is used.
- **Style differs by class**: disaster tweets are longer and contain more URLs (news-like); others have more @mentions.
- **Label noise**: identical tweets appear with opposite labels → resolved by majority vote.

## Results

> Fill in after running the notebook on Kaggle (GPU T4, Internet on) and submitting.

| Model | OOF F1 @0.5 | OOF F1 @best thr | Public LB |
|-------|------------:|-----------------:|----------:|
| A. TF-IDF + Logistic Regression | _TBD_ | _TBD_ | _TBD_ |
| B. TF-IDF + meta → LightGBM | _TBD_ | _TBD_ | _TBD_ |
| C. RoBERTa-base (5-fold) | _TBD_ | _TBD_ | _TBD_ |
| **Ensemble (A + B + C)** | _TBD_ | _TBD_ | **_TBD_** |

The notebook writes one `submission_<model>.csv` per model plus the blended `submission.csv`, so every row above can
be scored on the public leaderboard.

> ⚠️ This competition's public leaderboard has many 1.000 scores obtained by submitting the publicly available
> test labels. Honest solutions typically land between **0.80 and 0.85**.

## Lessons learned

- A well-tuned linear model on word + char TF-IDF is a surprisingly strong baseline for short texts.
- Light cleaning (replacing URLs/mentions with tokens) works better than deleting them — they carry signal.
- Pre-trained language models understand figurative language that n-grams miss; blending them with classical models
  adds diversity.
- Shared folds + OOF predictions make model comparison and ensembling honest and simple.

## Next steps

- Try `microsoft/deberta-v3-base` and `vinai/bertweet-base`; average several seeds.
- Pseudo-label confident test predictions.
- Error analysis on the most confident OOF mistakes.
