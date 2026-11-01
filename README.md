# 🏆 Kaggle Competitions

My solutions to Kaggle competitions — focused on **clean, reproducible pipelines**, honest validation
and clear documentation rather than leaderboard tricks.

| Competition | Type | Metric | Approach | CV (OOF) | Public LB | Notebook |
|-------------|------|--------|----------|---------:|----------:|----------|
| [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) | NLP · binary classification | F1 | TF-IDF + LogReg · LightGBM · RoBERTa · weighted ensemble | _TBD_ | _TBD_ | [folder](nlp-getting-started/) · [Kaggle](https://www.kaggle.com/code/KAGGLE_USERNAME/disaster-tweets-eda-baselines-and-roberta-ensemble) |

## How I approach a competition

1. **EDA first** — understand the target, the metric and data quirks (missing values, leakage, label noise).
2. **Fixed validation scheme** — one set of stratified folds shared by every model, so scores are comparable
   and out-of-fold predictions can be blended.
3. **Strong simple baseline** — linear / tree models with sensible features before anything heavy.
4. **Pre-trained / complex models** — only once the baseline is solid.
5. **Ensemble + metric-aware post-processing** (e.g. threshold tuning for F1), tuned on OOF, never on the LB.

## Repository structure

```
kaggle-competitions/
├── README.md                     # this overview
├── requirements.txt
└── nlp-getting-started/
    ├── README.md                 # approach, results and lessons learned
    ├── disaster-tweets-pipeline.ipynb
    └── kernel-metadata.json      # lets `kaggle kernels push` publish the notebook
```

## Running locally

```bash
pip install -r requirements.txt
kaggle competitions download -c nlp-getting-started -p nlp-getting-started/data
unzip nlp-getting-started/data/nlp-getting-started.zip -d nlp-getting-started/data
```

Then open the notebook from inside `nlp-getting-started/`. The transformer section needs a CUDA GPU; without one
it is skipped automatically and the classical models + ensemble still run.
