# Transaction Fraud Risk — Which 500 Transactions Should the Fraud Team Check Today?

A payments company receives ~100,000 card transactions a day. Some are on stolen cards. The fraud team can only manually review about 500 of them.

The question isn't "is this fraud?" It's **which 500 do they open first?**

---

## Headline

> **At 500 manual reviews a day, ranking by expected value recovers 22.2% of fraud value at 47% precision.**
> A perfect model at the same budget recovers 53.7%, so this captures 41% of what is achievable.

---

## Which 500 — the strategy that matters more than the model

Same model, same 500 reviews, three ways of ordering the queue:

| strategy | frauds caught | precision | fraud value recovered |
|---|---|---|---|
| by probability | 461 | 0.922 | 7.1% |
| by probability, then amount | 331 | 0.662 | 18.2% |
| **by expected value (p × amount)** | **236** | **0.472** | **22.2%** |
| perfect model, 500 reviews | — | 1.000 | 53.7% |

Ranking by confidence catches the most frauds but the cheapest ones — median fraud here is **$66.40**, against $68.50 for legitimate transactions. Amount alone carries almost no signal, which the control row confirms.

Ranking by `probability × amount` recovers **three times the money** at half the precision. A 0.4 score on a $2,000 transaction is $800 at risk; a 0.99 on a $60 one is $59.

Which to ship is a business decision, not a modelling one. Chargeback losses → expected value. Account compromises → probability.

---

## Data

- **IEEE-CIS Fraud Detection** (Vesta Corporation), 590,540 card-not-present e-commerce transactions
- 434 columns after joining transaction and identity tables
- **3.5% positive class** (27:1 imbalance)
- 182-day window. `transactiondt` is a seconds offset from a reference point Vesta removed, not a calendar date
- Columns are anonymised. Meanings below are Vesta's published hints plus community reverse-engineering, not documentation

---

## Approach

**Split first, before any EDA.** Sorted by time, last 20% held out — 472,432 train / 118,108 test. Every EDA output becomes a decision, and a decision fitted on data you looked at is a decision that saw the test set.

`TimeSeriesSplit(n_splits=3)` for cross-validation. A random split would train on December to predict October.

**Metric: PR-AUC.** At 3.5% positives, predicting all-negative scores 96.5% accuracy. ROC-AUC divides false positives by 455,833 negatives, so a few thousand mistakes barely move it — while they'd fill a 500-row review queue four times over.

Random PR-AUC on this data is **0.035**, not 0.5. Every score below should be read against that.

---

## Model ladder

Same CV object, same features, only the model changes:

| model | cv PR-AUC | train PR-AUC | cv ROC-AUC | seconds |
|---|---|---|---|---|
| XGBoost | 0.5569 | 0.9850 | 0.8776 | 172.5 |
| **HistGradientBoosting** | **0.5502** | **0.8197** | **0.8931** | **170.5** |
| RandomForest (depth 20) | 0.5073 | 0.9250 | 0.8760 | 695.9 |
| DecisionTree (depth 10) | 0.3264 | 0.4394 | 0.7560 | 74.5 |
| LogisticRegression (17 cols) | 0.2994 | 0.2324 | 0.7616 | 11.1 |

**Shipped HistGradientBoosting, not the top scorer.** XGBoost leads PR-AUC by 0.0067 — inside noise — while sitting at 0.985 on training data against HGB's 0.820. HGB also wins ROC-AUC outright, runs in the same time, is native to sklearn with no extra dependency, and is far less overfit.

Unconstrained RandomForest scored 0.5672 with **train PR-AUC of 1.0000** — complete memorisation of 472,432 rows. Capping depth at 20 dropped it to 0.5073, below both boosters.

---

## Held-out test

| metric | cv | test |
|---|---|---|
| PR-AUC | 0.5502 | 0.5179 |
| ROC-AUC | 0.8931 | 0.8979 |

Test PR-AUC 0.5179 against a test base rate of 0.0344 is roughly **15x random**.

---


## Selected EDA findings

- **Missingness is structural, not random.** 12 columns are >90% empty, and every one carries signal. `d7` runs 2.7% fraud when blank against 15.0% when populated.
- **The V-block was engineered in groups.** 
- **Identity presence.** 25.5% of transactions carry identity data. Fraud rate is 2.13% without, 7.55% with — a 3.5x lift from one binary flag.
- **Hour of day.** Fraud runs 2.3% at hour 13 and 10.6% at hour 7.
- **Blanks per row.** Rows with the fewest blanks are 7.89% fraud against 1.58% in the middle bucket. Non-linear, so trees use it and the logistic model doesn't.

---

## Repo

```
notebook.ipynb            full analysis, Stage 0 to 7, with outputs
dashboard/
  fraud_dashboard.pbix    interactive review-queue dashboard
  screenshot.png
data/
  scored_test.csv         118,108 scored test transactions
  budget_curve.csv        recovery by review budget
model.pkl                 fitted pipeline
requirements.txt
```

Raw data is not committed. Download from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection).

---

## What I'd do with more time

- Proper hyperparameter search. The ladder used near-default parameters throughout.
- CatBoost, which handles high-cardinality categoricals natively via ordered target statistics.
- Cost-sensitive thresholding using the real cost of an analyst hour against the real cost of a chargeback, rather than a fixed 500-row budget.

