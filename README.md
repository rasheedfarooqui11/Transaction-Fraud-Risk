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

A 6% PR-AUC drop on an out-of-time split. The model was fit on earlier data and evaluated on later data it never saw, so some degradation is expected — a test score matching cv exactly would be more suspicious than this. ROC-AUC barely moved, which is the same blind spot showing up again.

Test PR-AUC 0.5179 against a test base rate of 0.0344 is roughly **15x random**.

---

## Things that were tried and cut

Both are in the notebook with the numbers.

**UID reconstruction.** Vesta removed the customer identifier. `day_index - d1` gives the day a card first appeared, constant per card, so `card1 + addr1 + d1n` acts as a fingerprint — the same date-minus-counter trick as SQL gaps-and-islands. Coverage killed it:

| grouping | groups | test coverage |
|---|---|---|
| card1 | 12,730 | 98.9% |
| card1 + addr1 | 34,399 | 86.0% |
| card1 + addr1 + d1n | 167,111 | 41.6% |

The specific version is NaN for 58% of test rows — cards churn over six months. The coarse version has coverage but isn't a card: `card1` averages 43.6 transactions per value and one value covers 14,932 rows, making it a BIN rather than a cardholder. Its mean is a population average, which defeats the point of a deviation feature.

Either specific and unavailable at test time, or available and meaningless.

**SelectKBest feature filtering.** Scored materially worse. The filter needs complete data, so an imputer has to sit in front of it — which overwrites the informative missingness in the `id_` block that HGB was routing natively. Filters help models that can't ignore irrelevant features; boosters can.

Correlation pruning was kept: 71 V-columns dropped at r > 0.95, via `feature_engine.DropCorrelatedFeatures` fitted on train only.

---

## Selected EDA findings

- **Missingness is structural, not random.** 12 columns are >90% empty, and every one carries signal. `d7` runs 2.7% fraud when blank against 15.0% when populated.
- **The V-block was engineered in groups.** 339 V-columns share only 14 distinct missing rates, clustering into 47 correlated groups — consecutive runs go missing on identical rows.
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

- Nested UID aggregations at multiple granularities, so coarse groups cover the rows fine groups miss. This is where the competition winners got their gains.
- Proper hyperparameter search. The ladder used near-default parameters throughout.
- CatBoost, which handles high-cardinality categoricals natively via ordered target statistics.
- Cost-sensitive thresholding using the real cost of an analyst hour against the real cost of a chargeback, rather than a fixed 500-row budget.

## Production notes

Not built, but the shape: nightly batch scoring, feature-distribution drift monitoring with alerting, quarterly retrain triggered by PR-AUC decay, versioned model artefacts. The 6% out-of-time drop is the argument for the retrain cadence.
