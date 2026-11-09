# Baseline Modeling — Experimental Conclusions

## Objective
Compare classical ML algorithms for LendingClub default
prediction on the fixed chronological validation period.

## Evaluation policy
- Train: January 2013–September 2015
- Validation: October–December 2015
- Final test: 2016 (untouched)
- Primary selection metric: PR-AUC
- Secondary: ROC-AUC, F1, recall, risk concentration

## Model comparison

| model                  | configuration                                           |   accuracy |   balanced_accuracy |   precision_default |   recall_default |   f1_default |   roc_auc |   pr_auc |
|:-----------------------|:--------------------------------------------------------|-----------:|--------------------:|--------------------:|-----------------:|-------------:|----------:|---------:|
| Dummy (prior)          | No borrower features                                    |     0.8516 |              0.5    |              0      |           0      |       0      |    0.5    |   0.1484 |
| Logistic Regression    | Original features; L2; unweighted                       |     0.8514 |              0.5013 |              0.4259 |           0.0035 |       0.0069 |    0.6828 |   0.2642 |
| Decision Tree          | Depth=12; leaf=100; unweighted                          |     0.8509 |              0.5019 |              0.3561 |           0.0055 |       0.0109 |    0.6416 |   0.2277 |
| Random Forest          | 120 trees; depth=14; leaf=50; weight=None               |     0.8516 |              0.5    |              0      |           0      |       0      |    0.6847 |   0.2674 |
| Random Forest Balanced | 120 trees; depth=14; leaf=50; weight=balanced_subsample |     0.673  |              0.6249 |              0.2403 |           0.5566 |       0.3357 |    0.681  |   0.2633 |

## XGBoost engineered-feature ablation

| model            |   roc_auc |   pr_auc |   best_val_f1 |   best_val_threshold |   top10_lift |
|:-----------------|----------:|---------:|--------------:|---------------------:|-------------:|
| Original XGBoost |    0.6993 |   0.2818 |        0.3529 |               0.16   |       2.2734 |
| Enhanced XGBoost |    0.6991 |   0.2818 |        0.3507 |               0.1497 |       2.262  |

## Findings and decisions

1. Dummy shows high majority-class accuracy but
   zero default-class detection.

2. Logistic Regression learns predictive ranking
   information, although threshold 0.50 is unsuitable
   for detecting many defaults.

3. Decision Tree underperforms Logistic Regression
   on ranking metrics.

4. Random Forest improves upon the single Decision Tree.

5. Balanced Random Forest improves recall at the
   default threshold but does not improve PR-AUC
   over standard Random Forest.

6. XGBoost has the strongest baseline PR-AUC and ROC-AUC.

7. Two engineered features produce negligible improvement.

## Decision

Carry forward original-feature XGBoost as the leading
candidate for targeted feature experiments.

Retain LR and RF as meaningful comparative baselines.

## Limitations

- Validation-tuned F1 thresholds are exploratory.
- Only resolved, issued 36-month loans are modeled.
- Historical results do not guarantee future performance.
- Excluding lender grade/rate changes the prediction task.
- Final 2016 holdout evaluation has not yet occurred.
