# iML 26/27 Exercise 1: Quantifying interpretability

My metric is the Cognitive Tree Interpretability Score (CTIS). The idea is my own, and AI
helped me work it out. It measures how much a person has to read to explain one prediction of
a tree model, and it works for single decision trees, random forests and gradient boosting. I
tested it on three credit datasets and used the results to propose a threshold of
interpretability. The slides are in `presentation/presentation.pdf`.

## Notes for grading

Hello! A few things that might help when you look at this project:

- Light and Mid are in `data/`. Hard is too big for GitHub (166 MB), so please download
  `application_train.csv` from [Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data),
  save it as `data/hard_cardclient.csv` and run the commands under [Running it](#running-it). The
  experiment takes about 4 minutes, almost all of it on the Hard dataset. I ran everything
  with uv and Python 3.12 on macOS.
- The metric is the class `CTIS` in `iml_interpretability/metric.py`. In `tests/test_ctis.py`
  there are small trees whose numbers you can check by hand.
- The threshold of 50 comes from my results, not from a user study. I chose the weights in the
  formula with help from AI, and they are not fitted to anything. The last slide lists this
  and the other limitations.
- Accuracy alone makes the readable trees look fine. On Light and Hard they mostly predict
  "no risk", which only shows up in the recall.
- `load_titanic` and `RelativeNumberOfLeaves` are from the course template. I left them in,
  but the experiment does not use them.

Thanks for taking a look!

## The metric

For every training observation I follow its path from the root to the leaf and count

- `d`: the decisions on the path,
- `s`: how often the feature changes from one decision to the next,
- `r`: how often a feature shows up again after its first decision on the path.

```text
p(x) = d(x) + 0.5 * s(x) + 0.25 * r(x)
CTIS = 100 * exp(-(0.7 * mean(p) + 0.3 * log2(1 + L)) / 6)
```

`mean(p)` is the average over the training observations, so paths that many observations take
count more. `L` is the number of leaves. The path `[A, B, A]` has `d=3`, `s=2`, `r=1`, so
`p=4.25`. Higher CTIS means easier to read. A single leaf gets the maximum of about 95.

For an ensemble you have to read one path per tree, so `d`, `s` and `r` are summed over all
trees and `L` counts the leaves of all trees. Switches and repetitions only count inside one
tree. A single tree is just an ensemble with one tree, so all scores are comparable.

The code is in `iml_interpretability/metric.py`.

## Experiment

### Data

| Name  | File in `data/`        | Rows    | Features | Target                       | Source |
|-------|------------------------|---------|----------|------------------------------|--------|
| Light | `light_cardclient.csv` | 1,000   | 20       | `Kreditwuerdigkeit`          | [Statlog (German Credit Data), UCI](https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data) |
| Mid   | `mid_cardclients.csv`  | 30,000  | 23       | `default payment next month` | [Default of Credit Card Clients, UCI](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| Hard  | `hard_cardclient.csv`  | 307,511 | 120      | `TARGET`                     | [Home Credit Default Risk, Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data) |

- Light is `german.data` from UCI, saved as a CSV with `;` as separator and German column names
  (`Kontostatus`, `Laufzeit_Monate`, ..., `Kreditwuerdigkeit`). The category codes (`A11`, ...)
  are unchanged.
- Mid is the UCI Excel file saved as CSV. The first row still holds `X1`, `X2`, ..., the second
  row the real column names.
- Hard is `application_train.csv` from Kaggle, renamed. You need a Kaggle account for it, and it
  is not in the repository because it is larger than GitHub's 100 MB limit.

### Models and split

Models: decision trees with max depth 1, 3, 5, 10 and no limit, plus random forests and
gradient boosting with 3, 10 and 100 trees of depth 3. Each dataset gets a stratified 75/25
split with seed 0. Missing numbers are filled with the training median, categories are one-hot
encoded, and every one-hot column counts as its own feature. Customer IDs are dropped.

Precision and recall refer to the risk class: bad credit risk (class 2) for Light, default or
payment difficulties (class 1) for Mid and Hard.

## Results

- Only trees of depth 1 and 3 score above 50. Three trees of depth 3 already drop to about 20,
  and ensembles with 10 or more trees end up near 0.
- Accuracy hardly changes with CTIS. The best model below 50 beats the depth-3 tree by 2.4
  points on Light and by at most 0.1 points on Mid and Hard.
- Recall does change. On Light and Hard the readable trees almost only predict "no risk". The
  depth-3 tree finds 5.3 % and 0 % of the risk cases, deeper trees up to 46.7 % and 17.5 %.

I propose CTIS 50 as the threshold, which is about one tree of depth 3. This comes from the
results above and is not tested with people. The weights in the formula are my own choice.

## Running it

Put `hard_cardclient.csv` into `data/` first (see [Data](#data)), then:

```bash
uv sync --extra dev
uv run pytest
uv run python -m scripts.run_experiment      # writes results/scores.csv (about 4 minutes)
uv run python -m scripts.plot_results        # charts in results/metric_plots/
uv run python -m scripts.plot_data_overview  # charts in results/data_plots/
uv run python -m scripts.plot_trees          # depth-3 trees in results/trees/
```

With pip instead of uv: `pip install -e ".[dev]"`. Run everything from the repository root.
`run_experiment` takes `--data-dir` and `--output` if the data is somewhere else.

The tree diagrams use German labels (`scripts/tree_labels.py`, based on the UCI descriptions of
[German Credit](https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data) and
[Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
and the [Home Credit](https://www.kaggle.com/competitions/home-credit-default-risk/data) column
descriptions). One-hot splits are shown as "category present?" with "Nein" on the left.

## Files

```
iml_interpretability/
  data.py       loading the CSVs
  models.py     the 11 models with preprocessing
  metric.py     CTIS
  evaluate.py   fits every model on every dataset
scripts/        experiment and plots
tests/          tests for CTIS, the evaluation and the tree labels
presentation/   presentation.pdf
```

## AI statement

The idea for CTIS is my own. AI helped me turn it into the formula. I used Claude Code
(Claude Opus 5.5) to generate the code and its implementation in this repository, and it
helped me create the presentation.
