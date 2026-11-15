"""Run the interpretability evaluation and store the results as a CSV file.

Usage::

    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from iml_interpretability import CTIS, evaluate, get_models, load_credit_datasets

OUTPUT = Path("results/scores.csv")
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CTIS on the three credit datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    results = evaluate(
        datasets=load_credit_datasets(args.data_dir),
        models=get_models(seed=SEED),
        metrics=[CTIS()],
        seed=SEED,
        positive_labels={"light_cardclient": 2, "mid_cardclients": 1, "hard_cardclient": 1},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
