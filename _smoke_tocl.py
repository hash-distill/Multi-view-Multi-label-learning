"""Smoke test for the TOCL entry point.

TOCL performs a tensor SVD per iteration, so it is the slowest algorithm and a
full-budget cross-validation can take many minutes per fold. This script caps
the iteration budget (``MVML_MAX_ITER``) and checks the algorithm's contract on
two datasets:

  * the ranking is a permutation of every feature index
  * the ranking length equals the concatenated feature count
  * MLkNN evaluation runs end to end

Run with:
    MVML_MAX_ITER=15 python _smoke_tocl.py
"""

import os
import time

from main import DATASETS, DATA_DIR, evaluate_ranking, get_data, prune_labels, split_data_fold

os.environ.setdefault("MVML_MAX_ITER", "15")

from alg.TOCL import view7  # noqa: E402  (import after the env default)

for ds in ["3sources", "emotions"]:
    filename, discretize = DATASETS[ds]
    X, view_dims, Y = get_data(DATA_DIR / filename, discretize)
    trainX, Ytrain, testX, Ytest = split_data_fold(
        X, Y, int(round(len(Y) / 5)), 0, shuffle=True)
    trainX, Ytrain, testX, Ytest, _ = prune_labels(Ytrain, Ytest, trainX, testX)

    t0 = time.time()
    record, n_iter = view7(trainX, view_dims, Ytrain, ds, 1.0, 1.0, 1.0, 1.0)
    ranked = len(record["idx"])
    assert ranked == X.shape[1], (ranked, X.shape[1])
    assert sorted(record["idx"]) == list(range(X.shape[1])), "ranking is not a permutation"

    scores = evaluate_ranking(trainX, Ytrain, testX, Ytest, record["idx"], 5)
    print(f"{ds:10s} iters={n_iter:3d} time={time.time() - t0:6.1f}s "
          f"ranked={ranked:5d} AP@5={scores['average_precision'][-1]:.4f}")

print("TOCL smoke test OK")
