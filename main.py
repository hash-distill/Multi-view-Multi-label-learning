"""
Multi-view Multi-label feature selection evaluation framework.

Replaces the original Windows-only, single-algorithm smoke-test script with a
portable, configurable runner:

  data/*.mat -> feature ranking (alg/*.py) -> top-k features -> MLkNN -> 5 metrics

Usage:
    python main.py --alg DHLI --data emotions
    python main.py --alg TOCL --data emotions yeast --folds 5
    python main.py --list

Outputs (results/):
    rankings_<alg>_<dataset>.csv   full feature ranking + per-view breakdown
    metrics_<alg>_<dataset>.csv    metric-vs-#features curves
    summary_<alg>.csv              one row per dataset (best value + AUC per metric)
    summary_all.csv                appended across runs/algorithms
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io as io

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

# --------------------------------------------------------------------------- #
# Registries
# --------------------------------------------------------------------------- #

#: dataset -> (filename, discretize)
DATASETS: dict[str, tuple[str, bool]] = {
    "emotions": ("emotions.mat", True),
    "yeast": ("yeast.mat", True),
    "3sources": ("3sources.mat", False),
    "SCENE": ("SCENE.mat", False),
    "OBJECT": ("OBJECT.mat", False),
    "VOC07": ("VOC07.mat", False),
    "MIRFlickr": ("MIRFlickr.mat", False),
    "corel5k_5": ("corel5k_5.mat", False),
    "iaprtc12": ("iaprtc12.mat", False),
    "espgame": ("espgame.mat", False),
}

#: algorithm -> (module, entry function)
ALGORITHMS: dict[str, tuple[str, str]] = {
    "DHLI": ("alg.DHLI", "DHLI"),
    "TOCL": ("alg.TOCL", "view7"),
    "EF2FS": ("alg.EF2FS", "EF2FS"),
    "GRAFS": ("alg.GRAFS", "view6"),
    "I2VSLC": ("alg.I2VSLC", "I2VSLC"),
    "UGRFS": ("alg.UGRFS", "UGRFS"),
}

#: extra positional args required by some entry points, in call order
EXTRA_ARGS: dict[str, tuple[str, ...]] = {
    "EF2FS": ("V_dim",),
    "GRAFS": ("kk",),
}

#: per-dataset hyper-parameters (alpha, beta, gamma, lamb) + extras
DEFAULT_PARAMS: dict[str, dict] = {
    "DHLI": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lamb": 1.0},
    "TOCL": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lamb": 1.0},
    "EF2FS": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lamb": 1.0, "V_dim": 30},
    "GRAFS": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lamb": 1.0, "kk": 20},
    "I2VSLC": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lamb": 1.0},
    "UGRFS": {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lamb": 1.0},
}

METRICS = [
    ("hamming_loss", "HL", False),                 # lower is better
    ("label_ranking_loss", "RL", False),           # lower is better
    ("coverage_error", "CV", False),               # lower is better
    ("average_precision", "AP", True),             # higher is better
    ("zero_one_loss", "ZL", False),                # lower is better
]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def read_general_mat_data(filename: Path) -> dict:
    """Load a .mat file, dropping the MATLAB header keys."""
    mat = io.loadmat(str(filename))
    keys = [k for k in mat.keys() if not k.startswith("__")]
    return {k: mat[k] for k in keys}


def discrete_3(X: np.ndarray) -> np.ndarray:
    """Equal-frequency 3-bin discretization, column by column."""
    import pandas as pd

    out = np.zeros_like(X)
    for i in range(X.shape[1]):
        codes = pd.cut(X[:, i], 3, labels=[0, 1, 2]).codes
        out[:, i] = np.where(codes < 0, 0, codes)
    return out


def get_data(filename: Path, discretize: bool = False):
    """Return (X, view_dims, Y) with views concatenated column-wise.

    ``view_dims`` is an int array of per-view feature counts; it is consumed in
    order by every algorithm in ``alg/``.
    """
    mat = read_general_mat_data(filename)
    if "view" not in mat:
        raise KeyError(f"{filename.name}: no 'view' variable found")
    view = mat["view"]
    n_view = view.shape[1]

    parts = [np.asarray(view[0][i], dtype=float) for i in range(n_view)]
    X = np.hstack(parts) if len(parts) > 1 else parts[0]

    if "features" in mat:
        view_dims = np.asarray(mat["features"], dtype=int).ravel()
    else:
        view_dims = np.array([p.shape[1] for p in parts], dtype=int)

    if int(view_dims.sum()) != X.shape[1]:
        raise ValueError(
            f"{filename.name}: view dims {view_dims.tolist()} sum to "
            f"{int(view_dims.sum())} but X has {X.shape[1]} columns"
        )

    if discretize:
        X = discrete_3(X)

    if "label" not in mat:
        raise KeyError(f"{filename.name}: no 'label' variable found")
    Y = np.asarray(mat["label"], dtype=float)
    return X, view_dims, Y


# --------------------------------------------------------------------------- #
# Cross validation
# --------------------------------------------------------------------------- #

def split_data_fold(X, Y, n_test, i, rng=None, shuffle=False):
    """Contiguous (or shuffled) hold-out fold.

    The original implementation shuffled ``X`` in place but never ``Y``, which
    desynchronised features from labels. We now permute a shared index array
    with an explicit RNG so X and Y always move together.
    """
    m = Y.shape[0]
    indices = np.arange(m)
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(0)
        rng.shuffle(indices)

    start = i * n_test
    stop = min(start + n_test, m)
    test_idx = indices[start:stop]
    train_idx = np.setdiff1d(indices, test_idx, assume_unique=False)

    return X[train_idx, :], Y[train_idx, :], X[test_idx, :], Y[test_idx, :]


def prune_labels(Ytrain, Ytest, trainX, testX):
    """Drop training rows with no positive label and labels absent from train."""
    keep_rows = np.any(Ytrain, axis=1)
    keep_cols = np.any(Ytrain, axis=0)

    Ytrain = Ytrain[keep_rows][:, keep_cols]
    Ytest = Ytest[:, keep_cols]
    trainX = trainX[keep_rows]
    return trainX, Ytrain, testX, Ytest, keep_cols


def read_data(X: np.ndarray, F) -> np.ndarray:
    """Select the columns of ``X`` listed in ``F`` (feature indices)."""
    idx = [int(float(i)) for i in F]
    return X[:, idx]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def _make_mlknn():
    """Build an MLkNN classifier compatible with modern scikit-learn.

    ``scikit-multilearn`` 0.2.0 instantiates ``NearestNeighbors(self.k)``
    positionally; scikit-learn >= 1.0 only accepts keyword parameters and raises
    ``TypeError: NearestNeighbors.__init__() takes 1 positional argument but 2
    were given``. We subclass MLkNN and reimplement only ``_compute_cond`` with
    the keyword form, keeping the original probability estimates untouched.
    """
    import scipy.sparse as sparse
    from skmultilearn.adapt import MLkNN
    from skmultilearn.utils import get_matrix_in_format
    from sklearn.neighbors import NearestNeighbors

    class _MLkNN(MLkNN):
        def _compute_cond(self, X, y):
            self.knn_ = NearestNeighbors(n_neighbors=self.k).fit(X)

            c = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='i8')
            cn = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='i8')
            label_info = get_matrix_in_format(y, 'dok')

            neighbors = [
                a[self.ignore_first_neighbours:]
                for a in self.knn_.kneighbors(
                    X, self.k + self.ignore_first_neighbours, return_distance=False)
            ]

            for instance in range(self._num_instances):
                deltas = label_info[neighbors[instance], :].sum(axis=0)
                for label in range(self._num_labels):
                    if label_info[instance, label] == 1:
                        c[label, int(deltas[0, label])] += 1
                    else:
                        cn[label, int(deltas[0, label])] += 1

            c_sum = c.sum(axis=1)
            cn_sum = cn.sum(axis=1)

            cond_prob_true = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='float')
            cond_prob_false = sparse.lil_matrix((self._num_labels, self.k + 1), dtype='float')
            for label in range(self._num_labels):
                for neighbor in range(self.k + 1):
                    cond_prob_true[label, neighbor] = (
                        self.s + c[label, neighbor]) / (self.s * (self.k + 1) + c_sum[label, 0])
                    cond_prob_false[label, neighbor] = (
                        self.s + cn[label, neighbor]) / (self.s * (self.k + 1) + cn_sum[label, 0])
            return cond_prob_true, cond_prob_false

        def _neighbor_count(self, X):
            """Per-instance label counts over the k nearest neighbours."""
            neighbors = [
                a[self.ignore_first_neighbours:]
                for a in self.knn_.kneighbors(
                    X, self.k + self.ignore_first_neighbours, return_distance=False)
            ]
            for instance in range(X.shape[0]):
                deltas = self._label_cache[neighbors[instance], ].sum(axis=0)
                yield deltas

        def predict(self, X):
            result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype='i8')
            for instance, deltas in enumerate(self._neighbor_count(X)):
                for label in range(self._num_labels):
                    idx = int(deltas[0, label])
                    p_true = float(self._prior_prob_true[label]) * self._cond_prob_true[label, idx]
                    p_false = float(self._prior_prob_false[label]) * self._cond_prob_false[label, idx]
                    result[instance, label] = int(p_true >= p_false)
            return result

        def predict_proba(self, X):
            result = sparse.lil_matrix((X.shape[0], self._num_labels), dtype='float')
            for instance, deltas in enumerate(self._neighbor_count(X)):
                for label in range(self._num_labels):
                    idx = int(deltas[0, label])
                    result[instance, label] = (
                        float(self._prior_prob_true[label]) * self._cond_prob_true[label, idx])
            return result

    return _MLkNN()


def evaluate_ranking(trainX, Ytrain, testX, Ytest, ranking, select_num):
    """Run MLkNN for k = 1..select_num and return averaged metric curves."""
    from sklearn.metrics import (
        coverage_error,
        hamming_loss,
        label_ranking_average_precision_score,
        label_ranking_loss,
        zero_one_loss,
    )

    scores = {name: np.zeros(select_num) for name, _, _ in METRICS}

    for k in range(1, select_num + 1):
        Fk = ranking[:k]
        data_tr = read_data(trainX, Fk)
        data_te = read_data(testX, Fk)

        clf = _make_mlknn()
        clf.fit(data_tr, Ytrain)
        pred = clf.predict(data_te)
        proba = clf.predict_proba(data_te).toarray()

        scores["hamming_loss"][k - 1] = hamming_loss(Ytest, pred)
        scores["label_ranking_loss"][k - 1] = label_ranking_loss(Ytest, proba)
        scores["coverage_error"][k - 1] = coverage_error(Ytest, proba)
        scores["average_precision"][k - 1] = label_ranking_average_precision_score(Ytest, proba)
        scores["zero_one_loss"][k - 1] = zero_one_loss(Ytest, pred)

    return scores


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def save_run(alg, dataset, ranking, view_dims, scores, select_num, record,
             cv_summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    params = record.get("param", {})

    # 1) feature ranking, 1-based rank with the originating view
    view_of = np.zeros(len(ranking), dtype=int)
    offset = 0
    for vi, dim in enumerate(view_dims):
        view_of[offset:offset + dim] = vi
        offset += dim
    rows = [(r + 1, int(f), int(view_of[f])) for r, f in enumerate(ranking)]
    write_csv(out_dir / f"rankings_{alg}_{dataset}.csv",
              ["rank", "feature_index", "view"], rows)

    # 2) metric curves over the number of selected features
    header = ["num_features"] + [short for _, short, _ in METRICS]
    rows = []
    for k in range(select_num):
        rows.append([k + 1] + [float(scores[name][k]) for name, _, _ in METRICS])
    write_csv(out_dir / f"metrics_{alg}_{dataset}.csv", header, rows)

    # 3) per-dataset summary: best value and area under the curve per metric
    summary = {"algorithm": alg, "dataset": dataset,
               "n_samples": int(record.get("n_samples", 0)),
               "n_features": int(len(ranking)),
               "n_labels": int(record.get("n_labels", 0)),
               "n_views": int(len(view_dims)),
               "select_num": select_num,
               "iterations": int(record.get("iter", 0)),
               "running_time_sec": round(float(record.get("running_time", float("nan"))), 3),
               "cv_folds": cv_summary.get("folds", 0),
               "cv_time_sec": round(float(cv_summary.get("time", float("nan"))), 3)}
    for name, short, higher_better in METRICS:
        curve = np.asarray(scores[name], dtype=float)
        summary[f"{short}_best"] = float(np.nanmax(curve) if higher_better else np.nanmin(curve))
        summary[f"{short}_best_k"] = int(np.nanargmax(curve) if higher_better else np.nanargmin(curve)) + 1
        summary[f"{short}_auc"] = float(np.trapz(curve, dx=1.0) / max(len(curve) - 1, 1))
    for key, val in sorted(params.items()):
        summary[f"param_{key}"] = val

    write_csv(out_dir / f"summary_{alg}.csv", list(summary.keys()), [list(summary.values())])

    # 4) global summary, appended across runs
    all_path = out_dir / "summary_all.csv"
    exists = all_path.exists()
    with all_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(list(summary.keys()))
        writer.writerow(list(summary.values()))

    return summary


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def resolve_algorithm(alg: str):
    if alg not in ALGORITHMS:
        raise SystemExit(f"unknown algorithm {alg!r}; choose from {sorted(ALGORITHMS)}")
    module_name, func_name = ALGORITHMS[alg]
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def run_selfcheck(subset: int = 200, seed: int = 100) -> int:
    """Import every algorithm and call it on a small slice of every dataset.

    Catches the failure modes that used to crash this project: missing
    dependencies, wrong entry-point names, wrong extra arguments, float feature
    indices, view-dimension mismatches and non-permutation rankings. It does not
    run MLkNN; use a normal run for that.
    """
    failures = 0
    for name, (filename, discretize) in sorted(DATASETS.items()):
        path = DATA_DIR / filename
        if not path.exists():
            print(f"[SKIP] {name}: {path} missing")
            continue
        try:
            X, view_dims, Y = get_data(path, discretize)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {name}: load error {type(exc).__name__}: {exc}")
            failures += 1
            continue

        n = min(subset, X.shape[0])
        Xs, Ys = X[:n], Y[:n]
        line = [f"{name:12s} X={str(X.shape):14s} views={view_dims.tolist()}"]
        for alg in sorted(ALGORITHMS):
            entry = resolve_algorithm(alg)
            params = DEFAULT_PARAMS[alg]
            extras = [params[k] for k in EXTRA_ARGS.get(alg, ())]
            try:
                rec, _ = entry(Xs, view_dims, Ys, name, params["alpha"],
                               params["beta"], params["gamma"], params["lamb"], *extras)
                idx = np.asarray(rec["idx"])
                ok = (len(idx) == X.shape[1]
                      and np.issubdtype(idx.dtype, np.integer)
                      and sorted(idx.tolist()) == list(range(X.shape[1])))
                line.append(f"{alg}:{'ok' if ok else 'BAD_RANKING'}")
                if not ok:
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                line.append(f"{alg}:FAIL({type(exc).__name__})")
                failures += 1
        print("  ".join(line))

    print(f"self-check: {'OK' if failures == 0 else str(failures) + ' failure(s)'}")
    return 1 if failures else 0


def run_one(alg, dataset, params, folds, ratio, select_ratio, shuffle, seed,
            out_dir, verbose=True):
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    filename, discretize = DATASETS[dataset]
    path = DATA_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing dataset file: {path}")

    entry = resolve_algorithm(alg)
    X, view_dims, Y = get_data(path, discretize)
    if verbose:
        print(f"[{alg}] {dataset}: X={X.shape} views={view_dims.tolist()} Y={Y.shape}")

    m = Y.shape[0]
    n_test = int(round(m / folds))
    select_num = max(1, int(X.shape[1] * select_ratio))

    score_mat = {name: np.zeros(select_num) for name, _, _ in METRICS}
    ranking = None
    record = {}
    used_folds = 0
    t0 = time.time()

    for j in range(folds):
        rng = np.random.default_rng(seed + j)
        trainX, Ytrain, testX, Ytest = split_data_fold(
            X, Y, n_test, j, rng=rng, shuffle=shuffle)
        trainX, Ytrain, testX, Ytest, _ = prune_labels(Ytrain, Ytest, trainX, testX)

        if Ytrain.shape[1] == 0 or not np.any(Ytrain):
            print(f"  fold {j}: skipped (no usable labels after pruning)")
            continue

        call_args = {k: params[k] for k in ("alpha", "beta", "gamma", "lamb")}
        extras = [params[k] for k in EXTRA_ARGS.get(alg, ())]
        rec, n_iter = entry(trainX, view_dims, Ytrain, dataset,
                            call_args["alpha"], call_args["beta"],
                            call_args["gamma"], call_args["lamb"], *extras)

        if ranking is None:
            ranking = np.asarray(rec["idx"], dtype=int)
            record = dict(rec)
            record["iter"] = n_iter
            record["n_samples"] = m
            record["n_labels"] = Y.shape[1]

        if verbose:
            print(f"  fold {j}/{folds}: {n_iter} iters, "
                  f"{rec['running_time']:.2f}s, {len(rec['idx'])} features ranked")

        fold_scores = evaluate_ranking(trainX, Ytrain, testX, Ytest, rec["idx"], select_num)
        # average the metric curves of every fold, each computed from that
        # fold's own ranking (the original averaged curves produced by
        # different rankings while reporting only the last fold's ranking)
        for name in score_mat:
            score_mat[name] = (score_mat[name] * used_folds + fold_scores[name]) / (used_folds + 1)
        used_folds += 1

    if ranking is None:
        raise SystemExit("no fold produced a usable feature ranking")

    summary = save_run(alg, dataset, ranking, view_dims, score_mat, select_num,
                       record, {"folds": used_folds, "time": time.time() - t0}, out_dir)
    if verbose:
        print(f"  -> best AP={summary['AP_best']:.4f} @k={summary['AP_best_k']}, "
              f"HL={summary['HL_best']:.4f}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Multi-view multi-label feature selection evaluation")
    parser.add_argument("--alg", default="DHLI", help="algorithm name, or 'all'")
    parser.add_argument("--data", nargs="+", default=["emotions"],
                        help="dataset name(s), or 'all'")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ratio", type=float, default=0.2, help="test ratio per fold")
    parser.add_argument("--select-ratio", type=float,
                        default=float(os.environ.get("MVML_SELECT_RATIO", 0.2)))
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle before folding (X and Y stay aligned)")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=str(RESULT_DIR))
    parser.add_argument("--list", action="store_true", help="list datasets/algorithms")
    parser.add_argument("--check", action="store_true",
                        help="self-check: run every algorithm on a slice of every dataset")
    parser.add_argument("--check-samples", type=int, default=200,
                        help="instances per dataset used by --check")
    args = parser.parse_args(argv)

    if args.list:
        print("datasets  :", ", ".join(sorted(DATASETS)))
        print("algorithms:", ", ".join(sorted(ALGORITHMS)))
        return 0

    if args.check:
        return run_selfcheck(subset=args.check_samples, seed=args.seed)

    algs = sorted(ALGORITHMS) if args.alg == "all" else [args.alg]
    data = sorted(DATASETS) if args.data == ["all"] else args.data
    out_dir = Path(args.out)

    start = time.localtime()
    for alg in algs:
        params = dict(DEFAULT_PARAMS[alg])
        for ds in data:
            run_one(alg, ds, params, args.folds, args.ratio, args.select_ratio,
                    args.shuffle, args.seed, out_dir)
    end = time.localtime()
    print("start time =", time.asctime(start))
    print("  end time =", time.asctime(end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
