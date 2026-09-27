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

#: the four metrics the TOCL (ACM MM 2025) and UGRFS (AAAI 2025) papers report.
#: ``ZL`` above is a code-only extra and has no counterpart in either paper.
PAPER_METRICS = ("AP", "CV", "HL", "RL")

#: "feature percentages ranging from 1% to 20%" -- both papers scan this range.
PAPER_PCTS = tuple(range(1, 21))


def paper_aggregate(curve, pcts=PAPER_PCTS, select_ratio=0.2):
    """Collapse a metric-vs-k curve into the scalar the papers tabulate.

    Both papers scan feature percentages 1%..20% and report one mean value per
    dataset. ``curve[k - 1]`` holds the metric at ``k`` selected features, and
    the run used ``select_num = int(n_features * select_ratio)``, so the total
    feature count is recovered as ``len(curve) / select_ratio`` and each
    percentage maps back to an absolute ``k``.

    NOTE: that the papers average over the 20 percentage points is inferred --
    neither PDF states its aggregation rule explicitly.
    """
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return float("nan")
    total = curve.size / float(select_ratio)
    idx = np.clip([int(round(p / 100.0 * total)) - 1 for p in pcts], 0, curve.size - 1)
    return float(np.nanmean(curve[idx]))


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

def split_data_fold(X, Y, n_test, i, rng=None, shuffle=False, indices=None):
    """Hold-out fold ``i`` of a cross-validation partition.

    The original implementation shuffled ``X`` in place but never ``Y``, which
    desynchronised features from labels. We now permute a shared index array
    with an explicit RNG so X and Y always move together.

    Two modes of use:

    * ``indices`` supplied -- genuine k-fold. The caller builds **one** shared
      permutation and every fold takes a disjoint contiguous slice of it, so the
      folds partition the data exactly once. This is what :func:`run_one` does.
    * ``indices`` omitted -- legacy path. When ``shuffle`` is true the fold is
      drawn from a *fresh* permutation, so different values of ``i`` produce
      overlapping test sets and the folds do **not** form a partition. Kept only
      for direct callers such as ``_smoke_tocl.py``.
    """
    m = Y.shape[0]
    if indices is None:
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
# Fold-level checkpointing
# --------------------------------------------------------------------------- #

def run_cache_key(alg, dataset, params, folds, ratio, select_ratio, shuffle,
                  seed, view_order):
    """Stable hash of every knob that can change a run's numbers.

    Anything that would alter the result must be in here, otherwise a stale cache
    would silently be replayed for a different configuration.
    """
    import hashlib
    import json

    payload = {
        "alg": alg,
        "dataset": dataset,
        "params": {k: float(params[k]) for k in sorted(params)},
        "folds": int(folds),
        "ratio": None if ratio is None else float(ratio),
        "select_ratio": float(select_ratio),
        "shuffle": bool(shuffle),
        "seed": int(seed),
        "view_order": None if view_order is None else [int(k) for k in view_order],
        # iteration budget changes the result, so it belongs in the key too
        "max_iter": os.environ.get("MVML_MAX_ITER", ""),
        "seed_env": os.environ.get("MVML_SEED", ""),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_fold_cache(cache_dir: Path, select_num: int):
    """Return {fold_index: (curve_dict, ranking, n_iter, running_time)}."""
    import json

    out = {}
    if not cache_dir.is_dir():
        return out
    for meta_path in sorted(cache_dir.glob("fold_*.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            j = int(meta["fold"])
            if int(meta["select_num"]) != int(select_num):
                continue                      # different curve length -> unusable
            rank_path = cache_dir / f"fold_{j}_ranking.csv"
            curve_path = cache_dir / f"fold_{j}_curve.csv"
            if not (rank_path.exists() and curve_path.exists()):
                continue
            ranking = np.loadtxt(rank_path, dtype=np.int64, delimiter=",", ndmin=1)
            table = np.loadtxt(curve_path, dtype=np.float64, delimiter=",", ndmin=2)
            if table.shape != (select_num, len(METRICS)):
                continue
            curve = {name: table[:, i] for i, (name, _, _) in enumerate(METRICS)}
            out[j] = (curve, ranking, int(meta.get("iter", 0)),
                      float(meta.get("running_time", float("nan"))))
        except Exception:                      # noqa: BLE001 - a bad cache entry
            continue                           # is ignored, never fatal
    return out


def save_fold_cache(cache_dir: Path, fold: int, curve, ranking, n_iter, seconds):
    """Persist one completed fold so an interrupted run can resume from it."""
    import json

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_dir / f".fold_{fold}.tmp"
    # %.17g is the shortest format that round-trips a float64 exactly, so a
    # resumed run reproduces the interrupted one bit for bit. %.10g is NOT
    # enough: it perturbs the averaged curves at ~1e-11.
    np.savetxt(cache_dir / f"fold_{fold}_curve.csv",
               np.column_stack([np.asarray(curve[name], dtype=float)
                                for name, _, _ in METRICS]),
               delimiter=",", fmt="%.17g")
    np.savetxt(cache_dir / f"fold_{fold}_ranking.csv",
               np.asarray(ranking, dtype=np.int64), delimiter=",", fmt="%d")
    (cache_dir / f"fold_{fold}.json").write_text(json.dumps({
        "fold": int(fold), "iter": int(n_iter), "running_time": float(seconds),
        "select_num": int(len(next(iter(curve.values())))),
        "metrics": [short for _, short, _ in METRICS],
    }), encoding="utf-8")
    if tmp.exists():
        tmp.unlink()


def parse_view_order(text, n_views: int):
    """Parse a ``--view-order`` string into a validated permutation.

    ``order[k]`` is the 0-based index of the ORIGINAL view that should become
    view ``k``. Accepts e.g. ``"0,1,2,4,3"``.
    """
    try:
        order = [int(tok) for tok in str(text).replace(" ", "").split(",") if tok != ""]
    except ValueError:
        raise SystemExit(f"--view-order must be comma-separated integers, got {text!r}")
    if sorted(order) != list(range(n_views)):
        raise SystemExit(f"--view-order {order} is not a permutation of "
                         f"0..{n_views - 1} (this dataset has {n_views} views)")
    return order


def permute_views(X: np.ndarray, view_dims: np.ndarray, order):
    """Reorder the view blocks of ``X`` according to ``order``.

    Needed because the papers' Table 1 and the shipped ``.mat`` files disagree on
    the view order for the 5-view datasets: the papers list
    ``DH, DHV3H1, GIST, HHV3H1, HH`` (i.e. ``[100, 300, 512, 300, 100]``) while
    ``iaprtc12.mat`` / ``corel5k_5.mat`` hold ``DH, DHV3H1, GIST, HH, HHV3H1``
    (``[100, 300, 512, 100, 300]``). TOCL's tensor nuclear norm runs an FFT along
    the view axis, so it is sensitive to that ordering; UGRFS largely is not.
    """
    blocks, off = [], 0
    for dim in view_dims:
        blocks.append(X[:, off:off + int(dim)])
        off += int(dim)
    if len(order) != len(blocks):
        raise SystemExit(f"--view-order has {len(order)} entries but the dataset "
                         f"has {len(blocks)} views")
    dims = np.asarray([int(view_dims[k]) for k in order], dtype=int)
    return np.hstack([blocks[k] for k in order]), dims


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


def build_summary(alg, dataset, ranking, view_dims, scores, select_num, record,
                  cv_summary, fold_scores=None, select_ratio=None, view_order=None):
    """Assemble one per-dataset summary row.

    Besides ``*_best`` / ``*_best_k`` / ``*_auc`` (computed from the fold-averaged
    curve), this adds ``*_paper_mean`` / ``*_paper_std``: the aggregate over the
    1%..20% feature-percentage grid, with the mean and standard deviation taken
    **across folds** so the numbers line up with the ``mean +- std`` columns the
    TOCL and UGRFS papers tabulate.

    The std is only recoverable because ``fold_scores`` keeps each fold's curve;
    the fold-averaged ``scores`` alone cannot produce it.
    """
    params = record.get("param", {})
    summary = {"algorithm": alg, "dataset": dataset,
               "n_samples": int(record.get("n_samples", 0)),
               "n_features": int(len(ranking)),
               "n_labels": int(record.get("n_labels", 0)),
               "n_views": int(len(view_dims)),
               "select_num": select_num,
               "select_ratio": select_ratio,
               "view_order": ("native" if view_order is None
                              else ",".join(str(int(k)) for k in view_order)),
               "iterations": int(record.get("iter", 0)),
               "running_time_sec": round(float(record.get("running_time", float("nan"))), 3),
               "cv_folds": cv_summary.get("folds", 0),
               "cv_time_sec": round(float(cv_summary.get("time", float("nan"))), 3)}
    for name, short, higher_better in METRICS:
        curve = np.asarray(scores[name], dtype=float)
        summary[f"{short}_best"] = float(np.nanmax(curve) if higher_better else np.nanmin(curve))
        summary[f"{short}_best_k"] = int(np.nanargmax(curve) if higher_better else np.nanargmin(curve)) + 1
        summary[f"{short}_auc"] = float(np.trapz(curve, dx=1.0) / max(len(curve) - 1, 1))

    # The papers' aggregate is only well defined when the scan actually reaches
    # 20% of the features: with a smaller --select-ratio the percentage grid gets
    # clipped at the end of the curve and the number is not comparable to their
    # tables, so it is omitted rather than reported misleadingly.
    covers_paper_range = bool(select_ratio) and select_ratio >= max(PAPER_PCTS) / 100.0
    if fold_scores is not None and covers_paper_range:
        for name, short, _ in METRICS:
            rows = np.asarray(fold_scores[name], dtype=float)        # (folds, select_num)
            per_fold = np.array([paper_aggregate(row, select_ratio=select_ratio)
                                 for row in rows])
            per_fold = per_fold[~np.isnan(per_fold)]
            summary[f"{short}_paper_mean"] = (float(per_fold.mean())
                                              if per_fold.size else float("nan"))
            summary[f"{short}_paper_std"] = (float(per_fold.std(ddof=1))
                                             if per_fold.size > 1 else float("nan"))

        # The papers report Coverage normalised by the label count: a raw
        # coverage_error counts "labels needed to cover the true set" and is not
        # comparable across datasets with different label cardinality. Checked
        # against both PDFs -- on all-features MLkNN the raw/|L| ratios (SCENE
        # 0.411, yeast 0.574, MIRFlickr 0.563, 3sources 0.642) land on exactly
        # the published scale (0.419-0.639), while raw counts and count/cardinality
        # do not.
        if summary.get("n_labels"):
            summary["CV_paper_mean_norm"] = summary["CV_paper_mean"] / summary["n_labels"]
            summary["CV_paper_std_norm"] = summary["CV_paper_std"] / summary["n_labels"]
    elif select_ratio:
        pass  # run_one reports this once per run, so sweeps do not spam it

    for key, val in sorted(params.items()):
        summary[f"param_{key}"] = val
    return summary


def save_run(alg, dataset, ranking, view_dims, scores, select_num, record,
             cv_summary, out_dir: Path, fold_scores=None, select_ratio=None,
             view_order=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) feature ranking, 1-based rank with the originating view
    view_of = np.zeros(len(ranking), dtype=int)
    offset = 0
    for vi, dim in enumerate(view_dims):
        view_of[offset:offset + dim] = vi
        offset += dim
    rows = [(r + 1, int(f), int(view_of[f])) for r, f in enumerate(ranking)]
    write_csv(out_dir / f"rankings_{alg}_{dataset}.csv",
              ["rank", "feature_index", "view"], rows)

    # 2) metric curves over the number of selected features (fold-averaged)
    header = ["num_features"] + [short for _, short, _ in METRICS]
    rows = []
    for k in range(select_num):
        rows.append([k + 1] + [float(scores[name][k]) for name, _, _ in METRICS])
    write_csv(out_dir / f"metrics_{alg}_{dataset}.csv", header, rows)

    # 3) the same curves kept per fold -- this is what makes mean +- std (and any
    #    other fold-level statistic) recoverable after the run.
    if fold_scores is not None and len(fold_scores):
        n_folds = len(next(iter(fold_scores.values())))
        pf_header = ["fold", "num_features"] + [short for _, short, _ in METRICS]
        pf_rows = []
        for fi in range(n_folds):
            for k in range(select_num):
                pf_rows.append([fi + 1, k + 1]
                               + [float(fold_scores[name][fi][k])
                                  for name, _, _ in METRICS])
        write_csv(out_dir / f"perfold_{alg}_{dataset}.csv", pf_header, pf_rows)

    # 4) per-dataset summary row for this run
    summary = build_summary(alg, dataset, ranking, view_dims, scores, select_num,
                            record, cv_summary, fold_scores=fold_scores,
                            select_ratio=select_ratio, view_order=view_order)

    # 5) both summary files are rebuilt from the accumulated history.
    #    summary_all.csv is the append-only audit log; summary_<alg>.csv holds one
    #    row per dataset for that algorithm (as README_zh.md describes), so a
    #    re-run replaces its own row instead of leaving the previous dataset's row
    #    behind. The column set grows when fields are added, so an older header is
    #    migrated (new columns blank) rather than blindly appended to, which keeps
    #    the history behind PAPER_COMPARISON.md intact.
    all_path = out_dir / "summary_all.csv"
    new_header = list(summary.keys())
    history, schema_changed = _read_history(all_path, new_header)
    if schema_changed:
        # NOTE: the handle must be closed first -- unlinking an open file raises
        # PermissionError [WinError 32] on Windows (it is legal on POSIX, which is
        # why this has to be spelled out explicitly).
        all_path.unlink()
        print(f"  note: summary_all.csv schema changed; migrated "
              f"{len(history)} earlier row(s), new columns left blank")

    history.append(dict(summary))
    write_csv(all_path, new_header, [[r[c] for c in new_header] for r in history])

    latest = {}
    for row in history:
        latest[(row["algorithm"], row["dataset"])] = row
    alg_rows = [r for (a, _), r in sorted(latest.items()) if a == alg]
    write_csv(out_dir / f"summary_{alg}.csv", new_header,
              [[r[c] for c in new_header] for r in alg_rows])

    return summary


def _read_history(all_path: Path, new_header):
    """Return (rows_as_dicts, schema_changed) for an existing summary_all.csv."""
    if not all_path.exists():
        return [], False
    with all_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        old_header = reader.fieldnames or []
        if old_header == new_header:
            return [dict(row) for row in reader], False
        return [{c: row.get(c, "") for c in new_header} for row in reader], True


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
            out_dir, verbose=True, save=True, view_order=None, checkpoint=True):
    """Run one algorithm on one dataset over a ``folds``-fold cross-validation.

    Returns ``(summary, fold_scores)``. ``fold_scores`` maps metric name to a
    list holding each fold's metric-vs-k curve, which is what allows mean +- std
    to be computed afterwards. With ``save=False`` nothing is written to disk,
    which is what the hyper-parameter sweeps in ``tune.py`` rely on.

    ``view_order`` optionally reorders the view blocks before any algorithm sees
    them (see :func:`permute_views`).
    """
    if dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    filename, discretize = DATASETS[dataset]
    path = DATA_DIR / filename
    if not path.exists():
        raise SystemExit(f"missing dataset file: {path}")

    if folds < 2:
        raise SystemExit(
            "--folds must be >= 2: the fold size is n_test = round(m / folds), so "
            "folds=1 makes the single test split cover the whole dataset and leaves "
            "an empty training set")
    if ratio is not None and abs(float(ratio) - 1.0 / folds) > 1e-9:
        print(f"  note: --ratio {ratio:g} is ignored; the test-set size comes from "
              f"--folds {folds} (n_test = round(m / folds) = 1/{folds} of the data)")

    entry = resolve_algorithm(alg)
    X, view_dims, Y = get_data(path, discretize)
    if view_order is not None:
        # accept the raw CLI string here so the permutation is validated against
        # THIS dataset's view count (it differs from dataset to dataset)
        if isinstance(view_order, str):
            view_order = parse_view_order(view_order, len(view_dims))
        X, view_dims = permute_views(X, view_dims, view_order)
    if verbose:
        print(f"[{alg}] {dataset}: X={X.shape} views={view_dims.tolist()} Y={Y.shape}")
        if view_order is not None:
            print(f"  view blocks reordered to {view_order}; feature indices in "
                  f"rankings_{alg}_{dataset}.csv refer to this layout")
        if select_ratio < max(PAPER_PCTS) / 100.0:
            print(f"  note: --select-ratio {select_ratio:g} does not reach "
                  f"{max(PAPER_PCTS)}% of the features, so the paper-style "
                  f"(1%-{max(PAPER_PCTS)}%) aggregates are omitted for this run")

    m = Y.shape[0]
    n_test = int(round(m / folds))
    select_num = max(1, int(X.shape[1] * select_ratio))

    # ONE shared permutation for the whole cross-validation. Every fold then
    # takes a disjoint slice of it, so the folds are a genuine partition of the
    # data. Computing a fresh permutation per fold (the previous behaviour) made
    # the test sets overlap: with 5 folds only ~65% of the instances were ever
    # held out and pairs of folds shared ~39% of their test sets.
    fold_indices = np.arange(m)
    if shuffle:
        np.random.default_rng(seed).shuffle(fold_indices)

    score_mat = {name: np.zeros(select_num) for name, _, _ in METRICS}
    fold_scores_all = {name: [] for name, _, _ in METRICS}
    ranking = None
    record = {}
    used_folds = 0
    t0 = time.time()

    # Fold-level checkpointing. TOCL on SCENE/MIRFlickr/iaprtc12 costs 20-60 min
    # per fold, so a run that is interrupted (Ctrl-C, a reclaimed background job,
    # a machine reboot) would otherwise throw away hours of work. Each completed
    # fold is cached under a key that covers every knob that can change the
    # result, and a later run replays the cache instead of recomputing.
    cache_dir = None
    cached = {}
    if checkpoint:
        key = run_cache_key(alg, dataset, params, folds, ratio, select_ratio,
                            shuffle, seed, view_order)
        cache_dir = out_dir / ".cache" / key
        cached = load_fold_cache(cache_dir, select_num)
        if cached and verbose:
            print(f"  resume: {len(cached)} cached fold(s) found under "
                  f"{cache_dir.relative_to(out_dir)} -- skipping those")

    for j in range(folds):
        hit = cached.get(j)
        if hit is not None:
            fold_curve, fold_ranking, fold_iter, fold_secs = hit
            if ranking is None:
                ranking = fold_ranking
                record = {"method": alg, "dataset": dataset,
                          "running_time": fold_secs, "selected_num": None,
                          "param": {k: params[k] for k in ("alpha", "beta", "gamma", "lamb")},
                          "iter": fold_iter, "n_samples": m, "n_labels": Y.shape[1]}
            if verbose:
                print(f"  fold {j}/{folds}: cached ({fold_iter} iters)")
            for name in score_mat:
                fold_scores_all[name].append(fold_curve[name])
                score_mat[name] = (score_mat[name] * used_folds + fold_curve[name]) / (used_folds + 1)
            used_folds += 1
            continue

        trainX, Ytrain, testX, Ytest = split_data_fold(
            X, Y, n_test, j, indices=fold_indices)
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

        fold_curve = evaluate_ranking(trainX, Ytrain, testX, Ytest, rec["idx"], select_num)
        # average the metric curves of every fold, each computed from that
        # fold's own ranking (the original averaged curves produced by
        # different rankings while reporting only the last fold's ranking)
        for name in score_mat:
            fold_scores_all[name].append(np.asarray(fold_curve[name], dtype=float))
            score_mat[name] = (score_mat[name] * used_folds + fold_curve[name]) / (used_folds + 1)
        used_folds += 1

        if cache_dir is not None:
            save_fold_cache(cache_dir, j, fold_curve,
                            np.asarray(rec["idx"], dtype=int), n_iter,
                            float(rec["running_time"]))

    if ranking is None:
        raise SystemExit("no fold produced a usable feature ranking")

    cv_summary = {"folds": used_folds, "time": time.time() - t0}
    if save:
        summary = save_run(alg, dataset, ranking, view_dims, score_mat, select_num,
                           record, cv_summary, out_dir, fold_scores=fold_scores_all,
                           select_ratio=select_ratio, view_order=view_order)
    else:
        summary = build_summary(alg, dataset, ranking, view_dims, score_mat, select_num,
                                record, cv_summary, fold_scores=fold_scores_all,
                                select_ratio=select_ratio, view_order=view_order)
    if verbose:
        print(f"  -> best AP={summary['AP_best']:.4f} @k={summary['AP_best_k']}, "
              f"HL={summary['HL_best']:.4f}")
        if "AP_paper_mean" in summary:
            print(f"  -> paper-style (1%-20%, {used_folds} folds): "
                  f"AP={summary['AP_paper_mean']:.4f}+-{summary['AP_paper_std']:.4f}, "
                  f"CV={summary['CV_paper_mean']:.4f}, "
                  f"HL={summary['HL_paper_mean']:.4f}, "
                  f"RL={summary['RL_paper_mean']:.4f}")
    return summary, fold_scores_all


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
    parser.add_argument("--view-order", default=None,
                        help="reorder view blocks before running, e.g. '0,1,2,4,3'. "
                             "The papers' Table 1 lists the 5-view datasets as "
                             "DH,DHV3H1,GIST,HHV3H1,HH while the .mat files hold "
                             "DH,DHV3H1,GIST,HH,HHV3H1, and TOCL is sensitive to "
                             "that order")
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                        help="do not replay cached folds (default: resume from "
                             "results/.cache when a previous run was interrupted)")
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
                    args.shuffle, args.seed, out_dir, view_order=args.view_order,
                    checkpoint=args.resume)
    end = time.localtime()
    print("start time =", time.asctime(start))
    print("  end time =", time.asctime(end))
    return 0


if __name__ == "__main__":
    sys.exit(main())
