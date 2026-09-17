"""Shared helpers for the algorithm implementations in this package."""

from __future__ import annotations

import os

import numpy as np

#: fixed default seed, matching the historical hard-coded value
DEFAULT_SEED = 100


def make_rng(seed=None, default=DEFAULT_SEED) -> np.random.Generator:
    """Return a fresh Generator.

    Priority: explicit ``seed`` argument > ``MVML_SEED`` environment variable >
    ``default``. Using an explicit Generator keeps ``numpy.random.seed()`` from
    leaking into other libraries and makes runs reproducible via ``MVML_SEED``.
    """
    if seed is None:
        env = os.environ.get("MVML_SEED")
        if env is not None:
            try:
                seed = int(env)
            except ValueError:
                seed = default
        else:
            seed = default
    return np.random.default_rng(int(seed))


def view_dim(x_view, i: int) -> int:
    """Return the number of features in view ``i``.

    ``x_view`` may be a flat sequence of ints, an ``(n_views, 1)`` array (as
    produced by the original MATLAB loader) or any mix of those — indexing it
    and casting keeps every calling convention working.
    """
    value = x_view[i]
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        value = value[0]
    return int(value)


def max_iter(default: int) -> int:
    """Iteration cap for an algorithm, overridable via ``MVML_MAX_ITER``.

    The published implementations hard-code a 500-iteration ceiling that is
    never reached on the benchmark splits but makes a full grid run take hours.
    Setting ``MVML_MAX_ITER`` makes a quick sweep or a smoke test affordable
    without editing any algorithm.
    """
    raw = os.environ.get("MVML_MAX_ITER")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return int(default)


def as_int_indices(values) -> np.ndarray:
    """Coerce a sequence of feature indices to int64.

    ``np.array(list(range(a, b)))`` is an int array on Linux but a float array
    when the bounds come from float arithmetic; float indices raise
    ``IndexError: only integers ... are valid indices``. This normalises both.
    """
    return np.asarray(values).astype(np.int64)


def dense(matrix):
    """Densify a scipy sparse matrix across scipy/skfeature API changes."""
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    if hasattr(matrix, "A"):  # removed in scipy >= 1.14
        return np.asarray(matrix.A, dtype=float)
    return np.asarray(matrix, dtype=float)
