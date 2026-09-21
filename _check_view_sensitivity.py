"""Does the view order actually change a result?

The six papers list Corel5K / IAPRTC12 as
[DH, DHV3H1, GIST, HHV3H1, HH] while data/*.mat holds
[DH, DHV3H1, GIST, HH, HHV3H1] -- positions 4 and 5 are swapped.

Both algorithms slice their per-view matrices by position, so the question is
whether that positional order is observable in the output.

Test: run the algorithm on a dataset, then on the same dataset with its view
blocks permuted, and compare the two feature rankings after undoing the
permutation. Blocks of equal width are used (3sources = three 1000-dim views) so
the random initialisation draws exactly the same shapes in both runs.
"""
import os

os.environ.setdefault("MVML_MAX_ITER", "20")

import numpy as np

import main as m
from alg._util import as_int_indices

PERM = [2, 0, 1]          # new block order = old blocks [2, 0, 1]
DS = "3sources"
SEED = 100


def split_blocks(X, dims):
    out, off = [], 0
    for d in dims:
        out.append(X[:, off:off + d])
        off += d
    return out


def feature_permutation(dims, perm):
    """Map original feature index -> index in the permuted layout.

    ``perm[k]`` names the ORIGINAL block that ends up in new position ``k``, so
    the source offset is ``sum(dims[:perm[k]])`` -- not a running counter (that
    was a bug in the first version of this test and invalidated its output).
    """
    offsets = [int(sum(dims[:k])) for k in range(len(dims))]
    mapping = {}
    off_new = 0
    for k in perm:
        d = dims[k]
        for j in range(d):
            mapping[offsets[k] + j] = off_new + j
        off_new += d
    assert len(mapping) == sum(dims)
    return np.array([mapping[i] for i in range(sum(dims))])


def run(alg, X, view_dims, Y):
    entry = m.resolve_algorithm(alg)
    rec, n_iter = entry(X, np.asarray(view_dims), Y, DS,
                        1.0, 1.0, 1.0, 1.0)
    return as_int_indices(rec["idx"]), n_iter


def main():
    filename, disc = m.DATASETS[DS]
    X, view_dims, Y = m.get_data(m.DATA_DIR / filename, disc)
    dims = [int(d) for d in view_dims]
    print(f"{DS}: dims={dims}  permutation={PERM}")

    blocks = split_blocks(X, dims)
    X_perm = np.hstack([blocks[k] for k in PERM])
    assert X_perm.shape == X.shape

    fmap = feature_permutation(dims, PERM)      # old index -> new index
    inv = np.empty_like(fmap)
    inv[fmap] = np.arange(len(fmap))            # new index -> old index

    for alg in ("UGRFS", "TOCL"):
        rank_a, it_a = run(alg, X, view_dims, Y)
        rank_b, it_b = run(alg, X_perm, view_dims, Y)

        # map the permuted ranking back to the original feature numbering
        rank_b_back = inv[rank_b]

        same_top10 = list(rank_a[:10]) == list(rank_b_back[:10])
        # rank correlation between the two weight orderings
        pos_a = np.empty(len(rank_a), dtype=int)
        pos_a[rank_a] = np.arange(len(rank_a))
        pos_b = np.empty(len(rank_b_back), dtype=int)
        pos_b[rank_b_back] = np.arange(len(rank_b_back))
        rho = np.corrcoef(pos_a, pos_b)[0, 1]

        print(f"\n[{alg}] iters {it_a} vs {it_b}")
        print(f"  ranking identical (after undoing the permutation): "
              f"{list(rank_a) == list(rank_b_back)}")
        print(f"  top-10 identical : {same_top10}")
        print(f"  top-10 original  : {list(rank_a[:10])}")
        print(f"  top-10 permuted  : {list(rank_b_back[:10])}")
        print(f"  rank correlation : {rho:.6f}")


if __name__ == "__main__":
    main()
