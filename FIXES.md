# Repairs applied to the upstream code

> The English `README.md` is the **original upstream file and is left unchanged**.
> Chinese documentation for the repaired project lives in `README_zh.md`; this
> file records what changed and why.

This document records every change made to `Multi-view-Multi-label-learning`,
with the evidence and the reasoning. The original files are preserved in a
timestamped copy under `/tmp/mvml_backup_*` on the machine where the work was
done.

> **Behavioural warning.** Fixes 3–6 change numerical results. Experiments run
> with the buggy upstream code are **not** comparable to results produced after
> this patch set; everything must be re-run.

---

## 1. Windows-only paths removed (blocking on Linux/macOS)

**Symptom.** `main.py` contained

```python
data_addr = ['.\\data\\emotions.mat', ...]
dataname  = (data_addr[i].split('\\')[-1]).split('.')[0]
if f.split('\\')[-1] == 'emotions.mat' or ...:
writeperform("result\\view4\\hamming_loss.xlsx", ...)
```

On POSIX these literal `.\data\...` names do not exist and `split('\\')` never
finds a separator, so the script died before doing any work.

**Fix.** Paths are resolved from the repository root with `pathlib`
(`DATA_DIR`, `RESULT_DIR`), datasets are looked up through the `DATASETS`
registry, and output goes to `results/`. Added a CLI
(`--alg/--data/--folds/--seed/…`) and `--list`.

## 2. Hard-coded single-algorithm dispatch

**Symptom.** `main.py` did `import alg.DHLI` at module level and called
`DHLI.DHLI(...)`. The other five algorithms in `alg/` were unreachable.

**Fix.** `ALGORITHMS` maps each name to `(module, entry point)` and is resolved
at run time with `importlib`; `EXTRA_ARGS` declares the extra positional
arguments some entry points require (`EF2FS` → `V_dim`, `GRAFS` → `kk`).
`--alg all` runs the full set.

## 3. `alg/EF2FS.py` — `NameError`, algorithm could not run at all

**Symptom.** Line 46 of the original:

```python
nu_temp[i] = (1 / np.trace(V.T @ Lx_lst[i] @ V))
```

`Lx_lst` is referenced inside the optimisation loop but is **never assigned
anywhere in the file**. Every run crashed on the first iteration with
`NameError: name 'Lx_lst' is not defined`. `EF2FS` was therefore completely
non-functional in the published snapshot.

**Fix.** The per-view graph Laplacians are now constructed *before* the loop,
using the same construction as `I2VSLC` (`construct_W`, `k=5`, heat kernel,
`L = D - S`) so the two algorithms stay comparable. The dead `pylab` /
`construct_W` imports that hinted at the missing code were replaced by the real
call, and two hand-rolled index-slicing blocks were collapsed into direct
slices without changing the arithmetic.

A follow-on degenerate case surfaced during verification: for a view whose
Laplacian energy `trace(VᵀLV)` evaluates to `0`, the unguarded
`1 / trace(...)` produced `inf`/`NaN` view weights (and a
`RuntimeWarning: divide by zero`). Views with zero energy now receive weight
`0`, and if every view is degenerate the weights fall back to uniform. On the
benchmark splits this is bit-for-bit identical to the unguarded version.

## 4. `alg/DHLI.py` — a regulariser that was silently always zero

**Symptom.** The original loop body:

```python
B = 0
for j in range(n_view):
    if i != j:
        B = B * y[i]        # self-multiplication, and 0 * anything == 0
...
y[i] = np.multiply(y[i], np.true_divide(
    np.dot(x[i], u[i]) + lamb * tem1,
    y[i] + lamb * y[i] + lamb * y[i] * B * B + eps))
```

Two defects in three lines:
1. `B` was initialised to `0` and only ever multiplied, so `B == 0` forever.
2. Even with a correct identity, the loop multiplied by `y[i]` instead of the
   *other* views `y[j]`.

Consequence: the `lamb * y[i] * B * B` term in the denominator — the
view-orthogonality / opposing-consensus regulariser — was **identically zero**.
No exception was raised, so the published numbers were produced without it.

**Fix.**

```python
B = 1
for j in range(n_view):
    if i != j:
        B = B * y[j]
```

This makes `B = Π_{j≠i} y[j]`, which is exactly what the objective function
already assumes: the reported `temp7 = ‖Π_i y[i]‖_F²` term uses the same
all-views product. The update and the objective are now consistent.

## 5. `split_data_fold` — features and labels were desynchronised

**Symptom.**

```python
if shuffle:
    random.shuffle(X)   # X is a numpy array; labels Y are never shuffled
```

With `shuffle=True`, `random.shuffle` on a 2-D numpy array (a) does not shuffle
rows the way one would expect and (b) leaves `Y` in the original order, so
feature rows no longer correspond to their labels. The default
`shuffle=False` masked the bug and meant folding was a *contiguous* block split
rather than a random one.

**Fix.** Shuffling uses one shared permutation of an index array, applied to
`X` and `Y` together, with an explicit `np.random.default_rng(seed + fold)`:

```python
indices = np.arange(m)
if shuffle:
    rng.shuffle(indices)
test_idx  = indices[start:stop]
train_idx = np.setdiff1d(indices, test_idx)
return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
```

## 6. Evaluation averaged incompatible curves

**Symptom.** The metric curves were accumulated across folds with a running
mean (`(old * j + new) / (j + 1)`) but only the **last** fold's feature ranking
was evaluated for every `k`, and only that last ranking was written out. Folds
that produced failing rankings (e.g. a `NaN` objective, which `I2VSLC` can hit
and `break` out of) silently contributed a stale curve.

**Fix.** Each fold now calls `evaluate_ranking` with **its own** ranking, the
curves are averaged over the folds that actually succeeded, and the ranking
that is written to disk is documented as the first successful fold's. Skipped
folds are reported instead of being folded into the average.

## 7. Legacy Excel output replaced with CSV

**Symptom.** `writeperform` / `writeperform_obj` mixed `xlsxwriter` (write),
`xlrd` (read) and `xlutils.copy` (append), with a hard-coded branch per dataset
name. `xlrd >= 2.0` cannot open `.xlsx` at all, so the append path was already
broken; the original `result/view4/*.xlsx` files contain a single smoke-test
row (`emotions` / `view4`).

**Fix.** Four tidy CSV outputs instead of opaque Excel cells:

| File | Content |
|---|---|
| `rankings_<alg>_<dataset>.csv` | `rank, feature_index, view` |
| `metrics_<alg>_<dataset>.csv` | each metric vs. number of selected features |
| `summary_<alg>.csv` | `*_best`, `*_best_k`, `*_auc` per metric + hyper-parameters |
| `summary_all.csv` | every run appended, for cross-algorithm tables |

## 8. Reproducibility

- Each algorithm called `np.random.seed(...)` with a different constant
  (`100`, `2`, `4`, `1`) and `GRAFS` called **no** seeding function at all
  despite using `np.random.rand`. Global `np.random.seed` also leaks into every
  other library in the process.
- **Fix.** A shared `alg/_util.make_rng()` returns an explicit
  `np.random.default_rng`, seeded from an optional argument, else `MVML_SEED`,
  else a per-module default. All `np.random.rand/randint` calls were converted
  to the Generator API.

## 9. Index dtype and sparse-API robustness

- `np.array(list(range(a, b)))` is `int64` only when the bounds are Python
  ints; wherever view dimensions came from float arithmetic it produced a float
  array and indexing `X[:, cols]` raised
  `IndexError: only integers … are valid indices`. A `as_int_indices()` helper
  now normalises every index array, and the feature ranking written to disk is
  explicitly `int64`.
- `construct_W(...).A` relied on the scipy `spmatrix.A` attribute, removed in
  scipy 1.14. Densification goes through `dense()` which tries
  `toarray()`, then `.A`, then a plain cast.
- View dimensions were read as `x_view[i][0]`, which requires the MATLAB
  `(n_views, 1)` layout. `view_dim()` accepts a scalar, a 1-element array or a
  list, so both conventions work.

## 10. `scikit-multilearn` / modern scikit-learn incompatibility

**Symptom.** `MLkNN.fit` crashed with

```
TypeError: NearestNeighbors.__init__() takes 1 positional argument but 2 were given
```

`scikit-multilearn` 0.2.0 (last release 2018) constructs
`NearestNeighbors(self.k)` positionally, which scikit-learn ≥ 1.0 rejects; it
then also indexes sparse matrices with `numpy.float64` neighbour counts, which
scipy ≥ 1.8 rejects with
`ValueError: Inexact indices into sparse matrices are not allowed`.

**Fix.** `main.py` builds the classifier through `_make_mlknn()`, a subclass
that reimplements `_compute_cond` (keyword construction + `int()` indices) and
`predict` / `predict_proba` (int indices). The probability formulas are copied
verbatim from the library — only the incompatible calls are corrected.

## 11. Dead imports that would fail at import time

- `alg/EF2FS.py` and `alg/GRAFS.py` imported `pylab as pl`; `alg/GRAFS.py`
  imported `matplotlib.pyplot`; `alg/GRAFS.py` and `alg/EF2FS.py` imported
  `scipy.spatial.distance.pdist` without using it (well, `UGRFS` uses `pdist`).
- `alg/I2VSLC.py`, `alg/EF2FS.py`, `alg/GRAFS.py` imported `construct_W` but
  never called it (in `EF2FS` that absence *was* the bug — see fix 3).
- None of `pylab`, `matplotlib` were declared dependencies, so importing these
  modules could fail on a clean environment.

**Fix.** Unused imports removed, and every remaining third-party import is now
declared in `setup_env.sh`.

---

## Environment notes (host-specific)

Three problems on this machine were **not** code defects but had to be solved to
run anything. All of them are handled by `setup_env.sh` and `run.sh`.

1. **`/mnt/Data4` is mounted `noexec`.** A Python environment placed inside the
   repository cannot load its own shared objects:
   `ImportError: … _multiarray_umath…so: failed to map segment from shared
   object`. This is why the project uses a **conda environment** — conda creates
   it under the conda envs directory (here `~/miniconda3/envs/<name>`) on the
   executable home filesystem, so the problem does not arise. The environment is
   named `mvml` by default (`CONDA_ENV` overrides). For the same reason a script
   on that mount cannot be executed directly — invoke it as `bash run.sh …`
   rather than `./run.sh`.
2. **`pip` could not install from PyPI** (`pypi.org` unreachable, the Tsinghua
   mirror works) and neither `conda activate` nor `source …/activate` reliably
   takes effect in a non-interactive shell, so `pip` resolved to the system
   interpreter and tried to write into `/usr/local/lib` (permission denied).
   Therefore `run.sh` calls the environment interpreter directly and
   `setup_env.sh` installs with `$PY -m pip`, never relying on activation.
3. **A `~/.local` package tree shadows the environment.** This machine has
   numpy / scipy / scikit-learn / scikit-multilearn / skfeature under
   `~/.local/lib/python3.10/site-packages`. Two consequences, both handled in
   `setup_env.sh`:
   - `pip` inspects `~/.local` when deciding whether a requirement is already
     satisfied, so it *skipped* packages missing from the environment and left
     only a `.dist-info` behind. `import joblib` and `import skmultilearn` then
     failed. Installing with `--ignore-installed` forces every file into the
     environment.
   - At runtime `run.sh` sets `PYTHONNOUSERSITE=1` so the older `~/.local` tree
     cannot shadow the pinned versions. `setup_env.sh` verifies under the same
     flag and asserts that numpy/scipy/sklearn/pandas resolve inside the
     environment prefix.

   Related constraint: **Python 3.13 is not usable** — `scikit-multilearn` 0.2.0
   imports the removed `pkg_resources`, so the environment is pinned to
   Python 3.10.

Dependency versions are pinned to a self-consistent set
(`numpy==1.26.4`, `scipy==1.11.4`, `scikit-learn==1.3.2`,
`pandas==2.0.3`, `scikit-multilearn==0.2.0`, `skfeature-chappers==1.2.1`).
numpy 2.x was tested and rejected: `scikit-multilearn` 0.2.0 and
`skfeature-chappers` predate it.
