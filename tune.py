"""Single-parameter hyper-parameter sweep for the algorithms in ``alg/``.

Both reference papers tune their balance parameters over
``{1e-3, 1e-2, ..., 1e3}`` and report the best setting per dataset. A full grid
over four parameters is ``7**4 = 2401`` configurations, which costs far more
than the 5-fold evaluation itself. What the papers actually show in their
parameter-analysis figures -- and what this script implements -- is an
*individual* sweep: vary one parameter over the grid while holding the others
fixed, take the best, then move on to the next parameter. ``--rounds`` repeats
the whole cycle, which turns it into coordinate descent.

Every sweep point is a complete cross-validation, so a full round costs
``len(params) * len(values)`` cross-validations (28 with the defaults). Keep that
in mind for TOCL, whose per-fold cost dominates everything else: cap
``--max-iter``, sweep the cheapest dataset, or shrink the grid with ``--values``.
Sweeps call ``main.run_one(save=False)`` and therefore never touch
``results/summary_*.csv``, ``rankings_*.csv`` or ``metrics_*.csv``.

The value reported for each configuration is the paper-style aggregate
(mean over the 1%-20% feature-percentage grid, see ``main.paper_aggregate``),
not the best point of the curve. Use ``--aggregate`` to change that.

Examples::

    # one round, all four parameters, 7 values each  (28 CV runs x 5 folds)
    python tune.py --alg UGRFS --data SCENE

    # just one parameter, custom grid
    python tune.py --alg UGRFS --data SCENE --params alpha --values 0.001 0.1 10

    # TOCL is expensive: cap the iterations and sweep the cheapest dataset
    MVML_MAX_ITER=60 python tune.py --alg TOCL --data 3sources
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import main as m

BALANCE_PARAMS = ("alpha", "beta", "gamma", "lamb")
DEFAULT_VALUES = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)


def metric_direction(short: str) -> bool:
    """Return True when larger is better for ``short`` (e.g. ``AP``)."""
    for _, s, higher_better in m.METRICS:
        if s == short:
            return higher_better
    raise SystemExit(f"unknown metric {short!r}; choose from "
                     f"{[s for _, s, _ in m.METRICS]}")


def read_metric(summary: dict, short: str, aggregate: str) -> float:
    if short not in m.PAPER_METRICS:
        print(f"  note: {short} is not one of the metrics the papers report "
              f"({', '.join(m.PAPER_METRICS)}); it is a code-only extra")
    if aggregate == "paper":
        return float(summary[f"{short}_paper_mean"])
    if aggregate == "best":
        return float(summary[f"{short}_best"])
    return float(summary[f"{short}_auc"])


def coerce(value, template):
    """Keep an integer-typed parameter (``V_dim``, ``kk``) integral."""
    if isinstance(template, int) and not isinstance(template, bool):
        return int(round(value))
    return float(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Single-parameter hyper-parameter sweep")
    parser.add_argument("--alg", required=True, help="algorithm name")
    parser.add_argument("--data", required=True, help="dataset name")
    parser.add_argument("--params", nargs="+", default=list(BALANCE_PARAMS),
                        help=f"parameters to sweep in this order (default: "
                             f"{' '.join(BALANCE_PARAMS)})")
    parser.add_argument("--values", nargs="+", type=float, default=list(DEFAULT_VALUES),
                        help="grid for every swept parameter "
                             f"(default: {' '.join(str(v) for v in DEFAULT_VALUES)})")
    parser.add_argument("--rounds", type=int, default=1,
                        help="repeat the whole parameter cycle (coordinate descent)")
    parser.add_argument("--metric", default="AP",
                        help="metric short name to optimise (default: AP)")
    parser.add_argument("--aggregate", choices=("paper", "best", "auc"), default="paper",
                        help="how to collapse the metric curve (default: paper = the "
                             "1%%-20%% mean the papers tabulate)")
    parser.add_argument("--folds", type=int, default=5,
                        help="CV folds per sweep point (default: 5, the papers' "
                             "protocol; every extra fold multiplies the sweep cost)")
    parser.add_argument("--select-ratio", type=float, default=0.2)
    parser.add_argument("--ratio", type=float, default=0.2)
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="shuffle before folding (default: on)")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="sets MVML_MAX_ITER for the sweep")
    parser.add_argument("--baseline", nargs="*", default=None,
                        help="start from KEY=VALUE pairs instead of DEFAULT_PARAMS, "
                             "e.g. alpha=1 beta=0.1")
    parser.add_argument("--out", default=str(m.RESULT_DIR / "tuning"))
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit")
    args = parser.parse_args(argv)

    if args.alg not in m.ALGORITHMS:
        raise SystemExit(f"unknown algorithm {args.alg!r}; choose from {sorted(m.ALGORITHMS)}")
    if args.data not in m.DATASETS:
        raise SystemExit(f"unknown dataset {args.data!r}; choose from {sorted(m.DATASETS)}")
    if args.folds < 2:
        raise SystemExit("--folds must be >= 2 (see main.run_one)")

    if args.max_iter:
        os.environ["MVML_MAX_ITER"] = str(args.max_iter)

    if args.aggregate == "paper" and args.select_ratio < max(m.PAPER_PCTS) / 100.0:
        raise SystemExit(
            f"--aggregate paper needs --select-ratio >= {max(m.PAPER_PCTS) / 100.0:g} "
            f"so the curve actually covers the papers' 1%-{max(m.PAPER_PCTS)}% grid; "
            f"got {args.select_ratio:g}. Use --select-ratio 0.2, or pick "
            f"--aggregate best / auc for a cheaper sweep.")

    baseline = dict(m.DEFAULT_PARAMS[args.alg])
    if args.baseline:
        for item in args.baseline:
            key, _, val = item.partition("=")
            if not _:
                raise SystemExit(f"--baseline expects KEY=VALUE, got {item!r}")
            baseline[key] = coerce(float(val), baseline.get(key, 0.0))

    current = dict(baseline)
    higher_better = metric_direction(args.metric)
    sign = 1.0 if higher_better else -1.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"tuning_{args.alg}_{args.data}.csv"
    rows = []

    n_points = args.rounds * len(args.params) * len(args.values)
    print(f"[tune] {args.alg} on {args.data}")
    print(f"[tune] optimising {args.metric} "
          f"({'higher' if higher_better else 'lower'} is better), "
          f"aggregate={args.aggregate}, folds={args.folds}")
    print(f"[tune] MVML_MAX_ITER={os.environ.get('MVML_MAX_ITER', '(algorithm default)')}, "
          f"seed={args.seed}, shuffle={args.shuffle}")
    print(f"[tune] baseline : {current}")
    print(f"[tune] plan     : {args.rounds} round(s) x {len(args.params)} param(s) x "
          f"{len(args.values)} value(s) = {n_points} cross-validation run(s)")
    print(f"[tune] output   : {csv_path}")
    if args.dry_run:
        return 0
    print()

    t_start = time.time()
    for rnd in range(args.rounds):
        for param in args.params:
            if param not in current:
                raise SystemExit(f"parameter {param!r} does not exist for {args.alg}; "
                                 f"available: {sorted(current)}")
            template = current[param]
            best_val, best_score = template, None
            print(f"--- round {rnd + 1}/{args.rounds}  param {param} "
                  f"(others fixed at {_others(current, param)}) ---")
            for value in args.values:
                v = coerce(value, template)
                trial = dict(current)
                trial[param] = v
                t0 = time.time()
                try:
                    summary, _ = m.run_one(args.alg, args.data, trial, args.folds,
                                           args.ratio, args.select_ratio, args.shuffle,
                                           args.seed, out_dir, verbose=False, save=False)
                except SystemExit as exc:
                    print(f"  {param}={v:<10g} FAILED: {exc}")
                    continue
                except Exception as exc:                       # noqa: BLE001
                    print(f"  {param}={v:<10g} FAILED: {type(exc).__name__}: {exc}")
                    continue
                score = read_metric(summary, args.metric, args.aggregate)
                dt = time.time() - t0
                rows.append({"round": rnd + 1, "param": param, "value": v,
                             "metric": args.metric, "aggregate": args.aggregate,
                             "score": score, "folds": args.folds,
                             "iterations": summary.get("iterations", ""),
                             "seconds": round(dt, 2),
                             "config": _fmt(trial)})
                mark = ""
                if best_score is None or sign * score > sign * best_score:
                    best_score, best_val, mark = score, v, "  <- best"
                print(f"  {param}={v:<10g} {args.metric}={score:.4f}  "
                      f"({dt:.1f}s){mark}")
            if best_score is None:
                print(f"  every value failed for {param}; leaving it at {template}")
                continue
            print(f"  => {param}: {template} -> {best_val}  ({args.metric}={best_score:.4f})\n")
            current[param] = best_val

    _write_csv(csv_path, rows)

    print("=" * 72)
    print(f"[tune] done in {time.time() - t_start:.1f}s")
    print(f"[tune] baseline : {baseline}")
    print(f"[tune] tuned    : {current}")
    if rows:
        best = max(rows, key=lambda r: sign * r["score"])
        print(f"[tune] best {args.metric} ({args.aggregate}) = {best['score']:.4f}")
        print(f"[tune] best row : {best['config']}")
        print()
        print("[tune] paste into main.py DEFAULT_PARAMS if you want to fix it:")
        print(f'    "{args.alg}": {_py_dict(current)},')
    else:
        print("[tune] no successful configuration")
    return 0


def _others(cfg: dict, param: str) -> dict:
    return {k: v for k, v in cfg.items() if k != param}


def _fmt(cfg: dict) -> str:
    return " ".join(f"{k}={v:g}" for k, v in sorted(cfg.items()))


def _py_dict(cfg: dict) -> str:
    return "{" + ", ".join(f'"{k}": {v!r}' for k, v in cfg.items()) + "}"


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
