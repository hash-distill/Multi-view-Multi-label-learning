"""Turn ``results/`` into a paper-comparable table.

Reads ``results/summary_all.csv`` (which now carries the ``*_paper_mean`` /
``*_paper_std`` columns produced by ``main.build_summary``) and prints a
markdown table next to the numbers the reference papers tabulate.

    python paper_table.py                      # every algorithm present
    python paper_table.py --alg TOCL UGRFS     # only these
    python paper_table.py --write               # also write results/PAPER_COMPARISON.md

The reference values are transcribed from the two PDFs in ``pre-pdf/``:

* TOCL -- ACM MM 2025, Table 2 (AP, Coverage) and Table 3 (HL, RL)
* UGRFS -- AAAI 2025, Table 2 (AP, Coverage) and Table 3 (HL, RL)

Both papers evaluate "feature percentages ranging from 1% to 20%" under 5-fold
cross-validation and report ``mean +- std``. The ``std`` here is taken **across
folds**, matching that convention; whether the papers additionally average over
the 20 percentage points is inferred (neither PDF states its aggregation rule).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import main as m

#: metric -> (is_higher_better, display name, mean column, std column)
#: Coverage uses the label-count-normalised columns, because that is the scale
#: the papers report (see the comment in ``main.build_summary``).
METRIC_META = {
    "AP": (True, "AP", "AP_paper_mean", "AP_paper_std"),
    "CV": (False, "Coverage", "CV_paper_mean_norm", "CV_paper_std_norm"),
    "HL": (False, "Hamming Loss", "HL_paper_mean", "HL_paper_std"),
    "RL": (False, "Ranking Loss", "RL_paper_mean", "RL_paper_std"),
}

#: paper-reported values for the proposed method itself, by dataset key.
#: Source: the two PDFs in pre-pdf/ (Table 2 / Table 3).
PAPER_TARGETS = {
    "TOCL": {
        "SCENE":      {"AP": 0.7981, "CV": 0.4193, "HL": 0.09688, "RL": 0.0914},
        "OBJECT":     {"AP": 0.4959, "CV": 0.2893, "HL": 0.05600, "RL": 0.1607},
        "MIRFlickr":  {"AP": 0.6908, "CV": 0.5715, "HL": 0.17056, "RL": 0.1537},
        "corel5k_5":  {"AP": 0.2419, "CV": 0.4710, "HL": 0.01379, "RL": 0.2198},
        "iaprtc12":   {"AP": 0.1510, "CV": 0.5072, "HL": 0.01543, "RL": 0.1927},
        "3sources":   {"AP": 0.4883, "CV": 0.6017, "HL": 0.23114, "RL": 0.4488},
    },
    "UGRFS": {
        "yeast":      {"AP": 0.6725, "CV": 0.6239, "HL": 0.2257,  "RL": 0.2480},
        "SCENE":      {"AP": 0.8010, "CV": 0.4214, "HL": 0.0978,  "RL": 0.0942},
        "VOC07":      {"AP": 0.5871, "CV": 0.4425, "HL": 0.0847,  "RL": 0.2111},
        "MIRFlickr":  {"AP": 0.6770, "CV": 0.5885, "HL": 0.1775,  "RL": 0.1629},
        "iaprtc12":   {"AP": 0.1474, "CV": 0.4996, "HL": 0.01496, "RL": 0.2020},
        "3sources":   {"AP": 0.4728, "CV": 0.5301, "HL": 0.2097,  "RL": 0.4137},
    },
}

#: which datasets each paper used, in the paper's own row order
PAPER_DATASETS = {
    "TOCL": ["SCENE", "OBJECT", "MIRFlickr", "corel5k_5", "iaprtc12", "3sources"],
    "UGRFS": ["yeast", "SCENE", "VOC07", "MIRFlickr", "iaprtc12", "3sources"],
}


def load_summary(path: Path):
    """Return {(alg, dataset): row}, newest entry winning per (alg, dataset).

    Reads ``summary_all.csv`` and, when present, also merges
    ``summary_all.bak.csv``. That backup appears when a run written by an older
    revision of ``main.py`` meets a schema change and parks the accumulated
    history instead of migrating it in place; merging keeps those rows visible so
    the comparison never silently loses datasets. The live file takes precedence.

    Re-running a dataset appends a new row, so the newest entry wins.
    """
    if not path.exists():
        raise SystemExit(f"{path} not found -- run some experiments first "
                         f"(e.g. python main.py --alg UGRFS --data 3sources)")
    latest = {}
    for candidate in (path.with_name("summary_all.bak.csv"), path):
        if not candidate.exists():
            continue
        with candidate.open("r", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("algorithm") and row.get("dataset"):
                    latest[(row["algorithm"], row["dataset"])] = row
    return latest


def _fmt_cell(row, short, target=None):
    """Render one metric cell, appending the paper's value and the delta."""
    higher_better, _, key_mean, key_std = METRIC_META[short]
    if key_mean not in row or row[key_mean] in ("", "nan"):
        return "-"
    mean = float(row[key_mean])
    std = float(row.get(key_std) or "nan")
    cell = f"{mean:.4f}"
    if std == std:                                            # not NaN
        cell += f"±{std:.4f}"
    if target is not None:
        delta = mean - target
        better = (delta > 0) if higher_better else (delta < 0)
        arrow = "▲" if better else "▼"
        cell += f"<br><sub>paper {target:.4f} {arrow}{abs(delta):.4f}</sub>"
    return cell


def build_table(summary, algs, include_targets=True):
    lines = []
    for alg in algs:
        targets = PAPER_TARGETS.get(alg, {}) if include_targets else {}
        order = PAPER_DATASETS.get(alg) or sorted(
            ds for (a, ds) in summary if a == alg)
        present = [ds for ds in order if (alg, ds) in summary]
        if not present:
            lines.append(f"### {alg}\n\n_尚未跑出结果。_\n")
            continue
        lines.append(f"### {alg}\n")
        lines.append("| 数据集 | " + " | ".join(
            f"{METRIC_META[s][1]} {'↑' if METRIC_META[s][0] else '↓'}"
            for s in METRIC_META) + " | 折数 | 迭代 | 选特征数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for ds in present:
            row = summary[(alg, ds)]
            cells = [_fmt_cell(row, s, targets.get(ds, {}).get(s))
                     for s in METRIC_META]
            lines.append("| " + " | ".join(
                [ds] + cells
                + [row.get("cv_folds", "?"), row.get("iterations", "?"),
                   f"{row.get('select_num', '?')}/{row.get('n_features', '?')}"]) + " |")
        lines.append("")
        missing = [ds for ds in order if (alg, ds) not in summary]
        if missing:
            lines.append(f"_未跑：{', '.join(missing)}_\n")
        # datasets run for this algorithm that are NOT in the paper's Table 1
        # (e.g. TOCL on VOC07): surface them instead of dropping them silently
        extra = sorted(ds for (a, ds) in summary if a == alg and ds not in order)
        if extra:
            lines.append(f"_跑了但不属于该论文的数据集（不计入对比）："
                         f"{', '.join(extra)}_\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Paper-comparison table from results/")
    parser.add_argument("--alg", nargs="+", default=None,
                        help="algorithms to include (default: all in summary_all.csv)")
    parser.add_argument("--results", default=str(m.RESULT_DIR))
    parser.add_argument("--no-targets", dest="targets", action="store_false",
                        default=True, help="omit the paper reference values")
    parser.add_argument("--write", action="store_true",
                        help="also write results/PAPER_COMPARISON.md")
    args = parser.parse_args(argv)

    results_dir = Path(args.results)
    summary = load_summary(results_dir / "summary_all.csv")

    if args.alg:
        algs = [a for a in args.alg]
    else:
        algs = sorted({a for (a, _) in summary})
    if not algs:
        raise SystemExit("summary_all.csv is empty")

    header = ["# 复现结果 vs 论文 Table 2/3", ""]
    header.append("数值为 **1%–20% 特征百分比区间的聚合值**，(std) 为 **5 折间**标准差。"
                  "`paper ...` 后缀是论文自己报告的值与偏差方向。")
    header.append("")
    header.append("> 未做超参数调优时（`DEFAULT_PARAMS` 全为 1.0）数值与论文不可直接比较；"
                  "调优请用 `python tune.py --alg <ALG> --data <DS>`。")
    header.append("")
    body = build_table(summary, algs, include_targets=args.targets)
    text = "\n".join(header) + "\n" + body

    print(text)
    if args.write:
        out = results_dir / "PAPER_COMPARISON.md"
        out.write_text(text, encoding="utf-8")
        print(f"\n[paper_table] wrote {out}", file=__import__("sys").stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
