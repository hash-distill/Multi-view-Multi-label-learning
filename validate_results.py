"""Check that ``results/`` covers every dataset the six papers need, completely.

For each (algorithm, dataset) cell of the six papers' result tables this verifies

* a row exists in ``summary_all.csv`` with the paper-style 1%-20% aggregates,
* ``metrics_<ALG>_<DS>.csv`` has exactly ``select_num`` rows (one per k),
* ``perfold_<ALG>_<DS>.csv`` has exactly ``folds x select_num`` rows,
* the run really used 5 folds and reached 20% of the features.

    python validate_results.py              # report
    python validate_results.py --write      # also write results/VALIDATION.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import main as m
import paper_table as pt

FOLDS = 5
SELECT_RATIO = 0.2


def count_rows(path: Path) -> int:
    if not path.exists():
        return -1
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1          # minus the header


def check_cell(results: Path, row, alg: str, ds: str) -> list[str]:
    problems = []
    if row is None:
        return ["缺少 summary 行（未跑）"]

    for key in ("AP_paper_mean", "CV_paper_mean_norm", "HL_paper_mean", "RL_paper_mean"):
        if row.get(key, "") in ("", "nan"):
            problems.append(f"{key} 为空（未覆盖满 1%–20%）")

    if str(row.get("cv_folds", "")) != str(FOLDS):
        problems.append(f"cv_folds={row.get('cv_folds')!r} != {FOLDS}")

    try:
        select_num = int(float(row.get("select_num", 0)))
        n_features = int(float(row.get("n_features", 0)))
    except ValueError:
        problems.append("select_num/n_features 无法解析")
        return problems

    expected = max(1, int(n_features * SELECT_RATIO))
    if select_num != expected:
        problems.append(f"select_num={select_num} != int({n_features}*0.2)={expected}")

    metrics = count_rows(results / f"metrics_{alg}_{ds}.csv")
    if metrics != select_num:
        problems.append(f"metrics 行数={metrics} != {select_num}")

    perfold = count_rows(results / f"perfold_{alg}_{ds}.csv")
    if perfold != FOLDS * select_num:
        problems.append(f"perfold 行数={perfold} != {FOLDS}x{select_num}={FOLDS * select_num}")

    if not (results / f"rankings_{alg}_{ds}.csv").exists():
        problems.append("rankings 文件缺失")

    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Completeness check for results/")
    ap.add_argument("--results", default=str(m.RESULT_DIR))
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    results = Path(args.results)
    summary = pt.load_summary(results / "summary_all.csv")

    lines = ["# 复现结果完整性校验", ""]
    lines.append(f"口径：5 折 × `--select-ratio {SELECT_RATIO}`（1%–20% 特征）× `--shuffle`；"
                 "每格要求 summary 行 + metrics/perfold/rankings 三个文件齐全。")
    lines.append("")

    total = complete = 0
    all_problems: list[tuple[str, str, list[str]]] = []
    for alg in sorted(pt.PAPER_DATASETS):
        datasets = pt.PAPER_DATASETS[alg]
        done_here = 0
        for ds in datasets:
            total += 1
            problems = check_cell(results, summary.get((alg, ds)), alg, ds)
            if not problems:
                complete += 1
                done_here += 1
            else:
                all_problems.append((alg, ds, problems))
        lines.append(f"- **{alg}**：{done_here}/{len(datasets)} 完整"
                     + (f"（缺 {', '.join(d for a, d, _ in all_problems if a == alg)}）"
                        if done_here < len(datasets) else ""))

    lines.append("")
    lines.append(f"**合计：{complete}/{total} 个「论文 × 数据集」格子完整。**")
    lines.append("")

    if all_problems:
        lines.append("## 未完成 / 有问题的格子")
        lines.append("")
        lines.append("| 算法 | 数据集 | 问题 |")
        lines.append("|---|---|---|")
        for alg, ds, problems in all_problems:
            lines.append(f"| {alg} | {ds} | {'；'.join(problems)} |")
        lines.append("")

    extra = sorted((a, ds) for (a, ds) in summary
                   if a in pt.PAPER_DATASETS and ds not in pt.PAPER_DATASETS[a])
    if extra:
        lines.append("## 跑了但不属于该论文的数据集（不计入覆盖面）")
        lines.append("")
        for alg, ds in extra:
            lines.append(f"- {alg} / {ds}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.write:
        (results / "VALIDATION.md").write_text(text, encoding="utf-8")
        print(f"[validate] wrote {results / 'VALIDATION.md'}")
    return 0 if complete == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
