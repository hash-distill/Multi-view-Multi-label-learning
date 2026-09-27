"""Turn ``results/`` into a paper-comparable table.

Reads ``results/summary_all.csv`` (which now carries the ``*_paper_mean`` /
``*_paper_std`` columns produced by ``main.build_summary``) and prints a
markdown table next to the numbers the reference papers tabulate.

    python paper_table.py                      # every algorithm present
    python paper_table.py --alg TOCL UGRFS     # only these
    python paper_table.py --write               # also write results/PAPER_COMPARISON.md

The reference values are transcribed from the six PDFs in ``pre-pdf/``. The
plain ``pdftotext`` dumps in ``_pdftext/`` scramble these tables, so every value
below was re-read from the PDFs with PyMuPDF's table/coordinate extraction
and cross-checked against the shared baseline
columns the papers reprint from each other (e.g. TOCL, DHLI, EF2FS and I2VSLC
all print M2LD/MSFS/MoRE/MRDM/CLML identically).

* TOCL   -- ACM MM 2025,    Table 2 (AP, Coverage), Table 3 (HL, RL)
* UGRFS  -- AAAI 2025,      Table 2 (AP, Coverage), Table 3 (HL, RL)
* EF2FS  -- Pattern Recognition 157 (2025) 110888, Table 3 (AP, Coverage), Table 4 (HL, RL)
* DHLI   -- AAAI 2024,      Table 2 (AP, Coverage), Table 3 (HL, RL)
* I2VSLC -- Information Sciences 681 (2024) 121215, Table 3 (AP, Coverage), Table 4 (HL, RL)
* GRAFS  -- Information Sciences 679 (2024) 121124, Table 3 (AP, Coverage), Table 4 (HL, RL)

All six evaluate "feature percentages from 1% to 20%" under 5-fold
cross-validation and report ``mean +- std``. The ``std`` here is taken **across
folds**, matching that convention; whether the papers additionally average over
the 20 percentage points is inferred (only GRAFS states its aggregation rule:
"The mean value for each metric is calculated by all sample points").
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
#: Source: the six PDFs in pre-pdf/ (see the module docstring for the tables).
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
    "EF2FS": {
        "emotions":   {"AP": 0.6911, "CV": 0.5707, "HL": 0.2415,  "RL": 0.2732},
        "yeast":      {"AP": 0.6639, "CV": 0.6266, "HL": 0.2225,  "RL": 0.2547},
        "SCENE":      {"AP": 0.7935, "CV": 0.4244, "HL": 0.0978,  "RL": 0.0947},
        "MIRFlickr":  {"AP": 0.6759, "CV": 0.5774, "HL": 0.1744,  "RL": 0.1591},
        "iaprtc12":   {"AP": 0.1509, "CV": 0.4794, "HL": 0.01549, "RL": 0.1922},
        "3sources":   {"AP": 0.4608, "CV": 0.6251, "HL": 0.2332,  "RL": 0.4862},
    },
    "DHLI": {
        "SCENE":      {"AP": 0.7951, "CV": 0.4174, "HL": 0.0969,  "RL": 0.0926},
        "OBJECT":     {"AP": 0.4845, "CV": 0.2979, "HL": 0.0572,  "RL": 0.1642},
        "MIRFlickr":  {"AP": 0.6635, "CV": 0.5897, "HL": 0.1808,  "RL": 0.1671},
        "corel5k_5":  {"AP": 0.2370, "CV": 0.4832, "HL": 0.01379, "RL": 0.2283},
        "iaprtc12":   {"AP": 0.1421, "CV": 0.5091, "HL": 0.01549, "RL": 0.2068},
        "3sources":   {"AP": 0.4534, "CV": 0.6181, "HL": 0.2315,  "RL": 0.4761},
    },
    "I2VSLC": {
        "SCENE":      {"AP": 0.7953, "CV": 0.4178, "HL": 0.0975,  "RL": 0.0946},
        "OBJECT":     {"AP": 0.4873, "CV": 0.2903, "HL": 0.0556,  "RL": 0.1623},
        "corel5k_5":  {"AP": 0.2361, "CV": 0.4865, "HL": 0.01373, "RL": 0.2306},
        "iaprtc12":   {"AP": 0.1436, "CV": 0.4982, "HL": 0.01537, "RL": 0.2051},
        "espgame":    {"AP": 0.2248, "CV": 0.5533, "HL": 0.0226,  "RL": 0.2095},
        "3sources":   {"AP": 0.4676, "CV": 0.6064, "HL": 0.2337,  "RL": 0.4593},
    },
    "GRAFS": {
        "yeast":      {"AP": 0.6609, "CV": 0.6356, "HL": 0.2254,  "RL": 0.2547},
        "SCENE":      {"AP": 0.7977, "CV": 0.4169, "HL": 0.0949,  "RL": 0.0925},
        "OBJECT":     {"AP": 0.4987, "CV": 0.2890, "HL": 0.0557,  "RL": 0.1600},
        "VOC07":      {"AP": 0.5913, "CV": 0.4253, "HL": 0.0858,  "RL": 0.1977},
        "MIRFlickr":  {"AP": 0.6795, "CV": 0.5784, "HL": 0.1741,  "RL": 0.1586},
        "3sources":   {"AP": 0.4718, "CV": 0.6167, "HL": 0.2257,  "RL": 0.4775},
    },
}

#: which datasets each paper used, in the paper's own row order
PAPER_DATASETS = {
    "TOCL":   ["SCENE", "OBJECT", "MIRFlickr", "corel5k_5", "iaprtc12", "3sources"],
    "UGRFS":  ["yeast", "SCENE", "VOC07", "MIRFlickr", "iaprtc12", "3sources"],
    "EF2FS":  ["emotions", "yeast", "SCENE", "MIRFlickr", "iaprtc12", "3sources"],
    "DHLI":   ["SCENE", "OBJECT", "MIRFlickr", "corel5k_5", "iaprtc12", "3sources"],
    "I2VSLC": ["3sources", "SCENE", "iaprtc12", "corel5k_5", "espgame", "OBJECT"],
    "GRAFS":  ["yeast", "SCENE", "OBJECT", "VOC07", "MIRFlickr", "3sources"],
}

#: where each row of reference numbers comes from, shown above every block
PAPER_SOURCES = {
    "TOCL":   "ACM MM 2025 — Table 2 (AP/Coverage), Table 3 (HL/RL)",
    "UGRFS":  "AAAI 2025 — Table 2 (AP/Coverage), Table 3 (HL/RL)",
    "EF2FS":  "Pattern Recognition 157 (2025) 110888 — Table 3 (AP/Coverage), Table 4 (HL/RL)",
    "DHLI":   "AAAI 2024 — Table 2 (AP/Coverage), Table 3 (HL/RL)",
    "I2VSLC": "Information Sciences 681 (2024) 121215 — Table 3 (AP/Coverage), Table 4 (HL/RL)",
    "GRAFS":  "Information Sciences 679 (2024) 121124 — Table 3 (AP/Coverage), Table 4 (HL/RL)",
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
            lines.append(f"### {alg}\n")
            if alg in PAPER_SOURCES:
                lines.append(f"<sub>论文来源：{PAPER_SOURCES[alg]}</sub>\n")
            lines.append("_尚未跑出结果。_\n")
            continue
        lines.append(f"### {alg}\n")
        if alg in PAPER_SOURCES:
            lines.append(f"<sub>论文来源：{PAPER_SOURCES[alg]}</sub>\n")
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

    header = ["# 复现结果 vs 六篇论文的结果表", ""]
    header.append("数值为 **1%–20% 特征百分比区间的聚合值**，± 为 **5 折间**标准差。"
                  "`paper ...` 后缀是论文自己报告的值与偏差方向（▲ 表示比论文更好）。")
    header.append("")
    header.append("> 未做超参数调优时（`DEFAULT_PARAMS` 全为 1.0）数值与论文不可直接比较；"
                  "调优请用 `python tune.py --alg <ALG> --data <DS>`。")
    header.append(">")
    header.append("> 论文基准值由 `pre-pdf/*.pdf` 的表格重新抽取核对（PyMuPDF 表格/坐标提取，"
                  "见模块 docstring）；Coverage 用除以标签数的论文口径列。")
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
