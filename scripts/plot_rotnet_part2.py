#!/usr/bin/env python3
"""Part 2: CIFAR-10 test top-1 versus label fraction for RotNet and supervised.

Lines: RotNet fine-tune (encoder + classifier), supervised from scratch (same
subsets and recipe), and RotNet frozen encoder + linear probe. Single seed.

    python scripts/plot_rotnet_part2.py [--results PATH] [--out-dir results/rotnet]
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(ROOT, "results", "rotnet", "part2", "results.csv")
FRACTIONS = [0.01, 0.05, 0.08, 0.1, 0.2, 1.0]
SERIES = [  # (key, label, color, marker, linestyle)
    ("finetune", "RotNet fine-tune", "#2a78d6", "o", "-"),
    ("supervised", "Supervised from scratch", "#eb6834", "s", "-"),
    ("frozen", "RotNet frozen + linear", "#1baf7a", "^", "--"),
]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")
KIND = {"finetune": "finetune", "supervised": "supervised", "linear_eval_pool1": "frozen"}


def load(path):
    acc = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (KIND[row["experiment"]], float(row["label_fraction"]))
            if key in acc:
                raise SystemExit(f"duplicate result for {key}")
            acc[key] = {"top1": float(row["test_acc"]) * 100, "top3": float(row["test_top3"]) * 100}
    missing = [(k, f) for k, *_ in SERIES for f in FRACTIONS if (k, f) not in acc]
    if missing:
        raise SystemExit(f"missing results: {missing}")
    return acc


def summarize(acc):
    rows = []
    for f in FRACTIONS:
        row = {"label_fraction": f, "labeled_images": round(50000 * f)}
        for k, *_ in SERIES:
            row[f"{k}_top1"] = round(acc[k, f]["top1"], 2)
        row["finetune_minus_supervised"] = round(row["finetune_top1"] - row["supervised_top1"], 2)
        for k, *_ in SERIES:
            row[f"{k}_top3"] = round(acc[k, f]["top3"], 2)
        rows.append(row)
    return rows


def plot(acc, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    xs = list(range(len(FRACTIONS)))  # evenly spaced categories, not a numeric scale
    label_y = {"finetune": -2.2, "supervised": 2.2, "frozen": 0}  # supervised ends higher
    for key, label, color, marker, style in SERIES:
        ys = [acc[key, f]["top1"] for f in FRACTIONS]
        ax.plot(xs, ys, color=color, linestyle=style, linewidth=2, marker=marker, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(f"{label}  {ys[-1]:.1f}", (xs[-1], ys[-1]), xytext=(8, label_y[key] * 3),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    gap = acc["finetune", 0.01]["top1"] - acc["supervised", 0.01]["top1"]
    y_ft, y_sup = acc["finetune", 0.01]["top1"], acc["supervised", 0.01]["top1"]
    ax.annotate("", (-0.12, y_ft), xytext=(-0.12, y_sup),
                arrowprops={"arrowstyle": "<->", "color": INK_2, "linewidth": 0.8})
    ax.text(-0.18, (y_ft + y_sup) / 2, f"+{gap:.1f}\npts", ha="right", va="center",
            fontsize=8, color=INK_2)

    ax.set_xticks(xs, [f"{f * 100:g}%\n{round(50000 * f):,}" for f in FRACTIONS])
    ax.set_xlim(-0.75, len(FRACTIONS) - 0.4)
    ax.set_ylim(20, 100)
    ax.set_xlabel("Labeled training images (fraction of CIFAR-10 and count; categories, not to scale)")
    ax.set_ylabel("Test top-1 (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(-0.01, 1.14), fontsize=8,
              handlelength=2.2, columnspacing=1.4)
    ax.text(1.0, 1.02, "Single seed, seed-0 class-balanced subsets", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"rotnet_part2_label_fraction.{ext}")
        fig.savefig(path, facecolor=SURFACE)
        print("Wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results", "rotnet"))
    args = parser.parse_args()
    acc = load(args.results)
    rows = summarize(acc)
    os.makedirs(args.out_dir, exist_ok=True)
    table = os.path.join(args.out_dir, "rotnet_part2_summary.csv")
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", table)
    plot(acc, args.out_dir)


if __name__ == "__main__":
    main()
