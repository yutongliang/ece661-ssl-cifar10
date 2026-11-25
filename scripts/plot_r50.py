#!/usr/bin/env python3
"""Figures for the ResNet-50 comparison (see docs/RotNet.md, items F1-F3).

F1  r50_figure_b7       Linear-eval top-1 vs pretraining epoch, one bar per batch
                        size, one panel per method (SimCLR Figure B.7 format).
F2  r50_label_fraction  Fine-tune test top-1 vs label fraction, one line per
                        method, one panel per batch size (SimCLR Table 7 as a plot).
F3  r50_rotnet_pretext  RotNet rotation test top-1 (pretext) and class linear-eval
                        top-1 over pretraining, stacked panels sharing the epoch axis.

Methods appear only when their rows exist in the CSVs, so the same script adds
SimCLR, SimSiam and the supervised baseline once their results are merged.

    python scripts/plot_r50.py [--results results/r50] [--out-dir results/r50/figures]
"""
import argparse
import csv
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(ROOT, "results", "r50")
BATCHES = [256, 1024]
EPOCHS = list(range(100, 1001, 100))
FRACTIONS = [0.01, 0.03, 0.05, 0.1]
METHODS = ["rotnet", "simclr", "simsiam"]
NAMES = {"rotnet": "RotNet", "simclr": "SimCLR", "simsiam": "SimSiam",
         "supervised": "Supervised (from scratch)"}
# Batch size: two steps of the blue ramp used by the ResNet-20 Figure B.7.
BATCH_COLORS = {256: "#5598e7", 1024: "#1c5cab"}
BATCH_MARKERS = {256: "o", 1024: "s"}
# Method identity: fixed categorical slots; the supervised baseline is a neutral reference.
METHOD_COLORS = {"rotnet": "#257de0", "simclr": "#f47230", "simsiam": "#00b894",
                 "supervised": "#52514e"}
METHOD_MARKERS = {"rotnet": "o", "simclr": "s", "simsiam": "^", "supervised": "D"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def load_linear(path):
    acc, random_init = {}, None
    for row in read_csv(path):
        if row["method"] == "random_init":
            random_init = float(row["test_top1"]) * 100
            continue
        epoch = int(re.search(r"epoch(\d+)", row["run"]).group(1))
        acc[row["method"], int(row["batch_size"]), epoch] = float(row["test_top1"]) * 100
    return acc, random_init


def load_finetune(path):
    return {(row["method"], int(row["batch_size"]), float(row["label_fraction"])):
            float(row["test_top1"]) * 100 for row in read_csv(path)}


def load_pretext(log_dir):
    """Rotation test top-1 per logged epoch, keyed by batch size, from the Slurm logs."""
    curves = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*.out"))):
        text = open(path).read()
        batch = 256 if "195 steps/epoch" in text else 1024 if "48 steps/epoch" in text else None
        if batch is None:
            continue
        points = {int(e): float(a) * 100 for e, a in
                  re.findall(r"^epoch +(\d+) \|.*test rotation acc ([0-9.]+)", text, re.M)}
        curves[batch] = sorted(points.items())
    return curves


def setup_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, facecolor=SURFACE)
        print("Wrote", path)


def figure_b7(plt, acc, random_init, out_dir):
    methods = [m for m in METHODS if any(k[0] == m for k in acc)]
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2 if len(methods) == 1 else 3.6 * len(methods), 3.2),
                             dpi=200, sharey=True, squeeze=False)
    group, gap, y0 = 0.8, 0.03, 20
    width = group / len(BATCHES)
    for ax, method in zip(axes[0], methods):
        setup_axes(ax)
        for i, b in enumerate(BATCHES):
            xs = [k - group / 2 + (i + 0.5) * width for k in range(len(EPOCHS))]
            ys = [acc.get((method, b, e)) for e in EPOCHS]
            ax.bar([x for x, y in zip(xs, ys) if y is not None],
                   [y - y0 for y in ys if y is not None], bottom=y0, width=width - 2 * gap,
                   color=BATCH_COLORS[b], linewidth=0, label=f"Batch {b}")
            # Direct label on each series' peak only.
            best = max((y, x) for x, y in zip(xs, ys) if y is not None)
            ax.annotate(f"{best[0]:.1f}", (best[1], best[0]), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, color=INK_2)
        if random_init is not None:
            ax.axhline(random_init, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
        ax.set_xlim(-0.5, len(EPOCHS) - 0.5)
        ax.set_ylim(y0, 80)
        ax.set_xticks(range(len(EPOCHS)), [str(e) for e in EPOCHS])
        ax.set_xlabel("Pretraining epochs")
        if len(methods) > 1:
            ax.set_title(NAMES[method], fontsize=9, loc="left", color=INK)
    axes[0][0].set_ylabel("Linear eval top-1 (%)")
    axes[0][0].legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(-0.01, 1.16),
                      fontsize=8, handlelength=1.2, columnspacing=1.2)
    note = f"ResNet-50, CIFAR-10 test. Axis starts at {y0}%"
    if random_init is not None:
        note += f". Dashed: random-init encoder {random_init:.1f}%"
    axes[0][-1].text(1.0, 1.04, note,
                     transform=axes[0][-1].transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, out_dir, "r50_figure_b7")


def label_fraction(plt, ft, out_dir):
    methods = [m for m in ["supervised"] + METHODS if any(k[0] == m for k in ft)]
    fig, axes = plt.subplots(1, len(BATCHES), figsize=(7.2, 3.2), dpi=200, sharey=True)
    xs = list(range(len(FRACTIONS)))
    for ax, b in zip(axes, BATCHES):
        setup_axes(ax)
        for m in methods:
            ys = [ft.get((m, b, f)) for f in FRACTIONS]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts:
                continue
            style = (0, (4, 3)) if m == "supervised" else "-"
            ax.plot(*zip(*pts), color=METHOD_COLORS[m], linewidth=2, linestyle=style,
                    marker=METHOD_MARKERS[m], markersize=5, markeredgecolor=SURFACE,
                    markeredgewidth=1.2, label=NAMES[m])
            for x, y in pts:  # few points per line, so label each value
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK_2)
        ax.set_xticks(xs, [f"{f * 100:g}%" for f in FRACTIONS])
        ax.set_xlim(-0.4, len(FRACTIONS) - 0.6)
        ax.set_ylim(20, 100)
        ax.set_xlabel("Labeled training images")
        ax.set_title(f"Batch {b}", fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("Fine-tune test top-1 (%)")
    if len(methods) > 1:
        axes[0].legend(frameon=False, loc="upper left", fontsize=7.5)
    else:
        axes[-1].text(1.0, 1.04, f"{NAMES[methods[0]]} fine-tune, ResNet-50, CIFAR-10 test",
                      transform=axes[-1].transAxes, ha="right", va="bottom", fontsize=6.5, color=MUTED)
    fig.tight_layout()
    save(fig, out_dir, "r50_label_fraction")


def pretext(plt, curves, acc, out_dir):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.2, 4.6), dpi=200, sharex=True,
                                      gridspec_kw={"hspace": 0.25})
    for ax in (top, bottom):
        setup_axes(ax)
    for b in BATCHES:
        color = BATCH_COLORS[b]
        if b in curves:
            e, a = zip(*curves[b])
            top.plot(e, a, color=color, linewidth=2, label=f"Batch {b}")
        pts = [(e, acc[("rotnet", b, e)]) for e in EPOCHS if ("rotnet", b, e) in acc]
        if pts:
            e, a = zip(*pts)
            bottom.plot(e, a, color=color, linewidth=2, marker=BATCH_MARKERS[b], markersize=5,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, label=f"Batch {b}")
            peak = max(pts, key=lambda p: p[1])
            bottom.annotate(f"peak {peak[1]:.1f} (epoch {peak[0]})", peak,
                            xytext=(0, -16 if b == 256 else 8), textcoords="offset points",
                            ha="center", fontsize=6.5, color=INK_2)
            bottom.annotate(f"{a[-1]:.1f}", (e[-1], a[-1]), xytext=(7, 4 if b == 256 else -5),
                            textcoords="offset points",
                            fontsize=6.5, color=INK_2, va="center")
    top.set_ylim(75, 100)
    for b in BATCHES:
        if b in curves:
            top.annotate(f"{curves[b][-1][1]:.1f}", curves[b][-1], xytext=(7, 4 if b == 256 else -5),
                         textcoords="offset points", fontsize=6.5, color=INK_2, va="center")
    top.set_ylabel("Rotation test top-1 (%)")
    top.set_title("Pretext task: predict the rotation (4 classes)", fontsize=9, loc="left", color=INK)
    top.legend(frameon=False, loc="lower right", fontsize=7.5, ncol=2)
    bottom.set_ylim(50, 76)
    bottom.set_ylabel("Linear eval top-1 (%)")
    bottom.set_title("Downstream: CIFAR-10 classes from frozen features (10 classes)",
                     fontsize=9, loc="left", color=INK)
    bottom.set_xlabel("Pretraining epochs")
    bottom.set_xlim(0, 1080)
    bottom.set_xticks(EPOCHS)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.1)
    save(fig, out_dir, "r50_rotnet_pretext")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.results, "figures")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8.5, "axes.labelcolor": INK_2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURFACE,
                         "pdf.fonttype": 42})

    acc, random_init = load_linear(os.path.join(args.results, "linear.csv"))
    ft = load_finetune(os.path.join(args.results, "finetune.csv"))
    figure_b7(plt, acc, random_init, out_dir)
    label_fraction(plt, ft, out_dir)
    pretext(plt, load_pretext(os.path.join(args.results, "pretrain")), acc, out_dir)


if __name__ == "__main__":
    main()
