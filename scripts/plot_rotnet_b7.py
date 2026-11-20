#!/usr/bin/env python3
"""RotNet version of SimCLR Figure B.7: linear-eval top-1 vs pretraining epoch.

Each bar is the mean over the three pretraining learning rates (0.03, 0.1, 0.3)
at one batch size and milestone; the error bar is their sample SD (rates, not seeds).

    python scripts/plot_rotnet_b7.py [--probes PATH] [--out-dir figures]
"""
import argparse
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROBES = os.path.join(ROOT, "dcc_results", "rotnet_sweep_20261115", "results",
                              "rotnet_linear_probes.csv")
BATCHES = [256, 512, 1024, 2048, 4096]
LRS = [0.03, 0.1, 0.3]
EPOCHS = list(range(100, 1001, 100))
# Ordinal one-hue ramp (light -> dark = small -> large batch); validated for light surface.
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                          "#e1e0d9", "#c3c2b7")


def load(path):
    acc, random_init = {}, None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["run_id"] == "random_init":
                random_init = float(row["test_acc"]) * 100
                continue
            assert row["probe_epochs"] == "100", row["run_id"]
            key = (int(row["global_batch_size"]), float(row["lr"]), int(row["checkpoint_epoch"]))
            acc[key] = float(row["test_acc"]) * 100
    missing = [(b, l, e) for b in BATCHES for l in LRS for e in EPOCHS if (b, l, e) not in acc]
    if missing:
        raise SystemExit(f"{len(missing)} probes missing, e.g. {missing[:3]}")
    return acc, random_init


def summarize(acc):
    rows = []
    for b in BATCHES:
        for e in EPOCHS:
            values = [acc[b, l, e] for l in LRS]
            rows.append({"global_batch_size": b, "checkpoint_epoch": e,
                         "mean_top1": round(statistics.mean(values), 3),
                         "sd_over_lr": round(statistics.stdev(values), 3),
                         **{f"top1_lr{l:g}": round(v, 3) for l, v in zip(LRS, values)}})
    return rows


def plot(rows, random_init, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    group, gap = 0.8, 0.02            # group width per epoch; surface gap between bars
    width = group / len(BATCHES)
    y0 = 50
    for i, b in enumerate(BATCHES):
        series = [r for r in rows if r["global_batch_size"] == b]
        xs = [k - group / 2 + (i + 0.5) * width for k in range(len(EPOCHS))]
        ax.bar(xs, [r["mean_top1"] - y0 for r in series], bottom=y0, width=width - 2 * gap,
               color=COLORS[i], linewidth=0, label=str(b))
        ax.errorbar(xs, [r["mean_top1"] for r in series], yerr=[r["sd_over_lr"] for r in series],
                    fmt="none", ecolor=INK_2, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=3)

    ax.set_xlim(-0.5, len(EPOCHS) - 0.5)
    ax.set_ylim(y0, 74)
    ax.set_xticks(range(len(EPOCHS)), [str(e) for e in EPOCHS])
    ax.set_xlabel("Pretraining epochs")
    ax.set_ylabel("Linear eval top-1 (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(title="Batch size", ncol=5, frameon=False, loc="upper left",
              bbox_to_anchor=(-0.01, 1.2), fontsize=8, title_fontsize=8, handlelength=1.2,
              columnspacing=1.2, alignment="left")
    ax.text(1.0, 1.02, f"Axis starts at {y0}%. Random-init encoder: {random_init:.1f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"rotnet_figure_b7.{ext}")
        fig.savefig(path, facecolor=SURFACE)
        print("Wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", default=DEFAULT_PROBES)
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()
    acc, random_init = load(args.probes)
    rows = summarize(acc)
    os.makedirs(args.out_dir, exist_ok=True)
    table = os.path.join(args.out_dir, "rotnet_figure_b7_summary.csv")
    with open(table, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", table)
    plot(rows, random_init, args.out_dir)


if __name__ == "__main__":
    main()
