from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


METRIC_SPECS = [
    # mean_col, std_col, title, y_label, scale
    ("mean_Z1", "std_Z1", "Z1", "Volume utilization (%)", 100.0),
    ("mean_Z3", "std_Z3", "Z3", "Avg. normalized CG deviation (–)", 1.0),
    ("mean_RDx", "std_RDx", "RDx", "RDx deviation (%)", 1.0),
    ("mean_RDy", "std_RDy", "RDy", "RDy deviation (%)", 1.0),
    ("mean_RDz", "std_RDz", "RDz", "RDz deviation (%)", 1.0),
    ("mean_time_sec", "std_time_sec", "Time", "Mean elapsed time per instance (s)", 1.0),
]
ROOT = Path(__file__).resolve().parents[1]

FILE_BASELINE = ROOT / "results" / "BR-Original" / "_summary" / "br_original_summary.xlsx"
FILE_TWO_PHASE = ROOT / "results" / "BR-Original-two_phase" / "_summary" / "BR-Original-two_phase_summary.xlsx"

ALGO_BASELINE = "Baseline-DBLF"
ALGO_TWO_PHASE = "Two-Phase-DBLF"

OUT_DIR = ROOT / "results" / "_comparison_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODES = ["six_way", "C1_respect"]

ALGO_COLORS = {
    ALGO_BASELINE: "#1f77b4",   # blue
    ALGO_TWO_PHASE: "#d62728",  # red
}

TITLE_FONT_SIZE = 13
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11
SUPTITLE_FONT_SIZE = 16


def load_summary(path: Path, algo_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    df["algorithm"] = algo_name
    return df


def br_sort_key(br: str) -> int:
    br = str(br).strip()
    return int(br.replace("BR", ""))


def plot_mode(df: pd.DataFrame, mode: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))
    axes = axes.flatten()

    algo_order = [ALGO_BASELINE, ALGO_TWO_PHASE]

    for i, (mean_col, std_col, title, ylab, scale) in enumerate(METRIC_SPECS):
        ax = axes[i]
        sub = df[df["mode"] == mode].copy()

        piv_mean = sub.pivot_table(index="br_class", columns="algorithm", values=mean_col, aggfunc="mean")
        piv_std = sub.pivot_table(index="br_class", columns="algorithm", values=std_col, aggfunc="mean")

        # sort BR0..BR15
        piv_mean = piv_mean.reindex(sorted(piv_mean.index, key=br_sort_key))
        piv_std = piv_std.reindex(piv_mean.index)

        # enforce algorithm order if present
        piv_mean = piv_mean[[c for c in algo_order if c in piv_mean.columns]]
        piv_std = piv_std[piv_mean.columns]

        # scale
        piv_mean = piv_mean * scale
        piv_std = piv_std * scale

        colors = [ALGO_COLORS.get(c, "#7f7f7f") for c in piv_mean.columns]

        is_time = mean_col == "mean_time_sec"
        n_algos = len(piv_mean.columns)
        group_centers = range(len(piv_mean.index))

        br_labels = [str(br_sort_key(b)) for b in piv_mean.index]
        if is_time:
            br_values = [br_sort_key(b) for b in piv_mean.index]
            for j, algo in enumerate(piv_mean.columns):
                xs = br_values
                ax.plot(
                    xs,
                    piv_mean[algo].values,
                    color=colors[j],
                    marker="o",
                    linewidth=1.5,
                )
                ax.errorbar(
                    xs,
                    piv_mean[algo].values,
                    yerr=piv_std[algo].values,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=2,
                    zorder=5,
                )
        else:
            # draw bars without legend
            piv_mean.plot(kind="bar", ax=ax, width=0.8, color=colors, legend=False)
            ax.set_xticks(range(len(br_labels)))
            ax.set_xticklabels(br_labels)

            # overlay std error bars
            bar_width = 0.8
            offsets = [(-bar_width / 2) + (j + 0.5) * (bar_width / n_algos) for j in range(n_algos)]

            for j, algo in enumerate(piv_mean.columns):
                xs = [k + offsets[j] for k in group_centers]
                ax.errorbar(
                    xs,
                    piv_mean[algo].values,
                    yerr=piv_std[algo].values,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1,
                    capsize=2,
                    zorder=5,
                )

        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.set_xlabel("BR class", fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(ylab, fontsize=LABEL_FONT_SIZE)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=0, labelsize=TICK_FONT_SIZE)
        ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
        if is_time:
            ax.set_ylim(0.1, 1.3)
            ax.set_yticks([round(0.1 * i, 1) for i in range(1, 14)])
            ax.set_xlim(0, 16)
            ax.set_xticks(list(range(-1, 16, 1)))

    legend_items = [
        Patch(facecolor=ALGO_COLORS[ALGO_BASELINE], label=ALGO_BASELINE),
        Patch(facecolor=ALGO_COLORS[ALGO_TWO_PHASE], label=ALGO_TWO_PHASE),
        Line2D(
            [0], [0],
            color="black",
            linewidth=1,
            marker="_",
            markersize=10,
            label="±1 std. dev.",
        ),
    ]
    fig.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=True,
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.suptitle(
        f"BR-Original comparison (mode = {mode})",
        fontsize=SUPTITLE_FONT_SIZE,
        y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out = OUT_DIR / f"compare_BR_original_{mode}.png"
    fig.savefig(out, dpi=1200)
    plt.close(fig)
    print("Saved:", out)


def main():
    df_base = load_summary(FILE_BASELINE, ALGO_BASELINE)
    df_two = load_summary(FILE_TWO_PHASE, ALGO_TWO_PHASE)
    df = pd.concat([df_base, df_two], ignore_index=True)

    print("\nSanity check (rows per algorithm/mode):")
    print(df.groupby(["algorithm", "mode"]).size())

    # quick validation: required columns
    required = {"br_class", "mode", "algorithm"}
    for mean_col, std_col, _, _, _ in METRIC_SPECS:
        required.add(mean_col)
        required.add(std_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in merged df: {sorted(missing)}")

    for mode in MODES:
        if (df["mode"] == mode).sum() == 0:
            print(f"Warning: no rows for mode={mode}")
            continue
        plot_mode(df, mode)

    merged = OUT_DIR / "merged_BR_original_summaries.xlsx"
    df.to_excel(merged, index=False)
    print("Saved:", merged)


if __name__ == "__main__":
    main()
