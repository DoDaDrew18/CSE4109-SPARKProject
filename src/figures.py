"""Demo figures from ``data/results/`` (run ``python -m src.nowcast`` first).

    python -m src.figures            # writes demo/fig_*.png

* ``fig_models.png``    MAE vs final for each model, official county-weeks
* ``fig_timeline.png``  one metro's season: final value, carry-forward, best model
* ``fig_ablation.png``  what dropping each input costs the history GBM
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import evaluate  # noqa: E402
from src.nowcast import ORDER, scoreable  # noqa: E402
from src.study import METROS  # noqa: E402

RESULTS = Path("data/results")
OUT = Path("demo")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29c"
LABELS = {"0_naive": "Carry last value forward", "1_ar_season": "Ridge: AR + season",
          "2_weather": "Ridge: + weather", "3_wastewater": "Ridge: + wastewater",
          "4_news": "Ridge: + news", "5_arima": "ARIMA(2,1,0)",
          "6_gbm_history": "Gradient boosting, US history",
          "7_xgb_history": "XGBoost, US history",
          "8_rf_history": "Random forest, US history",
          "9_svm_history": "SVM (RBF), US history",
          "10_linear_history": "Linear, US history"}


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x" if ax.get_xlabel() else "y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def models(metrics: pd.DataFrame, tests: pd.DataFrame, out: Path) -> None:
    m = metrics[metrics["group"] == "all"].set_index("rung")
    rungs = [r for r in ORDER if r in m.index]
    m = m.loc[rungs[::-1]]
    best = m["mae"].idxmin()
    p = tests[tests["vs"] == "0_naive"].set_index("rung")["pvalue"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=200)
    colors = [BLUE if r == best else (INK if r == "0_naive" else GRAY) for r in m.index]
    ax.barh([LABELS.get(r, r) for r in m.index], m["mae"], color=colors, height=0.6)
    for i, (r, row) in enumerate(m.iterrows()):
        note = f"{row['mae']:.2f}"
        if r != "0_naive":
            word = "better" if row["skill"] > 0 else "worse"
            note += f"   {abs(row['skill']):.0%} {word} than carry-forward"
            if r in p and p[r] < 0.001:
                note += " (p<0.001)"
            elif r in p:
                note += f" (p={p[r]:.3f})" if p[r] < 0.01 else f" (p={p[r]:.2f})"
        ax.text(row["mae"] + 0.01, i, note, va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Mean absolute error vs final value (% of ED visits)", color=MUTED, fontsize=9)
    ax.set_xlim(0, m["mae"].max() * 2.1)
    ax.set_title(f"Nowcasting this week's respiratory ED share ({int(m['n'].iloc[0])} "
                 "county-weeks, 4 metros)", loc="left", fontsize=10.5, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_models.png")
    plt.close(fig)


def timeline(preds: pd.DataFrame, metro: str, rung: str, out: Path) -> None:
    rows = preds[(preds["metro"] == metro) & preds["rung"].isin(["0_naive", rung])]
    rows = rows.drop_duplicates(["rung", "week_end"])
    wide = rows.pivot_table(index="week_end", columns="rung", values="pred")
    final = rows.drop_duplicates("week_end").set_index("week_end")["ed_resp_final"]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=200)
    ax.plot(final.index, final, color=INK, linewidth=2, label="Final value (known weeks later)")
    ax.plot(wide.index, wide["0_naive"], color=GRAY, linewidth=1.5, linestyle="--",
            label="Carry last value forward")
    ax.plot(wide.index, wide[rung], color=BLUE, linewidth=2, label=LABELS[rung])
    ax.set_ylabel("% of ED visits: COVID + flu + RSV", color=MUTED, fontsize=9)
    ax.set_title(f"{METROS[metro].name}: nowcast made each Saturday from that day's data",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / f"fig_timeline_{metro}.png")
    plt.close(fig)


def ablation(abl: pd.DataFrame, out: Path) -> None:
    a = abl[abl["variant"].str.startswith("gbm:")].copy()
    if a.empty:
        return
    names = {"gbm:full": "Full model", "gbm:-season": "Without season",
             "gbm:-momentum": "Without momentum (recent trend)",
             "gbm:-study_metros": "Without the study metros' own history"}
    a = a.set_index("variant").loc[[v for v in names if v in a["variant"].values][::-1]]
    fig, ax = plt.subplots(figsize=(7.5, 2.6), dpi=200)
    colors = [BLUE if v == "gbm:full" else GRAY for v in a.index]
    ax.barh([names[v] for v in a.index], a["skill_vs_naive"], color=colors, height=0.6)
    for i, (v, row) in enumerate(a.iterrows()):
        ax.text(max(row["skill_vs_naive"], 0) + 0.005, i, f"{row['skill_vs_naive']:+.0%}",
                va="center", fontsize=8.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("MAE reduction vs carrying the last value forward", color=MUTED, fontsize=9)
    ax.set_title("Ablation: what each input is worth", loc="left", fontsize=10.5, color=INK)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the demo figures.")
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--metro", default="chi")
    args = parser.parse_args()
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models(pd.read_csv(res / "metrics_ladder.csv"), pd.read_csv(res / "wilcoxon_ladder.csv"), out)
    preds = pd.read_csv(res / "predictions.csv", dtype={"fips": str},
                        parse_dates=["as_of", "week_end"])
    scored = evaluate.dedupe_series(scoreable(preds))
    for metro in dict.fromkeys([args.metro, "lou"]):
        timeline(scored, metro, "6_gbm_history", out)
    ablation(pd.read_csv(res / "metrics_ablation.csv"), out)
    print(f"wrote {out}/fig_*.png")


if __name__ == "__main__":
    main()
