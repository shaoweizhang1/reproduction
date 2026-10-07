"""Draw every figure whose logs exist.

Run: python3 -m src.plotting.plot_all (from deepproblog/)
"""

from .curves import plot_dual_axis
from ..utils.paths import LOG_DIR

FIGURES = [
    {
        "title": "T1  single-digit MNIST addition",
        "filename": "t1.png",
        "loss_max": 3.0,
        "score_label": "Accuracy on test set",
        "series": [
            ("DeepProbLog", "t1_deepproblog"),
            ("CNN (concat)", "t1_baseline_concat"),
            ("CNN (separate)", "t1_baseline_separate"),
        ],
    },
    {
        "title": "T2  multi-digit MNIST addition",
        "filename": "t2.png",
        "loss_max": 6.0,
        "score_label": "Accuracy on test set",
        "series": [
            ("DeepProbLog", "t2_deepproblog"),
            ("CNN (separate)", "t2_baseline"),
        ],
    },
    {
        "title": "T5  Forth word algebra problems",
        "filename": "t5.png",
        "loss_max": 6.0,
        "score_label": "Accuracy on dev set",
        "series": [("DeepProbLog", "t5_deepproblog")],
    },
    {
        "title": "T6  coin-ball problem",
        "filename": "t6.png",
        "loss_max": 3.5,
        "score_label": "Accuracy on dev set",
        "series": [("DeepProbLog", "t6_coin_ball")],
    },
]

def main():
    for figure in FIGURES:
        present = [
            (label, name)
            for label, name in figure["series"]
            if (LOG_DIR / f"{name}.log").exists()
        ]
        if not present:
            print(f"Skipping {figure['filename']}: no logs yet")
            continue
        plot_dual_axis(
            figure["title"],
            figure["filename"],
            present,
            loss_max=figure["loss_max"],
            score_label=figure["score_label"],
        )

if __name__ == "__main__":
    main()
