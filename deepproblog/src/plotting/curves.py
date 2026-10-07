"""log/*.log -> figs/*.png, in the shape of the paper's Figure 3: training
loss on the left axis, test accuracy on the right.
"""

import matplotlib.pyplot as plt

from .style import AXIS, INK_SOFT, apply_style, colors_for
from ..utils.paths import FIGS_DIR, LOG_DIR

from deepproblog.utils.logger import Logger

def load(name):
    logger = Logger()
    logger.read_from_file(str(LOG_DIR / f"{name}.log"))
    return logger

def _finish(figure, filename):
    FIGS_DIR.mkdir(exist_ok=True)
    path = FIGS_DIR / filename
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", transparent=True)
    plt.close(figure)
    print("Wrote", path)

def plot_dual_axis(
    title, filename, series, score="Accuracy", loss_max=3.0,
    score_label="Accuracy on test set",
):
    apply_style()
    figure, loss_axis = plt.subplots(figsize=(6.4, 4.0))

    score_axis = loss_axis.twinx()
    score_axis.grid(False)

    handles = []
    for label, name in series:
        logger = load(name)
        loss_color, score_color = colors_for(label)
        for axis, column, color, suffix in (
            (loss_axis, "loss", loss_color, "loss"),
            (score_axis, score, score_color, score),
        ):
            iterations, values = logger.get_attribute(column, include_indices=True)
            if not iterations:
                continue
            line, = axis.plot(
                iterations, values, color=color, label=f"{label} {suffix}"
            )
            handles.append(line)

    loss_axis.set_xlabel("iterations")
    loss_axis.set_ylabel("training loss")
    loss_axis.set_ylim(0, loss_max)
    loss_axis.spines["top"].set_visible(False)
    score_axis.set_ylabel(score_label)
    score_axis.set_ylim(0, 1)
    for spine in ("top", "left", "right"):
        score_axis.spines[spine].set_visible(spine == "right")
    score_axis.spines["right"].set_color(AXIS)

    loss_axis.set_title(title, loc="left")
    loss_axis.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        labelcolor=INK_SOFT,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    _finish(figure, filename)
