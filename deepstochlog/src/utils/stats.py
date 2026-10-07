from functools import partial

import numpy as np


def use_sample_std():
    # upstream's evaluate.py reports numpy's default population SD; the README tables use the sample SD
    import examples.evaluate

    examples.evaluate.std = partial(np.std, ddof=1)
