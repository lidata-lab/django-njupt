"""Some Functions used by other modules.

"""

import numpy as np

__ALL__ = [
    "to_numpy"
]

def to_numpy(values):
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if values.shape[0] == 1:
        return values
    else:
        return np.squeeze(values)
