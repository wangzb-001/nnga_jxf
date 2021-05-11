import numpy as np


def Create1(bound, row, cloum):
    cld_x = bound[0] + np.random.rand(row, cloum) * (bound[1] - bound[0])
    # cld_x = np.random.rand(row, cloum)
    return cld_x
