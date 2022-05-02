import numpy as np


def baseflow_sep_run(hin, alpha):
    """
    Apply the baseflow separation filter
    :param h:       as the measured streamflow
    :param alpha:   as the filtering parameter
    :return:
    """
    h = np.nan_to_num(np.asarray(hin), nan=0)
    n = len(h)
    y = np.zeros(n)
    term = 0.5 * (1 + alpha)
    for k in range(1, n):
        hk = h[k]
        hksub1 = h[k-1]
        qk = (alpha*y[k-1]) + term*(hk-hksub1)
        if qk < 0:
            qk = 0
        y[k] = qk
    return y

