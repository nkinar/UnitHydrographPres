import matplotlib.pyplot as plt
import numpy as np
from constants import *
from constant_labels import create_label
plt.rcParams.update({'font.size': FONT_SIZ})


def showplot_update(desc, q, fname):
    qplot = np.copy(q)
    qplot[qplot == 0] = np.nan

    plt.figure(figsize=FIG_SIZE_SMALL_NARROW)
    plt.plot(q)
    plt.xlabel('Time (minutes)')
    plt.ylabel(create_label('', 'm^3 s^-1'))
    plt.tight_layout()
    plt.savefig(fname + desc)
    plt.show(block=False)


def showplot_update_hours(desc, q, fname):
    qplot = np.copy(q)
    qplot[qplot == 0] = np.nan

    plt.figure(figsize=FIG_SIZE_SMALL_NARROW)
    plt.plot(q)
    plt.xlabel('Time (hours)')
    plt.ylabel(create_label('', 'm^3 s^-1'))
    plt.tight_layout()
    plt.savefig(fname + desc)
    plt.show(block=False)


def longer_duration(qin, Tknown, Treq_n, fname):
    """
    Signal processing to obtain longer duration
    :param qin:         as the known unit hydrograph (m^3/s)
    :param Tknown:      as the known time of excess precipitation (hr)
    :param Treq_n:      as the required time of the unit hydrograph in (Tknown * n) sequences
    :param fname:       as the filename to save out plots
    :return:
    """
    s = np.copy(qin)          # sum
    q = np.copy(qin)          # time-shifted sequence
    n = len(qin)              # length of original sequence
    nt = Tknown
    z = np.zeros(Tknown)
    showplot_update('0', q, fname)
    for k in range(Treq_n-1):
        q = np.insert(q, 0, z, axis=0)
        q = q[:-Tknown]
        s += q
        showplot_update_hours(str(k+1), q, fname)
    out = s / Treq_n
    return out, s


def shorter_duration(qin, Tknown, Treq, showplot, fname):
    """
    Signal processing function to obtain shorter duration hydrograph.
    The Tknown, Treq are assumed to have the same units.
    :param qin:             as the known unit hydrograph (m^3/s)
    :param Tknown:          as the known time of excess precipitation (hr)
    :param Treq:            as the required time of the unit hydrograph
    :param showplot:        True to plot output as the algorithm progresses
    :param fname:           as the filename to save out plots
    :return:
    """
    s = np.copy(qin)          # sum
    q = np.copy(qin)          # time-shifted sequence
    n = len(qin)              # length of original sequence

    nt = Tknown
    z = np.zeros(Tknown)
    plim_first = 0
    plim_second = nt/2 - nt/4
    plim_third = plim_second + nt/4
    for k in range(nt):
        q = np.insert(q, 0, z, axis=0)
        q = q[:-Tknown]
        s += q
        if showplot:
            if k == plim_first:
                showplot_update('first', q, fname)
            if k == plim_second:
                showplot_update('second', q, fname)
            if k == plim_third:
                showplot_update('third', q, fname)
        # END
    ztarget = np.zeros(Treq)
    slag = np.copy(s)
    slag = np.insert(slag, 0, ztarget, axis=0)
    slag = slag[:-Treq]
    diff = (slag - s) * Tknown
    diff[diff < 0] = 0

    plt.figure()
    plt.plot(s, label='S-curve', linewidth=LINE_WIDTH)
    plt.plot(slag, label='S-curve lagged', linewidth=LINE_WIDTH)
    plt.xlabel('Time (minutes)')
    plt.ylabel(create_label('Discharge', 'm^3 s^-1'))
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname + '-second-step')
    plt.show(block=False)
    return diff




