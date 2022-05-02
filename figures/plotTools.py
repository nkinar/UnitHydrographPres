#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from clean_numbers import clean_numbers
import matplotlib as mpl

def imagesc(Umat,x=None,y=None,**kwargs):
    """
    Replacement for imagesc function
    """
    origin = kwargs.pop('origin', 'upper')
    if (x is not None and y is not None):
        if origin == 'upper':
            extent = (x[0], x[-1], y[-1], y[0])
        elif origin == 'lower':
            extent = (x[0], x[-1], y[0], y[-1])
        plt.imshow(Umat, extent = extent, aspect = 'auto', interpolation = 'nearest', origin=origin,
                   **kwargs)
    else:
        plt.imshow(Umat, aspect = 'auto', interpolation='nearest', origin=origin, **kwargs)
    return plt
    
       
def abs_imagesc(Umat,x=None,y=None,**kwargs):
    """
    Plot abs value of matrix
    """
    return imagesc(np.abs(Umat),x,y,**kwargs)


def power_imagesc(Umat,x=None,y=None,log_power=False,**kwargs):
    """
    Plot power value of matrix as |U|^2
    """
    a = np.abs(Umat)**2
    if(log_power):
        a = np.log( a ) # warning: if values are close to zero
        a = clean_numbers(a)
    return imagesc(a,x,y,**kwargs)

def turn_off_ticks_xaxis():
    plt.xticks([])

def turn_off_ticks_yaxis():
    plt.yticks([])

def turn_ticklabels_off_xaxis():
    cur_axes = plt.gca()
    cur_axes.axes.get_xaxis().set_ticklabels([])

def turn_ticklabels_off_yaxis():
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_ticklabels([])

def make_axes_tight():
    mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
    mpl.rcParams['axes.xmargin'] = 0
    mpl.rcParams['axes.ymargin'] = 0

def set_figure_text_size(size):
    mpl.rcParams.update({'font.size': size})

def use_ggplot_style():
    plt.style.use('ggplot')

def remove_colorbar():
    plt.gca().images[-1].colorbar.remove()

def get_jet_colormap():
    cmap = plt.get_cmap('jet')
    return cmap

def set_matplotlib_fontsize(fs):
    mpl.rcParams.update({'font.size': fs})

def autoscale_tight_x():
    plt.autoscale(enable=True, axis='x', tight=True)

def autoscale_tight_y():
    plt.autoscale(enable=True, axis='y', tight=True)

def set_legend_marker_size(lgnd, siz):
    for handle in lgnd.legendHandles:
        handle._legmarker.set_markersize(siz)


def export_legend(legend, filename, dpi=300):
    """
    Export legend to figure file
    REFERENCE: https://stackoverflow.com/questions/4534480/get-legend-as-a-separate-picture-in-matplotlib
    :param legend:          as the legend object
    :param filename:        as the filename to save
    :param expand:          expansion around border
    :return:
    """
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox)

