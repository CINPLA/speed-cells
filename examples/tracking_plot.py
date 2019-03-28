import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from scipy.ndimage.measurements import center_of_mass


def plot_path(x, y, t, box_size, spike_times=None,
              color='grey', alpha=0.5, origin='upper',
              spike_color='r', rate_markersize=False, markersize=10.,
              animate=False, ax=None):
    """
    Plot path visited

    Parameters
    ----------
    x : array
        1d vector of x positions
    y : array
        1d vector of y positions
    t : array
        1d vector of time at x, y positions
    spike_times : array
    box_size : scalar
        size of spatial 2d square
    color : path color
    alpha : opacity of path
    spike_color : spike marker color
    rate_markersize : bool
        scale marker size to firing rate
    markersize : float
        size of spike marker
    animate : bool
    ax : matplotlib axes

    Returns
    -------
    out : ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(
            111, xlim=[0, box_size], ylim=[0, box_size], aspect=1)

    ax.plot(x, y, c=color, alpha=alpha)
    if spike_times is not None:
        spikes_in_bin, _ = np.histogram(spike_times, t)
        is_spikes_in_bin = spikes_in_bin > 0

        if rate_markersize:
            markersize = spikes_in_bin[is_spikes_in_bin] * markersize
        ax.scatter(x[:-1][is_spikes_in_bin], y[:-1][is_spikes_in_bin],
                   facecolor=spike_color, edgecolor=spike_color,
                   s=markersize)

    ax.grid(False)
    if origin == 'upper':
        ax.invert_yaxis()
    return ax


def animate_path(x, y, t, box_size, spike_times=None,
              color='grey', alpha=0.5, origin='upper',
              spike_color='r', rate_markersize=False, markersize=10.,
              animate=False, ax=None, title=''):
    """
    Plot path visited

    Parameters
    ----------
    x : array
        1d vector of x positions
    y : array
        1d vector of y positions
    t : array
        1d vector of time at x, y positions
    spike_times : array
    box_size : scalar
        size of spatial 2d square
    color : path color
    alpha : opacity of path
    spike_color : spike marker color
    rate_markersize : bool
        scale marker size to firing rate
    markersize : float
        size of spike marker
    animate : bool
    ax : matplotlib axes

    Returns
    -------
    out : ax
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(
            111, xlim=[0, box_size], ylim=[0, box_size], aspect=1)
    if spike_times is not None:
        spikes_in_bin, _ = np.histogram(spike_times, t)
        is_spikes_in_bin = np.array(spikes_in_bin, dtype=bool)

        if rate_markersize:
            markersizes = spikes_in_bin[is_spikes_in_bin]*markersize
        else:
            markersizes = markersize*np.ones(is_spikes_in_bin.size)
    ax.set_title(title)
    ax.grid(False)
    if origin == 'upper':
        ax.invert_yaxis()
    import time
    plt.show()
    for idx, x, y, active, msize in zip(range(len(x)), x, y):
        ax.plot(x, y, c=color, alpha=alpha)
        if spike_times is not None:
            if is_spikes_in_bin[idx]:
                ax.scatter(x, y, facecolor=spike_color, edgecolor=spike_color,
                           s=markersizes[idx])
        time.sleep(0.1)  # plt.pause(0.0001)
        plt.draw()
    return ax
