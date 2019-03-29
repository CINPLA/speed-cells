import numpy as np
import pytest
import quantities as pq
import neo
from tools import random_walk
from speed_cells.speed import speed_correlation


def nonstationary_poisson(times, rate):
    """
    Non-stationary Poisson process
    Implemented by Alexander Stasik (a.j.stasik@fys.uio.no)
    Parameters
    ----------
    times : array
        timepoints corresponding to rate.
    rate : array
        rates as function of time.
    Returns
    -------
    events : array
        time points from a poisson process which rate varies according to rate.
    """
    n_exp = rate.max() * (times.max()-times.min())
    t_events = np.sort(
        np.random.uniform(
            times.min(), times.max(), np.random.poisson(n_exp)))
    mask = np.digitize(t_events, times)
    ratio = rate[mask] / rate.max()
    mask = np.random.uniform(0., 1., len(ratio)) < ratio
    return t_events[mask]


def stationary_poisson(t_start, t_stop, rate):
    """
    Stationary Poisson process
    Parameters
    ----------
    t_start : float
        Start time of the process (lower bound).
    t_stop : float
        Stop time of the process (upper bound).
    rate : float
        rate of the Poisson process
    Returns
    -------
    events : array
        time points from a Poisson process with rate rate.
    """
    n_exp = rate * (t_stop - t_start)
    return np.sort(
        np.random.uniform(
            t_start, t_stop, np.random.poisson(n_exp)))


def test_speed_random():
    box_size = [1., 1.]
    rate = 5.
    n_step=10**4
    step_size=.01

    t = np.linspace(0, n_step * step_size / 1.5, n_step)
    dt = t[1]
    trajectory = random_walk(box_size, step_size, n_step)
    x, y = trajectory.T
    st = stationary_poisson(
        rate=rate, t_start=0, t_stop=t[-1])
    s = np.sqrt(x*x + y*y)
    speed = np.diff(s) / dt
    corr, inst_speed, inst_rate, times = speed_correlation(speed, t, st)
    assert  np.abs(corr) < 0.05
    assert inst_rate.mean().round() == rate


def test_speed_nonlinear():
    sampling_rate = 50 #hz
    f, a = .1, 1
    t = np.arange(0, 100, 1 / sampling_rate)
    speed = a * (np.sin(2*np.pi*f*t) + 1)
    spikes = nonstationary_poisson(t, speed)
    corr, inst_speed, inst_rate, times = speed_correlation(
        speed, t, spikes, stddev=.4, filter_speed=True, percentile=95)
    assert  round(corr, 2) > 0.5


def test_speed_linear():
    sampling_rate = 50 #hz
    t = np.arange(0, 100, 1 / sampling_rate)
    speed = np.linspace(0, 5, len(t))
    spikes = nonstationary_poisson(t, speed)
    corr, inst_speed, inst_rate, times = speed_correlation(
        speed, t, spikes, stddev=.4, filter_speed=True, percentile=95)
    assert  round(corr, 2) > 0.5
