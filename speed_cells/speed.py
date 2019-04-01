import numpy as np
import quantities as pq
import neo
import elephant as ep
from scipy.interpolate import interp1d


def _filter_speed(speed, times, min_speed=0.0, max_speed=100):
    """
    Filters speed above percentile and belov min_speed.
    Parameters:
    -----------
    speed : array
        assumed m/s
    times : array
        assumed s
    min_speed : float
        lower cutoff
    max_speed : float
        upper cutoff
    Returns
    -------
    out : correlation, (pearson, inst_speed)
    References
    ----------
    [1]:
    """
    min_speed = pq.Quantity(min_speed, 'm/s')
    max_speed = pq.Quantity(max_speed, 'm/s')
    mask = np.logical_and(speed < max_speed, speed > min_speed)
    return speed[mask], times[:speed.size][mask]


def speed_correlation(
    speed, times, spike_times, stddev=0.25, filter_speed=True,
    min_speed=0.0, max_speed=100, return_data=False):
    """
    Correlates instantaneous spike rate and rat velocity, using a method
    described in [1]
    Parameters:
    -----------
    speed : array
        assumed m/s
    times : array
        assumed s
    spike_times : array or neo.SpikeTrain
        assumed s
    stddev : float
        standard deviation of Gaussian kernel generating instantaneous spike rate
    filter_speed : tuple
        lower bound and upper percentile; see `filter_speed_percentile`
    Returns
    -------
    out : correlation, (pearson, inst_speed)
    References
    ----------
    [1]: Kropff, E., Carmichael, J. E., Moser, M. B., & Moser, E. I. (2015).
    Speed cells in the medial entorhinal cortex. Nature, 523(7561), 419.
    """
    times = pq.Quantity(times, 's')
    speed = pq.Quantity(speed, 'm/s')
    stddev = pq.Quantity(stddev, 's')
    if filter_speed:
        speed, times = _filter_speed(
            speed, times, min_speed, max_speed)
        spike_times = spike_times[spike_times < times[-1]]

    if not isinstance(spike_times, neo.SpikeTrain):
        spike_times = neo.SpikeTrain(
            spike_times, units='s', t_start=times[0], t_stop=times[-1])

    binsize = np.mean(np.diff(times)).rescale('s')

    rate = ep.statistics.instantaneous_rate(
        spike_times, binsize,
        kernel=ep.kernels.GaussianKernel(sigma=stddev))

    # interpolate original speed
    interp_speed = interp1d(
        times, speed,  bounds_error=False, fill_value=(speed[0], speed[-1]))

    # retain speed at rate timepoints
    mask = np.logical_and(rate.times <= times[-1], rate.times >= times[0])
    times = rate.times[mask].rescale('s').magnitude
    inst_speed = interp_speed(times)
    rate = rate.magnitude[mask, 0]
    correlation = np.corrcoef(inst_speed, rate)[1, 0]

    if return_data:
        return correlation, inst_speed, rate, times
    else:
        return correlation
