import numpy as np


def smooth_bandpass(f, center=0, bandwidth=200e6, transition=10e6):
    """
    f          : frequency axis (Hz)
    center     : center frequency (Hz)
    bandwidth  : passband width (Hz)
    transition : edge smoothing width (Hz)

    returns mask between 0 and 1
    """

    x = np.abs(f - center)
    return 0.5 * (np.tanh((bandwidth / 2 - x) / transition) + 1)

def gaussian(f, center, bandwidth):
    """
    f          : frequency axis (Hz)
    cutoff     : center (Hz)
    bandwidth  : passband width (1 sigma ) (Hz)

    returns mask between 0 and 1
    """
    return np.exp(-0.5 * ((f - center) / (bandwidth / 2)) ** 2)
