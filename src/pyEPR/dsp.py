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

def exp_fit(x, y, a=1., b=1., c=0.)-> dict:
    """
    Fit an exponential function to the data (x, y) using curve_fit.
    Returns a dictionary with the fitted parameters and their covariance.
    """
    from scipy.optimize import curve_fit
    def exp_func(x, a, b, c):
        return a * np.exp(-x/b) + c
    popt, pcov = curve_fit(exp_func, x, y, p0=[a, b, c])
    return {"a": popt[0], "b": popt[1], "c": popt[2], "cov": pcov}
