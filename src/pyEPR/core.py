import numpy as np

from .dsp import smooth_bandpass


class Signal:
    def __init__(self, data, metadata=None):
        self.data = data
        self.t = np.arange(len(data)) * 1e-9  # Assuming 1 ns sampling interval
        self.metadata = {}
        self.metadata.update(metadata or {})

        self.__spec()

    def __spec(self):
        self.spec = np.fft.fftshift(np.fft.fft(self.data))
        self.f = np.fft.fftshift(np.fft.fftfreq(len(self.spec), d=1e-9))  # Assuming 1 ns sampling interval

    def remove_dc(self, window=1/6):
        """
        Remove DC offset based on the last window_size samples.
        """
        window_size = int(len(self.data) * window)
        self.data = self.data - np.mean(self.data[-window_size:])  # Remove DC offset based on the last window_size samples
        self.__spec()

    def apply_bandpass(self, center=0., bandwidth=200e6, transition=10e6):
        mask = smooth_bandpass(self.f, center, bandwidth, transition)
        self.spec = self.spec * mask
        self.data = np.fft.ifft(np.fft.ifftshift(self.spec))
        self.__spec()

    def slice(self, start=None, end=None, domain="t", inplace=True):
        if domain == "time" or domain == "t":
            self.slice_time(start, end, inplace=inplace)
        elif domain == "frequency" or domain == "f" or domain == "freq":
            self.slice_frequency(start, end, inplace=inplace)
        else:
            raise ValueError("Domain must be 't' or 'f'.")

    def slice_time(self, start, end, inplace=True):
        if not inplace:
            raise NotImplementedError("Non-inplace slicing is not implemented for time domain.")
        if start is None:
            start = self.t[0]
        if end is None:
            end = self.t[-1]
        self.data = self.data[(self.t >= start) & (self.t <= end)]
        self.t = self.t[(self.t >= start) & (self.t <= end)]
        self.__spec()

    def slice_frequency(self, start, end, inplace=True):
        if not inplace:
            raise NotImplementedError("Non-inplace slicing is not implemented for frequency domain.")
        if start is None:
            start = self.t[0]
        if end is None:
            end = self.t[-1]
        self.spec = self.spec[(self.f >= start) & (self.f <= end)]
        self.f = self.f[(self.f >= start) & (self.f <= end)]
        self.__spec()

    def default_preprocessing(self):
        self.remove_dc()
        self.apply_bandpass(center=0, bandwidth=200e6, transition=10e6)

    def add_metadata(self, meta):
        self.metadata.update(meta)

    def __repr__(self):
        return f"Signal({self.metadata})"
