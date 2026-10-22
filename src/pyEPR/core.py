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

    def remove_dc(self):
        self.data = self.data - np.mean(self.data[-150:])  # Remove DC offset based on the last 150 samples
        self.__spec()

    def apply_bandpass(self, center=0, bandwidth=200e6, transition=10e6):
        mask = smooth_bandpass(self.f, center, bandwidth, transition)
        self.spec = self.spec * mask
        self.data = np.fft.ifft(np.fft.ifftshift(self.spec))
        self.__spec()

    def default_preprocessing(self):
        self.remove_dc()
        self.apply_bandpass(center=0, bandwidth=200e6, transition=10e6)

    def add_metadata(self, meta):
        self.metadata.update(meta)

    def __repr__(self):
        return f"Signal({self.metadata})"
