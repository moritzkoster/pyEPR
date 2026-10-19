
import matplotlib.pyplot as plt
import numpy as np


def plot_signal(signal, axs=None, show=True, components="RIM"):
    """
    Show the acquired data in a plot.

    Args:
        signal: The acquired data to be displayed.
    """

    sig = signal

    sig = sig - np.mean(
        sig[-150:]
    )  # fix the DC offset (last 150 samples are the tail of the echo, should be zero)

    #     f_demod = config["TX"]["FREQ"] - QRC.out0_in0_lo_freq() #difference from the clock?
    #     print(f"Demodulation frequency = {f_demod*1e-6:.2f} MHz")
    #     phi = 97
    #     phi = 0
    #
    t = np.linspace(0, sig.shape[0]*1e-9, sig.shape[0]) #time vector
    #
    #     ##2. Phase-Shifted Digital Down-Conversion (DDC)
    #     phi_rad = np.radians(phi)
    #     sig = sig * np.exp(-1j * (2 * np.pi * f_demod * t + phi_rad))

    ##Frequency Filter
    def smooth_bandpass(freqs, center=0, bandwidth=200e6, transition=10e6):
        """
        freqs      : frequency axis (Hz)
        center     : center frequency (Hz)
        bandwidth  : passband width (Hz)
        transition : edge smoothing width (Hz)

        returns mask between 0 and 1
        """

        x = np.abs(freqs - center)

        return 0.5 * (np.tanh((bandwidth / 2 - x) / transition) + 1)

    # Bandpass filter
    spec = np.fft.fftshift(np.fft.fft(sig))

    sw = 1e9
    f = np.linspace(-sw / 2, sw / 2, spec.shape[0])
    mask = smooth_bandpass(f, 0, 200e6, 10e6)
    spec *= mask

    # IFT to recover the echo
    sig = np.fft.ifft(np.fft.ifftshift(spec))
    if components == "all":
        components = "RIM"

    if axs is None:
        fig = plt.figure(2, figsize=(8, 6))
        axs = fig.subplots(1, 2)
    if "R" in components:
        axs[0].plot(t * 1e9, sig.real, "r", label="Real")
    if "I" in components:
        axs[0].plot(t * 1e9, sig.imag, "g", label="Imag.")
    if "M" in components:
        axs[0].plot(t * 1e9, np.abs(sig), "b", label="Mag.")
    axs[0].set_xlabel("Time (ns)")
    axs[0].legend(loc="upper right")

    sw = 1e9
    f = np.linspace(-sw / 2, sw / 2, sig.shape[0] * 8)
    spec = np.fft.fftshift(np.fft.fft(sig, n=sig.shape[0] * 8))
    if "R" in components:
        axs[1].plot(f * 1e-6, spec.real, "r", label="Real")
    if "I" in components:
        axs[1].plot(f * 1e-6, spec.imag, "g", label="Imag.")
    if "M" in components:
        axs[1].plot(f * 1e-6, np.abs(spec), "b", label="Mag.")

    # plt.plot(f*1e-6, abs(spec), 'b' )
    axs[1].set_xlabel("Offset Frequency (MHz)")
    axs[1].legend(loc="upper left")

    axs[1].set_xlim([-50, 50])

    # print('SNR = %.2f' % (proc.snr(abs(spec))))

    plt.savefig("result.png", format="png")
    if show:
        plt.show()

def plot_sequence(config, **kwargs):
    if config["SEQUENCE"] == "ECHO":
        plot_echo_sequence(config, **kwargs)
    else:
        raise ValueError(f"Unsupported sequence: {config['SEQUENCE']}")

def plot_echo_sequence(config, show=True):
    from .sequences import echo_timings
    echo_timings(config)

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(0, config["timestamps"]["t_rx_gate_on"] + config["TIMING"]["d_rx_gate_on"] * 1.2 , 1e-9)  # time vector in ns
    # start = 15e-9
    # Plot TX gate
    tx_gate = np.zeros_like(t)
    tx_gate[(t >= config["timestamps"]["t_tx_gate_on"]) & (t <= (config["timestamps"]["t_tx_gate_on"] + config["TIMING"]["d_tx_gate_on"]))] = 1
    axs[0].plot(t, tx_gate, label="TX Gate", color="k")

    tx_gate_noise = np.full(np.shape(t), np.nan)
    t1 = config["timestamps"]["t_tx_gate_on"] + config["TIMING"]["d_tx_gate_on"]
    t2 = t1 + 610e-9
    tx_gate_noise[(t >= t1 ) & (t < t2)] = np.exp(-(t[(t >= t1 ) & (t < t2)]-t1)/(t2-t1)*4)
    axs[0].plot(t, tx_gate_noise, label="Gate Noise", color="k", ls="--")

    # Plot pulse sequence
    pulse_seq = np.zeros_like(t)
    pulse_seq[(t >= config["timestamps"]["t_p1_start"]) & (t < config["timestamps"]["t_p1_start"] +config["TIMING"]["PULSE_DURATION_1"])] = config["TX"]["AMPLITUDE_1"]
    pulse_seq[(t >= config["timestamps"]["t_p2_start"]) & (t < config["timestamps"]["t_p2_start"] +config["TIMING"]["PULSE_DURATION_2"])] = config["TX"]["AMPLITUDE_2"]
    axs[0].plot(t, pulse_seq, label="Pulse Sequence", color="orange")

    pulse_seq_noise = np.full(np.shape(t), np.nan)
    t1 = config["timestamps"]["t_p2_start"] + config["TIMING"]["PULSE_DURATION_2"]
    t2 = t1 + 730e-9
    pulse_seq_noise[(t >= t1 ) & (t < t2)] = np.exp(-(t[(t >= t1 ) & (t < t2)]-t1)/(t2-t1)*4) *config["TX"]["AMPLITUDE_2"]
    axs[0].plot(t, pulse_seq_noise, label="Sequence Noise", color="orange", ls="--")

    rx_gate = np.zeros_like(t)
    rx_gate[(t >= config["timestamps"]["t_rx_gate_on"]) & (t <= (config["timestamps"]["t_rx_gate_on"] + config["TIMING"]["d_rx_gate_on"]))] = 1
    axs[1].plot(t, rx_gate, label="RX Gate", color="green")
    rf6_gate = np.zeros_like(t)
    rf6_gate[(t >= (config["timestamps"]["t_rx_gate_on"] - config["TIMING"]["RX_GATE_SYNC"])) & (t <= (config["timestamps"]["t_rx_gate_on"] + config["TIMING"]["d_rx_gate_on"] - config["TIMING"]["RX_GATE_SYNC"] ))] = 1
    axs[1].plot(t, rf6_gate, label="RX Gate", color="green", ls="--")

    rx = np.zeros_like(t)
    rx[(t >= config["timestamps"]["t_acq_start"]) & (t <= (config["timestamps"]["t_acq_start"] + config["ACQ"]["DURATION"]))] = 1
    axs[2].plot(t, rx, label="RX", color="red")



    axs[0].set_ylabel("Amplitude")
    if show:
        plt.show()
