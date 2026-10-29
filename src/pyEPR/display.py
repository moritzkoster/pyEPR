import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

import __main__

from .core import Signal


def autosave():
    if "nosave" not in sys.argv:
        if sys.platform == "win32":
            FILENAME = __main__.__file__.split("\\")[-1].split(".")[0]
        else:
            FILENAME = __main__.__file__.split("/")[-1].split(".")[0]

        # PATH_HISTORY = "../../figures/history"
        # PATH_CURRENT = "../../figures/current"
        PATH_HISTORY = "./figures/history"
        PATH_CURRENT = "./figures/current"

        if not os.path.exists(PATH_HISTORY):
            os.makedirs(PATH_HISTORY)
        if not os.path.exists(PATH_CURRENT):
            os.makedirs(PATH_CURRENT)

        i = 0

        while True:
            if os.path.isfile(f"{PATH_HISTORY}/{FILENAME}_{i}.png"):
                i += 1
            else:
                break

        fignums = plt.get_fignums()
        if len(fignums) == 0:
            print("No figures to save.")
            return
        shutil.copy(f"{__main__.__file__}", f"{PATH_HISTORY}/{FILENAME}_{i}.py")

        with open(f"{PATH_HISTORY}/{FILENAME}_{i}.py", "a") as f:
            f.write(f"\n\n# {sys.argv}\n")

        if len(fignums) == 1:
            fig = plt.gcf()
            fig.savefig(f"{PATH_HISTORY}/{FILENAME}_{i}.pgf", dpi=300)
            fig.savefig(f"{PATH_HISTORY}/{FILENAME}_{i}.png", dpi=300)
            fig.savefig(f"{PATH_CURRENT}/{FILENAME}.pgf", dpi=300)
            fig.savefig(f"{PATH_CURRENT}/{FILENAME}.png", dpi=300)
            return
        for fi in fignums:
            fig = plt.figure(fi)
            fig.savefig(f"{PATH_HISTORY}/{FILENAME}_{i}{chr(65 + fi)}.pgf", dpi=300)
            fig.savefig(f"{PATH_HISTORY}/{FILENAME}_{i}{chr(65 + fi)}.png", dpi=300)
            fig.savefig(f"{PATH_CURRENT}/{FILENAME}_{chr(65 + fi)}.pgf", dpi=300)
            fig.savefig(f"{PATH_CURRENT}/{FILENAME}_{chr(65 + fi)}.png", dpi=300)

def plot_signal(signal:Signal, axs=None, show=True, components="RIM", bandpass=None):

    if components == "all":
        components = "RIM"

    if axs is None:
        fig = plt.figure(2, figsize=(8, 6))
        axs = fig.subplots(1, 2)
    if "M" in components:
        axs[0].plot(signal.t * 1e9, np.abs(signal.data), "k", label="Mag.")
        axs[1].plot(signal.f * 1e-6, np.abs(signal.spec), "k", label="Mag.")
    if "R" in components:
        axs[0].plot(signal.t * 1e9, signal.data.real, c="teal", ls="-", label="Real")
        axs[1].plot(signal.f * 1e-6, signal.spec.real, c="teal", ls="-", label="Real")
    if "I" in components:
        axs[0].plot(signal.t * 1e9, signal.data.imag, c="turquoise", ls="--", label="Imag.")
        axs[1].plot(signal.f * 1e-6, signal.spec.imag, c="turquoise", ls="--", label="Imag.")

    axs[0].set_xlabel("Time (ns)")
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel("Offset Frequency (MHz)")
    axs[1].legend(loc="upper left")

    axs[1].set_xlim([-50, 50])

    if show:
        plt.show()

def plot_sequence(config, **kwargs):
    if config["SEQUENCE"] == "ECHO":
        plot_echo_sequence(config, **kwargs)
    elif config["SEQUENCE"] == "SR_ECHO":
        plot_sr_echo_sequence(config, **kwargs)
    else:
        raise ValueError(f"Unsupported sequence: {config['SEQUENCE']}")

def plot_echo_sequence(config, show=True):
    from .sequences import echo_timings
    sequence_params = echo_timings(config)

    fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    t = np.arange(0, sequence_params["t_rx_gate_on"] + sequence_params["d_rx_gate_on"] * 1.2 , 1e-9)  # time vector in ns
    # start = 15e-9
    # Plot TX gate
    tx_gate = np.zeros_like(t)
    tx_gate[(t >= sequence_params["t_tx_gate_on"]) & (t <= (sequence_params["t_tx_gate_on"] + sequence_params["d_tx_gate_on"]))] = 1
    axs[0].plot(t, tx_gate, label="TX Gate", color="k")

    tx_gate_noise = np.full(np.shape(t), np.nan)
    t1 = sequence_params["t_tx_gate_on"] + sequence_params["d_tx_gate_on"]
    t2 = t1 + 610e-9
    tx_gate_noise[(t >= t1 ) & (t < t2)] = np.exp(-(t[(t >= t1 ) & (t < t2)]-t1)/(t2-t1)*4)
    axs[0].plot(t, tx_gate_noise, label="Gate Noise", color="k", ls="--")

    # Plot pulse sequence
    pulse_seq = np.zeros_like(t)
    pulse_seq[(t >= sequence_params["t_p1_start"]) & (t < sequence_params["t_p1_start"] +config["TIMING"]["PULSE_DURATION_1"])] = config["TX"]["AMPLITUDE_1"]
    pulse_seq[(t >= sequence_params["t_p2_start"]) & (t < sequence_params["t_p2_start"] +config["TIMING"]["PULSE_DURATION_2"])] = config["TX"]["AMPLITUDE_2"]
    axs[0].plot(t, pulse_seq, label="Pulse Sequence", color="orange")

    pulse_seq_noise = np.full(np.shape(t), np.nan)
    t1 = sequence_params["t_p2_start"] + config["TIMING"]["PULSE_DURATION_2"]
    t2 = t1 + 730e-9
    pulse_seq_noise[(t >= t1 ) & (t < t2)] = np.exp(-(t[(t >= t1 ) & (t < t2)]-t1)/(t2-t1)*4) *config["TX"]["AMPLITUDE_2"]
    axs[0].plot(t, pulse_seq_noise, label="Sequence Noise", color="orange", ls="--")

    rx_gate = np.zeros_like(t)
    rx_gate[(t >= sequence_params["t_rx_gate_on"]) & (t <= (sequence_params["t_rx_gate_on"] + sequence_params["d_rx_gate_on"]))] = 1
    axs[1].plot(t, rx_gate, label="RX Gate", color="green")
    rf6_gate = np.zeros_like(t)
    rf6_gate[(t >= (sequence_params["t_rx_gate_on"] - config["TIMING"]["RX_GATE_SYNC"])) & (t <= (sequence_params["t_rx_gate_on"] + sequence_params["d_rx_gate_on"] - config["TIMING"]["RX_GATE_SYNC"] ))] = 1
    axs[1].plot(t, rf6_gate, label="RX Gate", color="green", ls="--")

    rx = np.zeros_like(t)
    rx[(t >= sequence_params["t_acq_start"]) & (t <= (sequence_params["t_acq_start"] + config["ACQ"]["DURATION"]))] = 1
    axs[2].plot(t, rx, label="RX", color="red")



    axs[0].set_ylabel("Amplitude")
    if show:
        plt.show()

def plot_sr_echo_sequence(config,  show=True):
    from .sequences import sr_echo_schedule
    from .setup import AGENT

    schedule = sr_echo_schedule(config)
    comp_schedule = AGENT.compile(schedule)
    df = comp_schedule.timing_table.data
    print(df.columns)
