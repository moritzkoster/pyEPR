import warnings

from qcodes.utils import QCoDeSDeprecationWarning

warnings.filterwarnings("ignore", category=QCoDeSDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"qblox_scheduler\..*")
warnings.simplefilter("ignore", DeprecationWarning)
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import time

import colorama as col
from qblox_scheduler import ClockResource, HardwareAgent, Schedule
from qblox_scheduler.operations import (
    IdlePulse,
    MarkerPulse,
    ResetClockPhase,
    # SetClockFrequency,
    # ShiftClockPhase,
    SoftSquarePulse,
    SquarePulse,
    Trace,
)
from qblox_scheduler.operations.expressions import DType
from qblox_scheduler.operations.loop_domains import arange

from .core import Signal


def run_schedule(schedule: Schedule, agent: HardwareAgent): # unused
    acq = agent.run(schedule)
    return Signal(acq[0].values[0], None)

def run_config(config, agent: HardwareAgent):
    if config["SEQUENCE"] == "ECHO":
        return echo(config, agent)
    elif config["SEQUENCE"] == "CPMG":
        raise ValueError(f"Unknown sequence: {config['SEQUENCE']}")
        return cpmg(config)
    elif config["SEQUENCE"] == "JUST_ACQUIRE":
        return just_acquire(config, agent)
    else:
        raise ValueError(f"Unknown sequence: {config['SEQUENCE']}")

def echo(config: dict, agent: HardwareAgent, metadata: dict | None = None):
    schedule = echo_schedule(config)
    acq = agent.run(schedule)
    from .datamanagement import METADATA
    METADATA.update({"created": time.strftime("%Y-%m-%d %H:%M:%S")})
    return Signal(acq[0].values[0], METADATA)

def echo_schedule(config: dict):
    schedule = Schedule(name="Quantum Rocks")

    # We can add a clock with the frequency for NCO.
    # set the clock frequency
    schedule.add_resource(ClockResource(name="q1.ro", freq=config["TX"]["FREQ"]))
    schedule.add_resource(ClockResource(name="q6.ro", freq=1.6e9))

    sequence_params = echo_timings(config)  # calculate timings and update config

    d_p1_p2 = sequence_params["d_p1_p2"]
    d_tx_gate_on = sequence_params["d_tx_gate_on"]
    d_p2_acq = sequence_params["d_p2_acq"]
    d_txgateoff_rxgateon = sequence_params["d_txgateoff_rxgateon"]
    d_rx_gate_on = sequence_params["d_rx_gate_on"]
    d_p2_rf6on = sequence_params["d_p2_rf6on"]

    print(f"Total time the receiver gate is on: {d_rx_gate_on * 1e9:.1f} ns")
    print(f"Time from amp gate off to rec gate on: {d_txgateoff_rxgateon * 1e9:.1f} ns")
    print(f"Time from pulse 2 to RF6 (delay of RX_GATE_SYNC) on: {d_p2_rf6on * 1e9:.1f} ns")
    print(f"Time from pulse 2 to acquisition: {d_p2_acq * 1e9:.1f} ns")
    print(f"Time from pulse 1 to pulse 2: {d_p1_p2 * 1e9:.1f} ns")
    print(f"Total time the amplifier gate is on: {d_tx_gate_on * 1e9:.1f} ns")

    with (
        schedule.loop(arange(0, config["SCANS"], 1, dtype=DType.NUMBER)),
        schedule.loop(arange(0, 360, 180, dtype=DType.PHASE)) as phase1,
    ):  # exc and rx does [0, 180]

        # reset clock phase at the start
        rst = schedule.add(ResetClockPhase(clock="q1.ro"))
        first_idle = schedule.add(IdlePulse(duration=4e-9), ref_op=rst, rel_time=4e-9)

        amp_gate = schedule.add(
            MarkerPulse(duration=d_tx_gate_on, port="q1:ttl", clock="q1.ro"),
            ref_op=first_idle,
            rel_time=3e-9,
        )  # rel time to get the RF and marker synced rel. to idle

        # Pulse 1
        pulse_1 = schedule.add(
            SoftSquarePulse(
                amplitude=config["TX"]["AMPLITUDE_1"],
                duration=config["TIMING"]["PULSE_DURATION_1"],
                port="q1:res",
                clock="q1.ro",
            ),
            ref_op=first_idle,
            rel_time=config["TIMING"]["TX_GATE_LEAD"],
        )

        # Pulse 2 - identical for now
        pulse_2 = schedule.add(
            SoftSquarePulse(
                amplitude=config["TX"]["AMPLITUDE_2"],
                duration=config["TIMING"]["PULSE_DURATION_2"],
                phase_shift=phase1,
                # phase=phase1,
                port="q1:res",
                clock="q1.ro",
            ),
            ref_op=pulse_1,
            rel_time=d_p1_p2,
        )

        rec_gate = schedule.add(
            SquarePulse(
                amplitude=0.9, duration=d_rx_gate_on, port="q6:res", clock="q6.ro"
            ),
            ref_op=pulse_2,  # time this event relative to excitation pulse named exc
            rel_time=d_p2_rf6on,
        )

        schedule.add(
            Trace(duration=config["ACQ"]["DURATION"], port="q1:res", clock="q1.ro"),
            ref_op=pulse_2,
            rel_time=d_p2_acq,
        )

        schedule.add(
            IdlePulse(duration=config["TIMING"]["LAST_DELAY"])
        )  # wait rep_time after pulse sequence
    from .datamanagement import METADATA
    METADATA.update({"sequence_params": sequence_params})
    return schedule

def echo_timings(config):
    d_p1_p2 = (
        config["TIMING"]["TAU"]
        - (config["TIMING"]["PULSE_DURATION_1"] + config["TIMING"]["PULSE_DURATION_2"])
        / 2
    )  # compensated tau
    d_tx_gate_on = (
        config["TIMING"]["TX_GATE_LEAD"]
        + config["TIMING"]["PULSE_DURATION_1"]
        + d_p1_p2
        + config["TIMING"]["PULSE_DURATION_2"]
        + config["TIMING"]["TX_GATE_TRAIL"]
    )  # total time the gate is on
    d_p2_acq = (
        config["TIMING"]["TAU"]
        + config["TIMING"]["ACQ_WINDOW_SHIFT"]
        - config["TIMING"]["PULSE_DURATION_2"] / 2
        - config["ACQ"]["DURATION"] / 2
    )
    d_txgateoff_rxgateon = (
        d_p2_acq
        - max(0, config["TIMING"]["TX_GATE_TRAIL"])
        - config["TIMING"]["RX_GATE_LEAD"]
    )  # time from amp gate off to rec gate on
    echo_center = config["ACQ"]["DURATION"] / 2
    if d_txgateoff_rxgateon < config["TIMING"]["DEAD"]:
        d_p2_acq = (
            config["TIMING"]["DEAD"]
            + max(0, config["TIMING"]["TX_GATE_TRAIL"])
            + config["TIMING"]["RX_GATE_LEAD"]
        )
        d_txgateoff_rxgateon = config["TIMING"]["DEAD"]
        echo_center = config["TIMING"]["TAU"] + config["TIMING"]["ACQ_WINDOW_SHIFT"] - d_p2_acq
        print(f"""{col.Fore.YELLOW}Warning: Time between amp_off and rec gate on = {d_txgateoff_rxgateon} < DEAD = {config["TIMING"]["DEAD"]}. Is influenced by:
            - TAU = {config["TIMING"]["TAU"]}
            - Pulse Duration 2 = {config["TIMING"]["PULSE_DURATION_2"]}
            - TX_GATE_TRAIL = {config["TIMING"]["TX_GATE_TRAIL"]}
            - RX_GATE_LEAD = {config["TIMING"]["RX_GATE_LEAD"]}
            Adjusting d_p2_acq to {d_p2_acq} to ensure proper dead time. {col.Style.RESET_ALL}""")

    d_rx_gate_on = (
        config["TIMING"]["RX_GATE_LEAD"]
        + config["ACQ"]["DURATION"]
        + config["TIMING"]["RX_GATE_TRAIL"]
    )  # total time the gate is on

    d_p2_rf6on = (
        d_p2_acq - config["TIMING"]["RX_GATE_LEAD"] - config["TIMING"]["RX_GATE_SYNC"]
    )  # time from pulse 2 to rec gate on


    sequence_params = {
        "d_p1_p2": d_p1_p2,
        "d_tx_gate_on": d_tx_gate_on,
        "d_p2_acq": d_p2_acq,
        "d_txgateoff_rxgateon": d_txgateoff_rxgateon,
        "d_rx_gate_on": d_rx_gate_on,
        "d_p2_rf6on": d_p2_rf6on,
        "echo_center": echo_center,
    }

    start = 15e-9
    timestamps = {}
    timestamps["t_tx_gate_on"]    = start
    timestamps["t_p1_start"]      = timestamps["t_tx_gate_on"]    + config["TIMING"]["TX_GATE_LEAD"]
    timestamps["t_p2_start"]      = timestamps["t_p1_start"]      + config["TIMING"]["PULSE_DURATION_1"] + d_p1_p2
    timestamps["t_tx_gate_off"]   = timestamps["t_tx_gate_on"]    + d_tx_gate_on
    timestamps["t_rx_gate_on"]    = timestamps["t_tx_gate_off"]   + d_txgateoff_rxgateon
    timestamps["t_acq_start"]     = timestamps["t_p2_start"]      + config["TIMING"]["PULSE_DURATION_2"] + d_p2_acq
    timestamps["t_acq_end"]       = timestamps["t_acq_start"]     + config["ACQ"]["DURATION"]

    sequence_params.update(timestamps)
    return sequence_params

def just_acquire(config, agent: HardwareAgent):
    schedule = Schedule(name="Just Acquire")
    schedule.add_resource(ClockResource(name="q1.ro", freq=config["TX"]["FREQ"]))
    schedule.add_resource(ClockResource(name="q6.ro", freq=1.6e9))

    with schedule.loop(arange(0, config["SCANS"], 1, dtype=DType.NUMBER)):
        rst = schedule.add(ResetClockPhase(clock="q1.ro"))
        # long first idle to ensure RX_GATE_SYNC has enough time before
        first_idle = schedule.add(IdlePulse(duration=500e-9), ref_op=rst, rel_time=4e-9)

        rec_gate = schedule.add(
            SquarePulse(
                amplitude=0.9,
                duration= config["TIMING"]["RX_GATE_LEAD"]
                        + config["ACQ"]["DURATION"]
                        + config["TIMING"]["RX_GATE_TRAIL"],
                port="q6:res",
                clock="q6.ro",
            ),
            ref_op=first_idle ,
            rel_time= - config["TIMING"]["RX_GATE_SYNC"],
        )

        schedule.add(
            Trace(duration=config["ACQ"]["DURATION"], port="q1:res", clock="q1.ro"),
            ref_op=first_idle,
            rel_time= config["TIMING"]["RX_GATE_LEAD"]
        )

        schedule.add(IdlePulse(duration=config["TIMING"]["LAST_DELAY"]))

    acq = agent.run(schedule)
    from .datamanagement import METADATA
    METADATA.update({"created": time.strftime("%Y-%m-%d %H:%M:%S")})
    return Signal(acq[0].values[0], METADATA)

def cpmg(config):
    raise NotImplementedError("CPMG sequence is not implemented yet.")
