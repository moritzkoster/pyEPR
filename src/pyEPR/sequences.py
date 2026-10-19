import warnings

from qcodes.utils import QCoDeSDeprecationWarning

warnings.filterwarnings("ignore", category=QCoDeSDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"qblox_scheduler\..*")
warnings.simplefilter("ignore", DeprecationWarning)
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import colorama as col
import numpy as np
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


def echo(config: dict, agent: HardwareAgent):
    schedule = Schedule(name="Quantum Rocks")

    # We can add a clock with the frequency for NCO.
    schedule.add_resource(
        ClockResource(name="q1.ro", freq=config["TX"]["FREQ"])
    )  # set the clock frequency
    schedule.add_resource(ClockResource(name="q6.ro", freq=1.6e9))

    config = echo_timings(config)  # calculate timings and update config

    d_p1_p2 = config["TIMING"]["d_p1_p2"]
    d_tx_gate_on = config["TIMING"]["d_tx_gate_on"]
    d_p2_acq = config["TIMING"]["d_p2_acq"]
    d_txgateoff_rxgateon = config["TIMING"]["d_txgateoff_rxgateon"]
    d_rx_gate_on = config["TIMING"]["d_rx_gate_on"]
    d_p2_rf6on = config["TIMING"]["d_p2_rf6on"]

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
        rst = schedule.add(
            ResetClockPhase(clock="q1.ro")
        )  # reset clock phase at the start
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

    # Run the schedule and acquire the data
    acq = agent.run(schedule)
    return np.array(acq[0].values[0])

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
    if d_txgateoff_rxgateon < config["TIMING"]["DEAD"]:
        d_p2_acq = (
            config["TIMING"]["DEAD"]
            + max(0, config["TIMING"]["TX_GATE_TRAIL"])
            + config["TIMING"]["RX_GATE_LEAD"]
        )
        d_txgateoff_rxgateon = config["TIMING"]["DEAD"]
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

    config["TIMING"]["d_p1_p2"]              = d_p1_p2
    config["TIMING"]["d_tx_gate_on"]         = d_tx_gate_on
    config["TIMING"]["d_p2_acq"]             = d_p2_acq
    config["TIMING"]["d_txgateoff_rxgateon"] = d_txgateoff_rxgateon
    config["TIMING"]["d_rx_gate_on"]         = d_rx_gate_on
    config["TIMING"]["d_p2_rf6on"]           = d_p2_rf6on

    start = 15e-9
    config["timestamps"] = {}
    config["timestamps"]["t_tx_gate_on"]    = start
    config["timestamps"]["t_p1_start"]      = config["timestamps"]["t_tx_gate_on"]    + config["TIMING"]["TX_GATE_LEAD"]
    config["timestamps"]["t_p2_start"]      = config["timestamps"]["t_p1_start"]      + config["TIMING"]["PULSE_DURATION_1"] + d_p1_p2
    config["timestamps"]["t_tx_gate_off"]   = config["timestamps"]["t_tx_gate_on"]    + d_tx_gate_on
    config["timestamps"]["t_rx_gate_on"]    = config["timestamps"]["t_tx_gate_off"]   + d_txgateoff_rxgateon
    config["timestamps"]["t_acq_start"]     = config["timestamps"]["t_p2_start"]      + config["TIMING"]["PULSE_DURATION_2"] + d_p2_acq
    config["timestamps"]["t_acq_end"]       = config["timestamps"]["t_acq_start"]     + config["ACQ"]["DURATION"]

    return config

def cpmg(config):
    return config
