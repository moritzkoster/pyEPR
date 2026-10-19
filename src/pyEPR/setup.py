import os
import shutil

import yaml
from qblox_scheduler import HardwareAgent


def load_config(file_path):
    from .datamanagement import DATA_FOLDER
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    shutil.copyfile(file_path, os.path.join(DATA_FOLDER, "config.yaml"))  # copy config file to data folder
    print(f'Qblox Exp. Time = {2*config["SCANS"]*(config["TIMING"]["LAST_DELAY"]+config["TIMING"]["TAU"]+config["ACQ"]["DURATION"]) :.4f} s')
    print(f'Duty Cycle = {(config["TIMING"]["PULSE_DURATION_1"] + config["TIMING"]["PULSE_DURATION_2"])/ config["TIMING"]["LAST_DELAY"] *100:.2f} %' )
    return config

def initialize_hardware(config, hardware_config_file="./dependencies/hardware_configuration_file.json"):
    agent = HardwareAgent(hardware_configuration=hardware_config_file)
    agent.connect_clusters()

    #extracting the cluster
    cluster=agent.get_clusters()["cluster0"]
    # cluster.reset() #reset Qblox

    #extracting the module by number - 16??
    QRC=cluster.module2

    #we change the input attenuation with Qblox instruments command.
    QRC.in0_att(config["RX"]["ATTENUATION"])
    config.update({"MISC": {"__QRC.out0_in0_lo_freq": QRC.out0_in0_lo_freq()}})  # update config metadata
    return agent, QRC
