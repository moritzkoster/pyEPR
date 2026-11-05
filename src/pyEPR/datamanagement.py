import json
import os
import time

import h5py
import numpy as np
import yaml

from .core import Signal

DATA_FOLDER = "data"
METADATA = {
    "created": {},
    "config": {},
    "misc": {},
    "notes": ""
}

def create_folder(name=None, overwrite=False):
    global DATA_FOLDER
    import os

    if not name:
        i = 0
        while True:
            name = f"data_{i:03d}"
            if not os.path.exists(name):
                break
            i += 1

    if not overwrite and os.path.exists(name):
        raise FileExistsError(
            f"Folder {name} already exists. Use overwrite=True to overwrite."
        )
    DATA_FOLDER = name
    os.makedirs(name, exist_ok=True)

def select_folder(name):
    global DATA_FOLDER
    if not os.path.exists(name):
        raise FileNotFoundError(f"Folder {name} does not exist.")
    DATA_FOLDER = name

# --- saving ---
def save(filename, signal:Signal, overwrite=False):
    # save_v1(*args, **kwargs)
    save_v2(filename, signal, overwrite)  # default to v2 for new files

 # v1: compact binary format with metadata in YAML.
def save_v1(filename, data, metadata, overwrite=False):
    filepath = os.path.join(DATA_FOLDER, filename)
    if not overwrite and h5py.is_hdf5(filepath):
        raise FileExistsError(f"File {filepath} already exists. Use overwrite=True to overwrite.")
    with h5py.File(filepath, "w") as f:
        f.attrs["version"] = "VERSION_1"  # versioning for future compatibility
        f.attrs["config"] = "\n" + json.dumps(
            metadata, indent=2
        )  # readable key/value metadata
        f.create_dataset("signal", data=data)  # complex array, binary & compact

# v2: using Signal object
def save_v2(filename, signal:Signal, overwrite=False):
    filepath = os.path.join(DATA_FOLDER, filename)
    if not overwrite and os.path.exists(filepath):
        if input(f"File {filepath} already exists. Overwrite? (y/n): ") not in ["y", "Y", ""]:
            filepath = increase_filename(filepath)
            print(f"Saving to {filepath} instead.")
        else:
            print(f"Overwriting {filepath}.")

    d = {"metadata": to_native(signal.metadata), "data": signal.data.tobytes().hex()}

    with open(filepath, "w") as f:
        f.write("# VERSION_2\n\n")
        f.write(
            "# This file contains the signal data and metadata in a compact format.\n"
            "# The metadata is stored in YAML format for easy reading and parsing.\n"
            "# Any params starting in lowercase or __ are calculated from other params\n"
            "# Use 'yaml.safe_load('file.epr')' to get the metadata and data as a dictionary.\n"
            "# The data is stored as a hex-encoded string to save space.\n"
            "# The data can be converted back to a np.ndarray using 'np.frombuffer(bytes.fromhex(data_hex), dtype=np.complex128)'.\n\n"
        )
        yaml.dump(d, f, indent=2, sort_keys=False)

# --- loading ---
def load(filename) -> Signal:
    if os.path.splitext(filename)[1] == ".hdf5":
        with h5py.File(filename, "r") as f:
            version = f.attrs.get("version", None)
        if version == "VERSION_1":
            return load_v1(filename)
        raise ValueError(f"Unsupported file version: {version}")
    if os.path.splitext(filename)[1] == ".epr":
        with open(filename, "r") as f:
            version = f.readline().strip()
        if version == "# VERSION_2":
            return load_v2(filename)
        if version == "# VERSION_3":
            return load_v2(filename)  # v3 is compatible with v2 loading
        raise ValueError(f"Unsupported file version: {version}")

    else:
        raise ValueError(
            f"Unsupported file type: {os.path.splitext(filename)[1] == '.hdf5'}"
        )

def load_v1(filename):
    with h5py.File(filename, "r") as f:
        data = np.array(f["signal"])
        metadata = json.loads(str(f.attrs["config"]).strip())
    return Signal(data, metadata)
    # return data, metadata # was this fore some time


def load_v2(filename):
    with open(filename, "r") as f:
        d = yaml.safe_load(f)
        metadata = d["metadata"]
        data_hex = d["data"]
        data = np.frombuffer(bytes.fromhex(data_hex), dtype=np.complex128)
    if "misc" not in metadata:
        metadata["misc"] = {}
    metadata["misc"]["filename"] = filename.split(os.sep)[-1]
    return Signal(data, metadata)

def load_folder(folder, name="*.epr") -> list[Signal]: # also can use *.hdf5
    import glob

    data_files = glob.glob(os.path.join(folder, name))

    data_list = []
    for file in data_files:
        data_list.append(load(os.path.join(file)))

    return data_list

# --- utility functions ---
def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):  # np.float64, np.int32, np.bool_, ...
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def increase_filename(path):
    base, ext = os.path.splitext(path)
    i = 1
    new_path = f"{base}_{i}{ext}"
    while os.path.exists(new_path):
        i += 1
        new_path = f"{base}_{i}{ext}"
    return new_path
