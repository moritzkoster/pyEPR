import json

import h5py
import numpy as np
import yaml
from h5py._hl.compat import os

DATA_FOLDER = "data"

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
        raise FileExistsError(f"Folder {name} already exists. Use overwrite=True to overwrite.")
    DATA_FOLDER = name
    os.makedirs(name, exist_ok=True)

# --- saving ---
def save(filename, data, metadata, overwrite=False):
    # save_v1(filename, data, metadata, overwrite=overwrite)
    save_v2(filename, data, metadata, overwrite=overwrite)

def save_v1(filename, data, metadata, overwrite=False): # v1: compact binary format with metadata in YAML
    filepath = os.path.join(DATA_FOLDER, filename)
    if not overwrite and h5py.is_hdf5(filepath):
        raise FileExistsError(f"File {filepath} already exists. Use overwrite=True to overwrite.")
    with h5py.File(filepath, "w") as f:
        f.attrs["version"] = "VERSION_1"                      # versioning for future compatibility
        f.attrs["config"] = "\n" + json.dumps(metadata, indent=2)        # readable key/value metadata
        f.create_dataset("signal", data=data)          # complex array, binary & compact

def save_v2(filename, data, metadata, overwrite=False): # v2: compact binary format with metadata in YAML, data stored as hex string
    filepath = os.path.join(DATA_FOLDER, filename)
    if not overwrite and os.path.exists(filepath):
        raise FileExistsError(f"File {filepath} already exists. Use overwrite=True to overwrite.")
    d = {
        "metadata": metadata,
        "data": data.tobytes().hex()
    }
    with open (filepath, "w") as f:
        f.write("# VERSION_2\n\n")
        f.write("# This file contains the signal data and metadata in a compact format.\n")
        f.write("# The metadata is stored in YAML format for easy reading and parsing.\n")
        f.write("# Any params starting in lowercase or __ are calculated from other params\n")
        f.write("# Use 'yaml.safe_load('file.epr')' to get the metadata and data as a dictionary.\n")
        f.write("# The data is stored as a hex-encoded string to save space.\n")
        f.write("# The data can be converted back to a numpy array using 'np.frombuffer(bytes.fromhex(data_hex), dtype=np.complex128)'.\n\n")
        yaml.dump(d, f, indent=2, sort_keys=False)



# --- loading ---
def load(filename):
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
        raise ValueError(f"Unsupported file version: {version}")

    else:
        raise ValueError(f"Unsupported file type: {os.path.splitext(filename)[1] == ".hdf5"}")

def load_v1(filename):
    with h5py.File(filename, "r") as f:
        data = np.array(f["signal"])
        metadata = json.loads(str(f.attrs["config"]).strip())
    return data, metadata

def load_v2(filename):
    with open(filename, "r") as f:
        d = yaml.safe_load(f)
        metadata = d["metadata"]
        data_hex = d["data"]
        data = np.frombuffer(bytes.fromhex(data_hex), dtype=np.complex128)
    return data, metadata

def load_folder(folder, name="*.hdf5"):
    import glob
    data_files = glob.glob(os.path.join(folder, name))

    data_list = []
    for file in data_files:
        data, metadata = load(os.path.join(file))
        data_list.append((data, metadata))

    return data_list
