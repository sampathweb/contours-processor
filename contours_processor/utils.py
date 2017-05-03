import numpy as np
import h5py

from .exceptions import InvalidDatasetError


def dump_dataset(filename, datasets, sources=None):
    """Save the Datasets and Sources in file

    Create HDF5 formatted datasets and metadata keys to process data.
    """
    if datasets is None:
        datasets = {}
    if sources is None:
        sources = {}

    try:
        with h5py.File(filename, "w") as f:
            data_keys = []
            for key, dataset in datasets.items():
                f.create_dataset(key,
                                 dataset.shape,
                                 dtype=dataset.dtype,
                                data=dataset)
                data_keys.append(key)
            metadata = f.create_group("metadata")
            metadata["_data_keys"] = ",".join(data_keys)
            metadata["sources"] = f.create_group("sources")
            for key, value in sources.items():
                metadata["sources"][key] = value
    except Exception:
        raise InvalidDatasetError("Error Saving Datasets File: {}".format(filename))


def load_dataset(filename):
    """Return datasets and sources in the file

    Use the metadata keys to create dataset arrays.  Convert HDF5 to numpy arrays
    """
    try:
        datasets = {}
        sources = {}
        with h5py.File(filename, "r") as f:
            metadata = f.get("metadata")
            data_keys = metadata["_data_keys"].value
            data_keys = data_keys.split(",")
            for data_key in data_keys:
                datasets[data_key] = f.get(data_key).value
            for key, g_item in metadata.get("sources").items():
                sources[key] = g_item.value
            sources["filename"] = filename
    except Exception:
        raise InvalidDatasetError("Error Processing File: {}".format(filename))
    return datasets, sources


def is_subset_binary(main_array, subset_array):
    """Return True if subset_array is fully contained in the main_array.

    Both Arrays need to be of same size.
    Usage: to check if i-contours is within the boundaries of o-contours
    """
    assert main_array.shape == subset_array.shape
    incorrect_indxes = np.where((main_array == 0) & (subset_array > 0))[0]
    return len(incorrect_indxes) == 0
