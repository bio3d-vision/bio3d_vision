"""Function for loading bio3d-vision dataset files on disk into numpy.

"""
import json
import os

import numpy as np
import tifffile as tif

from .convert import dict_to_indexed

from typing import Optional, Union

# A source for a data volume can either be a string, indicating a path to a
# file saved to disk, or it can be a NumPy array.
VolSource = Union[str, np.ndarray]


def load(data_dir: Optional[str] = os.path.join('platelet-em', 'images'),
         data_file: Optional[Union[str, np.ndarray]] = '50-images.tif',
         normalization: Optional[str] = None):
    """Load data from the file system.

    Args:
        data_dir (str): String specifying the location of the source
            data.
        data_file (Optional[VolSource]): Name of the data image
            volume within the data_dir, or a numpy array.
        normalization (Optional[str]): Data normalization strategy. If none is
            supplied, no normalization takes place. If 'zero_mean' is supplied,
            data is translated and scaled to have mean 0 and standard deviation
            1. If 'white_balance' is supplied, auto white balancing is applied.

    Returns: (np.ndarray) The data volume loaded.

    """
    # Load volume
    if isinstance(data_dir, str) and isinstance(data_file, str):
        data_path = os.path.join(data_dir, data_file)
        data_ext = os.path.splitext(data_file)[1].lower()
        if 'tif' in data_ext:
            data_volume = tif.imread(data_path)
        elif 'json' in data_ext:
            with open(data_path, 'r') as fd:
                label_dict = json.load(fd)
                data_volume = dict_to_indexed(label_dict)
        else:
            raise ValueError('data_file extension not recognized.')

    elif isinstance(data_file, np.ndarray):
        data_volume = data_file
    else:
        raise ValueError(f'Need to either specify strings for both data_dir'
                         f'and data_file or supply data_file as np.ndarray. ')

    if normalization == 'zero_mean':
        data_volume = data_volume - data_volume.mean()
        data_volume = data_volume / data_volume.std()
    elif normalization == 'white_balance':
        perc = 0.05
        mi = np.percentile(data_volume, perc)
        ma = np.percentile(data_volume, 100 - perc)
        data_volume = np.float32(np.clip((data_volume - mi) / (ma - mi), 0, 255))
    elif isinstance(normalization, str):
        raise ValueError(f'Unrecognized value for `normalization')

    return data_volume
