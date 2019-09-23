"""Convert label data images between indexed and dict encodings.


Label images, typically numpy arrays, can also be saved as a dict of binary
masks (dict). The dict has one key per unique nonzero value in the image,
plus a '.info' key containing a dict with image shape and dtype information.
Masks are represented as run-length encoded strings.

Example: Synthetic blob image with two nonzero values.

blob1 = skimage.data.binary_blobs(length=10, seed=1, volume_fraction=0.2)
blob2 = skimage.data.binary_blobs(length=10, seed=2, volume_fraction=0.1)
image = np.zeros((10, 10), dtype=int)
image[blob1] = 1
image[blob2] = 2

# Basic encoder
img_as_dict = {int(i): binary_rle_encode(img == i)
               for i in np.unique(img) if i != 0}
img_as_dict['.info'] = {'shape': img.shape, 'dtype': img.dtype}

# `img`'s representation as a dict:
img_as_dict = {
  '.info': {'dtype': 'np.int', 'shape': (10, 10)},
  '1': '7 2 12 1 21 1 40 1 50 4 60 3 68 1 71 1 81 1',
  '2': '1 3 11 1 33 2 42 3 87 2'}

"""
import json
import os

import matplotlib
import numpy as np
import tifffile as tif

from typing import Any, Dict, Optional, Sequence, Tuple, Union

LabelDict = Dict[Union[int, str], Union[str, Dict[str, Any]]]


def json_to_indexed_tif(json_in: str, tif_out: Optional[str] = None):
    """Convert a dict-encoded label, saved as a JSON file, to a TIF volume.

    Args:
        json_in (str): Input JSON file name.
        tif_out (Optional[str]): Output TIF file name. If None, use the same
            base name as `json_in`.

    Returns: None

    """
    if tif_out is None:
        base, ext = os.path.splitext(json_in)
        tif_out = base + '.tif'

    with open(json_in, 'r') as fd:
        label_dict = json.load(fd)

    label_arr = dict_to_indexed(label_dict)

    tif.imsave(tif_out, label_arr, compress=6)
    pass


def json_to_rgb_tif(
        json_in: str,
        cmap: matplotlib.colors.LinearSegmentedColormap,
        background_color: Optional[Tuple[float, ...]] = None,
        tif_out: Optional[str] = None):
    """

    Args:
        json_in:
        cmap:
        background_color:
        tif_out:

    Returns: None

    """
    if tif_out is None:
        base, ext = os.path.splitext(json_in)
        tif_out = base + '.tif'

    if background_color is not None:
        cmap = replace_background(cmap, background_color)

    with open(json_in, 'r') as fd:
        label_dict = json.load(fd)

    label_arr = dict_to_indexed(label_dict)
    label_rgba = (255 * cmap(label_arr / label_arr.max())).astype(np.uint8)

    tif.imsave(tif_out, label_rgba, compress=6)
    pass


def dict_to_indexed(
        label_dict: LabelDict) -> np.ndarray:
    """Convert a dict-encoded label image into an index-encoded numpy array.

    Args:
        label_dict (LabelDict): A label image encoded as a dict of binary
            masks, one key per unique nonzero value in the image. Masks are
            stored as run-length encoded strings.

    Returns:
        (np.ndarray): Index-encoded image array.

    """
    label_keys = [k for k in label_dict.keys() if k != '.info']
    label_info = label_dict['.info']
    shape = label_info['shape']
    dtype = np.dtype(label_info['dtype'])
    label_img = np.zeros(shape=shape, dtype=dtype)
    for i in label_keys:
        binary_mask = binary_rle_decode(label_dict[i], shape=shape)
        label_img[binary_mask] = int(i)
    return label_img


def indexed_to_dict(label_img: np.ndarray, **kwargs) -> LabelDict:
    """Convert an index-encoded label image array into a dict of RLE string
    masks.

    Args:
        label_img (np.ndarray): Index-encoded label image array.

    Returns:
        (LabelDict): A label image encoded as a dict of binary masks,
            one key per unique nonzero value in the image. Masks are stored as
            run-length encoded strings.

    """
    as_dict = {str(int(i)): binary_rle_encode(label_img == i)
               for i in np.unique(label_img) if i != 0}
    as_dict['.info'] = {'shape': label_img.shape,
                        'dtype': str(label_img.dtype),
                        **kwargs}
    return as_dict


def binary_rle_decode(mask_rle: str, shape: Sequence[int]) -> np.ndarray:
    """Decode an RLE string into a numpy array.

    Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    Args:
        mask_rle (str): Run-length encoding string of a binary image.
        shape (Sequence[int]): Shape of the image to recover.

    Returns:
        (np.ndarray): Binary image mask as a numpy array.

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    pixels = np.zeros(np.prod(shape), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        pixels[lo:hi] = 1
    img = pixels.reshape(shape).astype(bool)
    return img


def binary_rle_encode(img: np.ndarray) -> str:
    """Encode a binary image mask as an RLE string.

    Source: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

    Args:
        img (np.ndarray): Binary image. 0 is background, 1 is mask.

    Returns:
        (str): Run-length encoding of `img`.

    """
    if set(np.unique(img)) != {0, 1}:
        raise ValueError('Input `img` is not a binary array.')

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def replace_background(cmap, background_color):
    cmap_list = list(cmap(np.linspace(0, 1, 256)))
    cmap_list[0] = background_color
    new_cmap = cmap.from_list('newcmap', cmap_list, N=256)
    return new_cmap
