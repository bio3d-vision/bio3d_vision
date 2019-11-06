"""Convert data and annotation images to the MS COCO API
<http://cocodataset.org/#format-data>.

"""
import json
import os

import numpy as np
import tifffile as tif

from typing import Optional


def images_to_coco_json(
    output_json_file: str,
    image_file: str,
    semantic_label_file: str,
    instance_label_file: str,
    convert_to_2d: Optional[bool] = True):
    """

    Args:
        output_json_file:
        image_file:
        semantic_label_file:
        instance_label_file:
        convert_to_2d:

    Returns: None

    """