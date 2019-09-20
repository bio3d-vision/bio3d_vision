"""Utility functions used in conjunction with the bio3d_vision package.

"""
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Dict, Optional, Sequence, Tuple


def imshow(images: Sequence[np.ndarray],
           figsize: Sequence[int],
           plot_settings: Optional[Sequence[Dict[str, Any]]] = None,
           layout: Optional[Tuple[int, int]] = None,
           frame: bool = True) -> None:
    """Simplify showing one or more images

    Args:
        images:
        figsize:
        plot_settings:
        layout:
        frame:

    Returns:

    """
    if not isinstance(images, Sequence):
        images = [images]

    if layout is None:
        layout = (1, len(images))

    f, axs = plt.subplots(
        nrows=layout[0],
        ncols=layout[1],
        figsize=figsize)

    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i, ax in enumerate(axs):
        if plot_settings is not None:
            ax.imshow(images[i], **plot_settings[i], extent=(0, 1, 1, 0))
        else:
            ax.imshow(images[i], extent=(0, 1, 1, 0))
        ax.axis('tight')
        if frame:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        else:
            ax.axis('off')

    pass
