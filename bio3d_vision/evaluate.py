import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import adjusted_rand_score, confusion_matrix
from typing import List

# Imports for testing remove later
from util import *
from rgb_to_index import *
from load import *

def adj_rand_index(y_pred: np.ndarray,
                   y_true: np.ndarray):
    """Compute the adjusted rand index.
    """
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    return adjusted_rand_score(y_true, y_pred)

def instance_indices_to_masks(indices: np.ndarray):
    """Converts a 3D array [z, x, y] of indices to a binary array
    [num_instances, z, x, y] of binary masks.
    """
    instances = [i for i in np.unique(indices) if i != 0]
    masks = np.zeros((len(instances), *indices.shape))
    instances = np.array(instances)
    for counter, i in enumerate(instances):
        locations = np.where(indices==i)
        masks[counter][locations] = 1
    return masks


def mean_iou(y_pred: np.ndarray,
             y_true: np.ndarray,
             classes: List[int]):
    # ytrue, ypred is a flatten vector
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # compute mean iou
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    # Subract to avoid double counting TP
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)

def plot_confusion_matrix(y_pred: np.ndarray,
                          y_true: np.ndarray,
                          classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    # ... and label them with the respective list entries
    xticklabels=classes, yticklabels=classes,
    title=title,
    ylabel='True label',
    xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__=="__main__":
    true = [0, 1, 1, 1, 0, 2, 3]
    pred = [1, 1, 1, 2, 0, 0, 3]
    classes = list(range(4))
    #plot_confusion_matrix(true, pred, classes=classes, normalize=True)
    #convert_platelet_files(os.path.join('..', 'platelet-em'))
    indices = load(data_dir=os.path.join('..',
                                           'platelet-em',
                                           'labels-instance'),
                data_file=os.path.join('24-instance-cell.tif'),
                data_type=np.int32)
    # Slice big volume for faster runtime
    indices = indices[:1, :, :]
    masks = instance_indices_to_masks(indices)
    # Visualize masks
    instances = [i for i in np.unique(indices) if i != 0]
    for i in [8]:#range(len(instances)):
        images = (indices[0], masks[i][0])
        settings = ({'cmap': 'jet'},
                    {'cmap': 'jet'})

        imshow(images, (10, 5), settings)
    plt.show()

