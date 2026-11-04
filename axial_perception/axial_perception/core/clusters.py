import numpy as np


def label_points(u, v, idx, class_img, mask_img, classes, min_points):
    """Assign projected points to the masks they fall on.

    Keeps points on masks of the given classes with at least min_points hits. Returns the kept point
    indices, their class and a per-frame segment id (1..K).
    """
    cls = class_img[v, u]
    mask = mask_img[v, u]
    on = np.isin(cls, classes) & (mask > 0)
    idx, cls, mask = idx[on], cls[on], mask[on]
    _, inverse, counts = np.unique(mask, return_inverse=True, return_counts=True)
    keep = counts[inverse] >= min_points
    _, segment = np.unique(inverse[keep], return_inverse=True)
    return idx[keep], cls[keep], segment + 1
