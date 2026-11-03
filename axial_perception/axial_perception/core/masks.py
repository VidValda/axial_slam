import numpy as np


def decode_gz_instance(data, height, width):
    """Gazebo instance labels_map (rgb8): class in B, per-class instance id in (G << 8) | R.

    Returns the class image and a mask image with one id per mask in the frame (0 = background).
    """
    a = np.frombuffer(data, np.uint8).reshape(height, width, 3).astype(np.int32)
    class_img = a[..., 2]
    mask_img = (class_img << 16) | (a[..., 1] << 8) | a[..., 0]
    mask_img[class_img == 0] = 0
    return class_img.astype(np.uint8), mask_img
