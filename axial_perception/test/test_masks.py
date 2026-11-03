import numpy as np

from axial_perception.core.masks import decode_gz_instance


def encode(class_img, instance_img):
    rgb = np.stack([instance_img & 0xFF, instance_img >> 8, class_img], axis=-1)
    return rgb.astype(np.uint8).tobytes()


def test_round_trip():
    class_img = np.array([[0, 4, 4], [8, 4, 6]], np.int32)
    instance_img = np.array([[0, 1, 300], [1, 1, 2]], np.int32)
    cls, mask = decode_gz_instance(encode(class_img, instance_img), 2, 3)
    np.testing.assert_array_equal(cls, class_img)
    assert cls.dtype == np.uint8
    assert mask[0, 0] == 0
    assert mask[0, 1] == mask[1, 1]
    assert len({mask[0, 1], mask[0, 2], mask[1, 0], mask[1, 2]}) == 4


def test_same_instance_id_in_different_classes_are_different_masks():
    cls, mask = decode_gz_instance(encode(np.array([[4, 8]]), np.array([[1, 1]])), 1, 2)
    assert mask[0, 0] != mask[0, 1]


def test_background_is_zero_even_with_instance_bits():
    _, mask = decode_gz_instance(encode(np.array([[0]]), np.array([[5]])), 1, 1)
    assert mask[0, 0] == 0
