import numpy as np

from openvrcardboard.pipeline import _resize_for_inference
from openvrcardboard.pose_tracking import BodyTracker, body_joint_names


def test_inference_downscale_preserves_aspect():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = _resize_for_inference(image, 960, 0)
    assert resized.shape == (540, 960, 3)


def test_inference_downscale_can_be_disabled():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert _resize_for_inference(image, 0, 0) is image


def test_inference_does_not_upscale():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    assert _resize_for_inference(image, 960, 0) is image


def test_inference_height_cap():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = _resize_for_inference(image, 0, 540)
    assert resized.shape == (540, 960, 3)


def test_body_roles_distinguish_none_and_empty():
    assert body_joint_names(None) is None
    assert body_joint_names([]) == set()
    assert body_joint_names(["waist", "chest"]) == {"hips", "chest"}
    assert body_joint_names(["left_foot"]) == {"left_foot"}
    assert body_joint_names(["custom_joint"]) == {"custom_joint"}
