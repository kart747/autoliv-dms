import numpy as np
import pytest

from dms.camera import Preprocessor, apply_source_override, to_bgr, to_rgb
from dms.config import CameraConfig


def test_gray_frame_to_rgb_and_bgr():
    g = np.full((48, 64), 100, np.uint8)
    assert to_rgb(g).shape == (48, 64, 3)
    assert to_bgr(g).shape == (48, 64, 3)


def test_bgr_to_rgb_swaps_channels():
    img = np.zeros((2, 2, 3), np.uint8)
    img[..., 0] = 255  # blue
    assert to_rgb(img)[0, 0].tolist() == [0, 0, 255]


def test_preprocessor_force_grayscale():
    pre = Preprocessor(CameraConfig(force_grayscale=True))
    out = pre(np.zeros((48, 64, 3), np.uint8))
    assert out.shape == (48, 64)


def test_preprocessor_squeezes_single_channel():
    pre = Preprocessor(CameraConfig())
    assert pre(np.zeros((48, 64, 1), np.uint8)).shape == (48, 64)


def test_preprocessor_keeps_colour_by_default():
    pre = Preprocessor(CameraConfig())
    assert pre(np.zeros((48, 64, 3), np.uint8)).shape == (48, 64, 3)


@pytest.mark.parametrize("img", [np.zeros((48, 64), np.uint8), np.zeros((48, 64, 3), np.uint8)])
def test_clahe_preserves_shape(img):
    pre = Preprocessor(CameraConfig(clahe=True))
    assert pre(img).shape == img.shape


def test_source_override(tmp_path):
    c = CameraConfig()
    apply_source_override(c, "2")
    assert (c.source, c.device) == ("opencv", 2)
    apply_source_override(c, "rpicam")
    assert c.source == "rpicam"
    video = tmp_path / "a.mp4"
    video.write_bytes(b"")
    apply_source_override(c, str(video))
    assert (c.source, c.path) == ("file", str(video))
    with pytest.raises(ValueError):
        apply_source_override(c, "nonexistent.mp4")
