"""Tests for geometry and background utilities."""

from __future__ import annotations

import numpy as np

from ciscopy import azimuthal_radial_intensity
from ciscopy import mask_disk
from ciscopy import minimum_background
from ciscopy import polar_transform
from ciscopy import uniform_background


def test_mask_disk_zeros_interior() -> None:
    image = np.ones((9, 9))

    masked = mask_disk(
        image,
        radius_rsun=1.0,
        pixel_scale_arcsec=1.0,
        rsun_obs_arcsec=2.0,
        center=(4.0, 4.0),
    )

    assert masked[4, 4] == 0.0
    assert masked[0, 0] == 1.0


def test_polar_transform_preserves_radial_line() -> None:
    image = np.zeros((11, 11))
    image[:6, 5] = 10.0

    polar, theta, radius = polar_transform(image, center=(5.0, 5.0), theta_samples=360, radial_samples=5, r_max=4.0)

    assert polar.shape == (360, 5)
    assert theta[0] == 0.0
    assert radius[-1] == 4.0
    assert polar[0, 1] > 0.0


def test_azimuthal_radial_intensity_returns_profile() -> None:
    image = np.ones((7, 7))

    radius, profile = azimuthal_radial_intensity(image, center=(3.0, 3.0), max_radius=4)

    assert radius.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert np.allclose(profile, 1.0)


def test_minimum_background_uses_low_percentile_positive_signal() -> None:
    cube = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[10.0, 9.0], [8.0, 7.0]],
        ]
    )

    background = minimum_background(cube)

    assert background.shape == (2, 2)
    assert np.all(background <= np.max(cube, axis=0))
    assert np.all(background > 0.0)


def test_minimum_background_ignores_zero_and_negative_values() -> None:
    cube = np.array(
        [
            [[0.0, -1.0], [2.0, 0.0]],
            [[3.0, 4.0], [0.0, -2.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    background = minimum_background(cube)

    assert np.all(background > 0.0)
    assert background[0, 0] >= 3.0
    assert background[0, 1] >= 4.0


def test_minimum_background_raises_when_no_positive_values_exist() -> None:
    cube = np.array(
        [
            [[0.0, -1.0], [0.0, -2.0]],
            [[0.0, -3.0], [0.0, -4.0]],
        ]
    )

    try:
        minimum_background(cube)
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("minimum_background should reject cubes without positive samples.")


def test_uniform_background_maps_profile_to_image_shape() -> None:
    profile = np.array([1.0, 2.0, 3.0, 4.0])

    background = uniform_background((7, 7), profile, center=(3.0, 3.0), smooth_sigma=0.0)

    assert background.shape == (7, 7)
    assert background[3, 3] == 1.0
    assert background[0, 0] == 4.0
