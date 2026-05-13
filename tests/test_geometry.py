"""Tests for geometry and background utilities."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from ciscopy import azimuthal_radial_intensity
from ciscopy import mask_disk
from ciscopy import minimum_background
from ciscopy import polar_transform
import ciscopy.geometry as geometry
from ciscopy.geometry import solar_angle_offset_deg
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


def test_polar_transform_uses_counterclockwise_angles_from_north() -> None:
    image = np.zeros((11, 11))
    image[5, :6] = 8.0

    polar, theta, _radius = polar_transform(image, center=(5.0, 5.0), theta_samples=360, radial_samples=5, r_max=4.0)

    assert theta[90] == 90.0
    assert polar[90, 1] > 0.0


def test_solar_angle_offset_uses_helioprojective_wcs_when_available() -> None:
    header = fits.Header()
    header["CRPIX1"] = 6.0
    header["CRPIX2"] = 6.0
    header["CDELT1"] = 1.0
    header["CDELT2"] = 1.0
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CROTA2"] = -10.0
    header["PC1_1"] = np.cos(np.deg2rad(-10.0))
    header["PC1_2"] = np.sin(np.deg2rad(-10.0))
    header["PC2_1"] = -np.sin(np.deg2rad(-10.0))
    header["PC2_2"] = np.cos(np.deg2rad(-10.0))

    offset = solar_angle_offset_deg((11, 11), header=header)

    assert 150.0 <= offset <= 210.0


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


def test_hpr_transform_converts_pixel_radii_to_arcsec(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, float] = {}

    class DummyHPRFrame:
        pass

    class DummyMap:
        def __init__(self, data: np.ndarray, header: fits.Header) -> None:
            self.data = np.asarray(data, dtype=float)
            self.header = header
            self.observer_coordinate = object()

        def reproject_to(self, header: object):
            _ = header
            polar_data = np.ones((360, 5), dtype=float)
            return type(
                "DummyPolarMap",
                (),
                {"data": polar_data, "coordinate_frame": DummyHPRFrame()},
            )()

    def fake_make_hpr_header(
        observer_coordinate: object,
        shape: tuple[int, int],
        *,
        theta_binsize,
        theta_min,
    ) -> dict[str, object]:
        _ = observer_coordinate, shape
        calls["theta_min_arcsec"] = float(theta_min.to_value(geometry.u.arcsec))
        calls["theta_binsize_arcsec"] = float(theta_binsize.to_value(geometry.u.arcsec))
        return {"dummy": True}

    monkeypatch.setattr(geometry.sunpy.map, "Map", DummyMap)
    monkeypatch.setattr(geometry, "make_hpr_header", fake_make_hpr_header)
    monkeypatch.setattr(geometry, "HelioprojectiveRadial", DummyHPRFrame)

    header = fits.Header()
    header["CDELT1"] = 5.0
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"

    polar, _theta, radius = geometry._polar_transform_with_sunpy_hpr(
        np.ones((16, 16), dtype=float),
        header=header,
        theta_samples=360,
        radial_samples=5,
        r_min=2.0,
        r_max=6.0,
    )

    assert polar.shape == (5, 360)
    assert radius[0] == 2.0
    assert radius[-1] == 6.0
    assert calls["theta_min_arcsec"] == 10.0
    assert calls["theta_binsize_arcsec"] == 5.0
