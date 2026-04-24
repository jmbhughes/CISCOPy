"""Geometry utilities built on standard scientific packages."""

from __future__ import annotations

from functools import lru_cache
from typing import Any
import warnings

import numpy as np
from scipy.ndimage import map_coordinates

try:
    from astropy.wcs import WCS
except ModuleNotFoundError:  # pragma: no cover
    WCS = Any

from ciscopy.sunpy_compat import prepare_legacy_solar_header

try:
    import astropy.units as u
    import sunpy.map
    from sunpy.coordinates import HelioprojectiveRadial
    from sunpy.map.header_helper import make_hpr_header
except Exception:  # pragma: no cover
    u = Any
    sunpy = None
    HelioprojectiveRadial = Any
    make_hpr_header = None


def _header_value(header: Any, key: str) -> Any:
    if header is None:
        return None
    if hasattr(header, "get"):
        return header.get(key)
    return getattr(header, key, None)


def infer_center(
    shape: tuple[int, int],
    *,
    header: Any | None = None,
    center: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Infer image center from explicit coordinates or FITS metadata."""

    if center is not None:
        return center

    x_center = _header_value(header, "CRPIX1")
    y_center = _header_value(header, "CRPIX2")
    if x_center is not None and y_center is not None:
        return float(x_center) - 1.0, float(y_center) - 1.0

    return (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0


def radius_in_pixels(
    *,
    header: Any | None = None,
    radius_rsun: float,
    pixel_scale_arcsec: float | None = None,
    rsun_obs_arcsec: float | None = None,
) -> float:
    """Convert a solar radius value to pixels."""

    prepared_header = prepare_legacy_solar_header(header)
    pixel_scale = pixel_scale_arcsec or _header_value(prepared_header, "CDELT1")
    rsun_obs = rsun_obs_arcsec or _header_value(prepared_header, "RSUN_OBS") or _header_value(prepared_header, "RSUN_ARC")
    if pixel_scale is None or rsun_obs is None:
        msg = "Need pixel scale and observed solar radius to convert solar radii to pixels."
        raise ValueError(msg)
    return float(rsun_obs) * radius_rsun / abs(float(pixel_scale))


def mask_disk(
    image: np.ndarray,
    *,
    radius_rsun: float,
    header: Any | None = None,
    center: tuple[float, float] | None = None,
    pixel_scale_arcsec: float | None = None,
    rsun_obs_arcsec: float | None = None,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Mask the solar disk interior."""

    data = np.asarray(image, dtype=float)
    x_center, y_center = infer_center(data.shape, header=header, center=center)
    radius_pixels = radius_in_pixels(
        header=header,
        radius_rsun=radius_rsun,
        pixel_scale_arcsec=pixel_scale_arcsec,
        rsun_obs_arcsec=rsun_obs_arcsec,
    )

    y_coords, x_coords = np.indices(data.shape)
    radial_distance = np.sqrt((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)
    masked = data.copy()
    masked[radial_distance < radius_pixels] = fill_value
    return masked


def radial_coordinate_grid(
    shape: tuple[int, int],
    *,
    center: tuple[float, float] | None = None,
    header: Any | None = None,
) -> np.ndarray:
    """Return a radius grid in pixels."""

    x_center, y_center = infer_center(shape, header=header, center=center)
    return _cached_radial_coordinate_grid(shape, x_center, y_center).copy()


@lru_cache(maxsize=32)
def _cached_radial_coordinate_grid(
    shape: tuple[int, int],
    x_center: float,
    y_center: float,
) -> np.ndarray:
    y_coords, x_coords = np.indices(shape, dtype=float)
    return np.sqrt((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)


def polar_transform(
    image: np.ndarray,
    *,
    center: tuple[float, float] | None = None,
    header: Any | None = None,
    wcs: WCS | None = None,
    theta_samples: int = 360,
    radial_samples: int = 256,
    r_min: float = 0.0,
    r_max: float | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an image into polar coordinates.

    For FITS-backed solar data, this uses SunPy's helioprojective-radial
    reprojection path based on the input header metadata. For non-solar array
    inputs without an appropriate solar header, it falls back to SciPy
    interpolation using the same north-at-zero, counterclockwise convention.
    """

    data = np.asarray(image, dtype=float)
    prepared_header = prepare_legacy_solar_header(header)
    x_center, y_center = infer_center(data.shape, header=prepared_header, center=center)
    theta_offset_deg = solar_angle_offset_deg(
        data.shape,
        header=prepared_header,
        wcs=wcs,
        center=(x_center, y_center),
    )

    if r_max is None:
        radius_grid = radial_coordinate_grid(data.shape, center=(x_center, y_center))
        r_max = float(radius_grid.max())

    if _solar_header_supports_hpr(prepared_header):
        return _polar_transform_with_sunpy_hpr(
            data,
            header=prepared_header,
            theta_samples=theta_samples,
            radial_samples=radial_samples,
            r_min=float(r_min),
            r_max=float(r_max),
        )

    theta, radius, coordinates = _cached_polar_sampling_coordinates(
        theta_samples,
        radial_samples,
        float(r_min),
        float(r_max),
        float(x_center),
        float(y_center),
        float(theta_offset_deg),
    )

    polar = map_coordinates(data, coordinates, order=order, mode=mode, cval=cval).reshape(theta_samples, radial_samples)
    return polar, theta.copy(), radius.copy()


@lru_cache(maxsize=64)
def _cached_polar_sampling_coordinates(
    theta_samples: int,
    radial_samples: int,
    r_min: float,
    r_max: float,
    x_center: float,
    y_center: float,
    theta_offset_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 360.0, theta_samples, endpoint=False)
    radius = np.linspace(r_min, r_max, radial_samples)
    theta_rad = np.deg2rad(theta - theta_offset_deg)
    radii, angles = np.meshgrid(radius, theta_rad, indexing="xy")
    x_coords = x_center - radii * np.sin(angles)
    y_coords = y_center - radii * np.cos(angles)
    coordinates = np.vstack((y_coords.ravel(), x_coords.ravel()))
    return theta, radius, coordinates


def solar_angle_offset_deg(
    shape: tuple[int, int],
    *,
    header: Any | None = None,
    wcs: WCS | None = None,
    center: tuple[float, float] | None = None,
) -> float:
    """Estimate the solar-angle offset of the image y-up polar axis.

    The raw polar transform assumes array-up is `0` degrees. For solar-image
    headers with helioprojective WCS, this helper measures the world angle of
    that raw axis and returns the offset needed to reinterpret the output angle
    grid as solar-north-up, counterclockwise position angle.
    """

    working_wcs = wcs or _wcs_from_header(header)
    if working_wcs is None:
        return 0.0

    ctype1 = str(_header_value(header, "CTYPE1") or "").upper()
    ctype2 = str(_header_value(header, "CTYPE2") or "").upper()
    if "HPLN" not in ctype1 or "HPLT" not in ctype2:
        return 0.0

    x_center, y_center = infer_center(shape, header=header, center=center)
    sample_scale = max(1.0, min(shape) / 64.0)
    x_values = np.asarray([x_center, x_center], dtype=float)
    y_values = np.asarray([y_center, y_center - sample_scale], dtype=float)

    try:
        world_x, world_y = working_wcs.all_pix2world(x_values, y_values, 0)
    except Exception:  # pragma: no cover - malformed or incomplete WCS
        return 0.0

    delta_x = float(world_x[1] - world_x[0])
    delta_y = float(world_y[1] - world_y[0])
    if np.isclose(delta_x, 0.0) and np.isclose(delta_y, 0.0):
        return 0.0
    return float(np.degrees(np.arctan2(delta_x, delta_y)) % 360.0)


def _wcs_from_header(header: Any | None) -> WCS | None:
    if header is None or WCS is Any:
        return None
    try:
        return WCS(header)
    except Exception:  # pragma: no cover - malformed or incomplete WCS
        return None


def _solar_header_supports_hpr(header: Any | None) -> bool:
    ctype1 = str(_header_value(header, "CTYPE1") or "").upper()
    ctype2 = str(_header_value(header, "CTYPE2") or "").upper()
    instrument_text = " ".join(
        str(value)
        for value in (
            _header_value(header, "INSTRUME"),
            _header_value(header, "TELESCOP"),
            _header_value(header, "DETECTOR"),
        )
        if value is not None
    ).upper()
    return ("HPLN" in ctype1 and "HPLT" in ctype2) or "LASCO" in instrument_text


def _polar_transform_with_sunpy_hpr(
    image: np.ndarray,
    *,
    header: Any,
    theta_samples: int,
    radial_samples: int,
    r_min: float,
    r_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sunpy is None or make_hpr_header is None or u is Any:
        raise ValueError("SunPy helioprojective-radial reprojection is unavailable in this environment.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_map = sunpy.map.Map(np.asarray(image, dtype=float), header)
    observer_coordinate = input_map.observer_coordinate
    pixel_scale = abs(float(_header_value(header, "CDELT1") or 1.0))
    theta_min = (r_min * pixel_scale) * u.arcsec
    theta_binsize = (((r_max - r_min) * pixel_scale) / max(radial_samples - 1, 1)) * u.arcsec
    hpr_header = make_hpr_header(
        observer_coordinate,
        (radial_samples, theta_samples),
        theta_binsize=theta_binsize,
        theta_min=theta_min,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        polar_map = input_map.reproject_to(hpr_header)

    if HelioprojectiveRadial is not Any and not isinstance(polar_map.coordinate_frame, HelioprojectiveRadial):
        msg = "SunPy reprojection did not produce a HelioprojectiveRadial map."
        raise ValueError(msg)

    polar = np.asarray(polar_map.data, dtype=float).T
    theta = np.linspace(0.0, 360.0, theta_samples, endpoint=False)
    radius_arcsec = theta_min.to_value(u.arcsec) + np.arange(radial_samples, dtype=float) * theta_binsize.to_value(u.arcsec)
    radius_pixels = radius_arcsec / pixel_scale
    return polar, theta, radius_pixels
