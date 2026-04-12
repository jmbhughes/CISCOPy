"""Geometry utilities built on standard scientific packages."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates


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

    pixel_scale = pixel_scale_arcsec or _header_value(header, "CDELT1")
    rsun_obs = rsun_obs_arcsec or _header_value(header, "RSUN_OBS") or _header_value(header, "RSUN_ARC")
    if pixel_scale is None or rsun_obs is None:
        msg = "Need pixel scale and observed solar radius to convert solar radii to pixels."
        raise ValueError(msg)
    return float(rsun_obs) * radius_rsun / float(pixel_scale)


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
    theta_samples: int = 360,
    radial_samples: int = 256,
    r_min: float = 0.0,
    r_max: float | None = None,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an image into polar coordinates using SciPy interpolation."""

    data = np.asarray(image, dtype=float)
    x_center, y_center = infer_center(data.shape, header=header, center=center)

    if r_max is None:
        radius_grid = radial_coordinate_grid(data.shape, center=(x_center, y_center))
        r_max = float(radius_grid.max())

    theta, radius, coordinates = _cached_polar_sampling_coordinates(
        theta_samples,
        radial_samples,
        float(r_min),
        float(r_max),
        float(x_center),
        float(y_center),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 360.0, theta_samples, endpoint=False)
    radius = np.linspace(r_min, r_max, radial_samples)
    theta_rad = np.deg2rad(theta)
    radii, angles = np.meshgrid(radius, theta_rad, indexing="xy")
    x_coords = x_center + radii * np.sin(angles)
    y_coords = y_center - radii * np.cos(angles)
    coordinates = np.vstack((y_coords.ravel(), x_coords.ravel()))
    return theta, radius, coordinates
