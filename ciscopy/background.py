"""Background and radial-profile utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ciscopy.geometry import infer_center, radial_coordinate_grid


def azimuthal_radial_intensity(
    image: np.ndarray,
    *,
    header: Any | None = None,
    center: tuple[float, float] | None = None,
    max_radius: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return annular mean intensity as a function of radius in pixels."""

    data = np.asarray(image, dtype=float)
    x_center, y_center = infer_center(data.shape, header=header, center=center)
    y_coords, x_coords = np.indices(data.shape)
    radius_map = np.sqrt((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)
    radius_index = np.floor(radius_map).astype(int)

    upper = max_radius or radius_index.max() + 1
    sums = np.bincount(radius_index.ravel(), weights=data.ravel(), minlength=upper)
    counts = np.bincount(radius_index.ravel(), minlength=upper)

    profile = np.divide(sums[:upper], counts[:upper], out=np.zeros(upper, dtype=float), where=counts[:upper] > 0)
    radius = np.arange(upper, dtype=float)
    return radius, profile


def minimum_background(cube: np.ndarray, *, positive_fraction: float = 0.05) -> np.ndarray:
    """Estimate a minimum background image using strictly positive intensities.

    Only values greater than zero are considered valid for the background
    estimate. Pixels with no positive samples fall back to the smallest positive
    value available in the cube, ensuring that the returned background remains
    strictly positive everywhere.
    """

    data = np.asarray(cube, dtype=float)
    if data.ndim != 3:
        msg = "Cube input must be 3D with time on axis 0."
        raise ValueError(msg)

    positive = np.where(data > 0, data, np.nan)
    if not np.any(np.isfinite(positive)):
        msg = "Minimum background requires at least one strictly positive value in the input cube."
        raise ValueError(msg)

    percentile = np.nanpercentile(positive, positive_fraction * 100.0, axis=0)
    global_positive_floor = float(np.nanmin(positive))
    background = np.where(np.isfinite(percentile), percentile, global_positive_floor)
    return np.clip(background, global_positive_floor, None)


def uniform_background(
    image_shape: tuple[int, int],
    radial_profile: np.ndarray,
    *,
    center: tuple[float, float] | None = None,
    header: Any | None = None,
    smooth_sigma: float = 2.0,
) -> np.ndarray:
    """Build a radially symmetric background image from a radial intensity profile."""

    profile = np.asarray(radial_profile, dtype=float)
    if smooth_sigma > 0:
        profile = gaussian_filter1d(profile, smooth_sigma, mode="nearest")

    radius_map = radial_coordinate_grid(image_shape, center=center, header=header)
    radius_index = np.clip(np.rint(radius_map).astype(int), 0, profile.size - 1)
    return profile[radius_index]
