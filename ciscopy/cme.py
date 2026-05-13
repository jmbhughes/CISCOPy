"""End-to-end CME detection and characterization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter, label, median_filter, sobel

from ciscopy.background import azimuthal_radial_intensity, minimum_background, uniform_background
from ciscopy.geometry import mask_disk, polar_transform
from ciscopy.presets import InstrumentPreset, resolve_preset
from ciscopy.sequence import SolarSequence

try:
    from astropy.table import Table
except ModuleNotFoundError:  # pragma: no cover
    Table = Any


@dataclass(slots=True)
class CMECandidate:
    """Detected CME properties."""

    date: str
    start_time: str
    position_angle: float
    width: float
    speed: float
    del_speed: float
    speed_min: float
    speed_max: float
    acceleration: float
    del_acceleration: float
    acceleration_min: float
    acceleration_max: float


@dataclass(slots=True)
class CMEFitDiagnostic:
    """Diagnostic products for a fitted CME height-time ridge."""

    position_angle: float
    width: float
    start_time: str
    end_time: str
    start_index: int
    end_index: int
    time_seconds: np.ndarray
    radius_rsun: np.ndarray
    height_time: np.ndarray
    ridge_time_seconds: np.ndarray
    ridge_radius_rsun: np.ndarray


@dataclass(slots=True)
class ProcessedSequence:
    """Intermediate products from the CME pipeline."""

    corrected_cube: np.ndarray
    polar_cube: np.ndarray
    polar_angles_deg: np.ndarray
    polar_radius_pixels: np.ndarray
    polar_radius_rsun: np.ndarray
    height_time_cube: np.ndarray
    motion_filtered_cube: np.ndarray
    cadence_seconds: float
    km_per_pixel: float


@dataclass(slots=True)
class CMEDetectionDebug:
    """Intermediate angle-time products used during CME region detection."""

    cme_map: np.ndarray
    work_map: np.ndarray
    detrended_map: np.ndarray
    threshold_map: np.ndarray
    binary_mask: np.ndarray
    edge_frames: int


def _header_value(header: Any | None, key: str) -> Any:
    if header is None:
        return None
    if hasattr(header, "get"):
        return header.get(key)
    return getattr(header, key, None)


def _cadence_seconds(sequence: SolarSequence, fallback_seconds: float = 60.0) -> float:
    valid_times = [time for time in sequence.times if time is not None]
    if len(valid_times) < 2:
        return fallback_seconds
    deltas = np.diff([time.unix for time in valid_times])
    positive = deltas[deltas > 0]
    if positive.size == 0:
        return fallback_seconds
    return float(np.median(positive))


def _solar_radii_metadata(sequence: SolarSequence) -> tuple[float, float]:
    header = sequence[0].header
    rsun_ref = float(_header_value(header, "RSUN_REF") or 695700.0)
    rsun_ref_km = rsun_ref / 1000.0 if rsun_ref > 1.0e7 else rsun_ref
    rsun_obs_arcsec = float(_header_value(header, "RSUN_OBS") or _header_value(header, "RSUN_ARC") or 960.0)
    return rsun_ref_km, rsun_obs_arcsec


def _build_uniform_background(sequence: SolarSequence, min_background: np.ndarray) -> np.ndarray:
    radius, profile = azimuthal_radial_intensity(min_background, header=sequence[0].header)
    _ = radius
    return uniform_background(min_background.shape, profile, header=sequence[0].header)


def _validate_sequence_for_kinematics(sequence: SolarSequence) -> None:
    """Ensure the sequence contains enough frames for CME kinematics."""

    if len(sequence) < 3:
        msg = (
            "CME characterization requires at least 3 time-ordered images. "
            "Single-image or two-image inputs cannot provide reliable speed and acceleration."
        )
        raise ValueError(msg)


def preprocess_sequence(
    sequence: SolarSequence,
    *,
    preset: str | InstrumentPreset | None = None,
    r_min_rsun: float | None = None,
    r_max_rsun: float | None = None,
    theta_samples: int | None = None,
    radial_samples: int | None = None,
    positive_fraction: float | None = None,
    disk_mask_rsun: float | None = None,
) -> ProcessedSequence:
    """Run the CIISCO-style preprocessing pipeline.

    Parameters
    ----------
    sequence
        Normalized solar image sequence.
    r_min_rsun, r_max_rsun
        Inner and outer radial bounds of the polar analysis region in solar
        radii.
    theta_samples
        Number of angular bins in the polar representation.
    radial_samples
        Number of radial bins in the polar representation.
    positive_fraction
        Fraction of the lowest positive values used to estimate the minimum
        background.
    disk_mask_rsun
        Radius of the solar-disk mask in solar radii.

    Returns
    -------
    ProcessedSequence
        Intermediate products needed by the downstream detection and
        characterization stages.
    """

    resolved_preset = resolve_preset(preset, header=sequence[0].header)
    r_min_rsun = resolved_preset.r_min_rsun if r_min_rsun is None else r_min_rsun
    r_max_rsun = resolved_preset.r_max_rsun if r_max_rsun is None else r_max_rsun
    theta_samples = resolved_preset.theta_samples if theta_samples is None else theta_samples
    radial_samples = resolved_preset.radial_samples if radial_samples is None else radial_samples
    positive_fraction = resolved_preset.positive_fraction if positive_fraction is None else positive_fraction
    disk_mask_rsun = resolved_preset.disk_mask_rsun if disk_mask_rsun is None else disk_mask_rsun

    cube = sequence.as_cube()
    min_background = minimum_background(cube, positive_fraction=positive_fraction)
    uniform_bg = _build_uniform_background(sequence, min_background)
    mask = mask_disk(
        np.ones_like(min_background),
        radius_rsun=disk_mask_rsun,
        header=sequence[0].header,
        fill_value=0.0,
    )
    uniform_bg = np.where(uniform_bg == 0, np.nanmedian(uniform_bg[uniform_bg > 0]), uniform_bg)
    corrected = np.nan_to_num(((cube - min_background[None, ...]) / uniform_bg[None, ...]) * mask[None, ...], nan=0.0)

    rsun_ref_km, rsun_obs_arcsec = _solar_radii_metadata(sequence)
    pixel_scale = float(_header_value(sequence[0].header, "CDELT1") or 1.0)
    r_min_pix = r_min_rsun * rsun_obs_arcsec / pixel_scale
    r_max_pix = r_max_rsun * rsun_obs_arcsec / pixel_scale
    km_per_pixel = ((r_max_rsun - r_min_rsun) * rsun_ref_km) / radial_samples

    polar_frames = []
    polar_angles = None
    polar_radius_pixels = None
    for frame in corrected:
        polar, polar_angles, polar_radius_pixels = polar_transform(
            frame,
            header=sequence[0].header,
            wcs=sequence[0].wcs,
            theta_samples=theta_samples,
            radial_samples=radial_samples,
            r_min=r_min_pix,
            r_max=r_max_pix,
        )
        polar_frames.append(median_filter(polar, size=3))

    polar_cube = np.stack(polar_frames, axis=0)
    median_polar = np.mean(polar_cube, axis=0) * 0.8
    median_vector = gaussian_filter(np.median(median_polar, axis=0), sigma=4.0)
    polar_cube = polar_cube / np.where(median_vector[None, None, :] == 0, 1.0, median_vector[None, None, :])
    height_time_cube = np.transpose(polar_cube, (1, 2, 0))
    motion_filtered_cube = fourier_motion_filter(height_time_cube)
    polar_radius_rsun = np.linspace(r_min_rsun, r_max_rsun, radial_samples)

    return ProcessedSequence(
        corrected_cube=corrected,
        polar_cube=polar_cube,
        polar_angles_deg=polar_angles,
        polar_radius_pixels=polar_radius_pixels,
        polar_radius_rsun=polar_radius_rsun,
        height_time_cube=height_time_cube,
        motion_filtered_cube=motion_filtered_cube,
        cadence_seconds=_cadence_seconds(sequence),
        km_per_pixel=km_per_pixel,
    )


def fourier_motion_filter(height_time_cube: np.ndarray) -> np.ndarray:
    """Apply the quadrant-masking Fourier motion filter used in the legacy code.

    Parameters
    ----------
    height_time_cube
        Array with shape `(n_angles, n_radii, n_times)`.

    Returns
    -------
    numpy.ndarray
        Motion-filtered height-time cube with the same shape as the input.
    """

    n_angles, n_radii, n_times = height_time_cube.shape
    filtered = np.zeros_like(height_time_cube)
    pad_size = max(n_radii, n_times)
    x_offset = (pad_size - n_times) // 2

    omega = np.fft.fftshift(np.fft.fftfreq(pad_size, d=1.0))
    kr = np.fft.fftshift(np.fft.fftfreq(pad_size, d=1.0))
    omega_grid, kr_grid = np.meshgrid(omega, kr, indexing="ij")
    with np.errstate(divide="ignore", invalid="ignore"):
        speed_grid = -omega_grid / kr_grid

    mask = np.isfinite(speed_grid) & (speed_grid > 0.0)
    center = pad_size // 2
    cent = max(3, min(9, pad_size // 16))
    mask[center - cent : center + cent + 1, center - cent : center + cent + 1] = False
    mask = gaussian_filter(mask.astype(float), sigma=2.0)

    for angle_index in range(n_angles):
        ht_map = height_time_cube[angle_index]
        padded = np.zeros((pad_size, pad_size), dtype=float)
        padded[x_offset : x_offset + n_times, :n_radii] = ht_map.T
        fft_map = np.fft.fftshift(np.fft.fft2(padded))
        filtered_fft = fft_map * mask
        filtered_map = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fft)))
        cropped = filtered_map[x_offset : x_offset + n_times, :n_radii].T
        filtered[angle_index] = median_filter(cropped, size=3)
    return filtered


def _sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """Return the two-dimensional Sobel gradient magnitude."""

    grad_time = sobel(image, axis=0, mode="nearest")
    grad_radius = sobel(image, axis=1, mode="nearest")
    return np.hypot(grad_time, grad_radius)


def _wrapped_component_masks(binary: np.ndarray) -> list[np.ndarray]:
    """Return connected components on a circular angle axis.

    The angle axis is duplicated once so structures spanning the 359/0 degree
    seam can be labeled as a single component. Equivalent duplicated labels are
    deduplicated by their modulo-angle footprint and time footprint, keeping the
    shortest-span representation.
    """

    n_angles, _n_times = binary.shape
    binary_ext = np.concatenate([binary, binary], axis=0)
    binary_ext = binary_closing(binary_ext, structure=np.ones((5, 5), dtype=bool))
    labels_ext, count = label(binary_ext)
    best_components: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, np.ndarray]] = {}

    for label_id in range(1, count + 1):
        rows, cols = np.where(labels_ext == label_id)
        if rows.size == 0:
            continue
        row_key = tuple(np.unique(rows % n_angles).tolist())
        col_key = tuple(np.unique(cols).tolist())
        key = (row_key, col_key)
        span = int(rows.max() - rows.min() + 1)
        mask = labels_ext == label_id
        existing = best_components.get(key)
        if existing is None or span < existing[0]:
            best_components[key] = (span, mask)

    return [item[1] for item in best_components.values()]


def compute_cme_detection_debug(
    processed: ProcessedSequence,
    sequence: SolarSequence,
    *,
    preset: str | InstrumentPreset | None = None,
    threshold_sigma: float | None = None,
) -> CMEDetectionDebug:
    """Build the adaptive angle-time detection products for CME candidates.

    The thresholding is adaptive in two senses:
    1. a low-frequency angle-time background is removed from the CME map
    2. a local robust-noise estimate defines a threshold surface instead of a
       single scalar threshold for the entire day
    """

    resolved_preset = resolve_preset(preset, header=sequence[0].header)
    threshold_sigma = resolved_preset.threshold_sigma if threshold_sigma is None else threshold_sigma

    cme_map = np.sum(np.abs(processed.motion_filtered_cube), axis=1)
    cme_map = gaussian_filter(cme_map, sigma=(2.0, 1.0))
    _n_angles, n_times = cme_map.shape
    edge_frames = min(5, max(1, n_times // 10))
    work = cme_map[:, edge_frames : n_times - edge_frames] if n_times > edge_frames * 2 else cme_map

    background = gaussian_filter(work, sigma=(12.0, 3.0))
    detrended = work - background

    local_center = gaussian_filter(detrended, sigma=(6.0, 1.5))
    local_abs_dev = gaussian_filter(np.abs(detrended - local_center), sigma=(6.0, 1.5))
    local_sigma = 1.4826 * local_abs_dev

    median_value = float(np.median(detrended))
    mad = float(np.median(np.abs(detrended - median_value)))
    global_sigma = max(1.4826 * mad, np.std(detrended) / 4.0, 1.0e-6)
    global_threshold = median_value + threshold_sigma * global_sigma
    adaptive_threshold = threshold_sigma * np.maximum(local_sigma, 0.2 * global_sigma)
    threshold_map = np.minimum(adaptive_threshold, global_threshold)
    binary = detrended >= threshold_map

    return CMEDetectionDebug(
        cme_map=cme_map,
        work_map=work,
        detrended_map=detrended,
        threshold_map=threshold_map,
        binary_mask=binary,
        edge_frames=edge_frames,
    )


def detect_cme_regions(
    processed: ProcessedSequence,
    sequence: SolarSequence,
    *,
    preset: str | InstrumentPreset | None = None,
    threshold_sigma: float | None = None,
    min_width_deg: int | None = None,
    min_area: int | None = None,
    min_speed_kms: float | None = None,
    max_speed_kms: float | None = None,
    max_duration_hours: float | None = None,
) -> list[dict[str, Any]]:
    """Find candidate CME regions in angle-time space.

    Parameters
    ----------
    processed
        Intermediate products returned by :func:`preprocess_sequence`.
    sequence
        Original normalized image sequence.
    threshold_sigma
        Detection threshold in robust-sigma units after detrending the
        angle-time activity map.
    min_width_deg
        Minimum angular width of a valid candidate, in degrees.
    min_area
        Minimum connected-region size in the angle-time detection mask.
    min_speed_kms, max_speed_kms
        Speed bounds used to reject unrealistically short or long events.
    max_duration_hours
        Maximum candidate duration allowed in the angle-time CME map. Regions
        that persist too long are rejected as likely background structures
        rather than transients.

    Returns
    -------
    list[dict[str, Any]]
        Candidate region descriptors used by the kinematics stage.
    """

    resolved_preset = resolve_preset(preset, header=sequence[0].header)
    threshold_sigma = resolved_preset.threshold_sigma if threshold_sigma is None else threshold_sigma
    min_width_deg = resolved_preset.min_width_deg if min_width_deg is None else min_width_deg
    min_area = resolved_preset.min_area if min_area is None else min_area
    min_speed_kms = resolved_preset.min_speed_kms if min_speed_kms is None else min_speed_kms
    max_speed_kms = resolved_preset.max_speed_kms if max_speed_kms is None else max_speed_kms
    max_duration_hours = resolved_preset.max_duration_hours if max_duration_hours is None else max_duration_hours

    debug = compute_cme_detection_debug(
        processed,
        sequence,
        preset=resolved_preset,
        threshold_sigma=threshold_sigma,
    )
    cme_map = debug.cme_map
    n_angles, _n_times = cme_map.shape
    edge_frames = debug.edge_frames
    binary = debug.binary_mask
    wrapped_masks = _wrapped_component_masks(binary)

    rsun_ref_km, _ = _solar_radii_metadata(sequence)
    radial_span_km = (processed.polar_radius_rsun[-1] - processed.polar_radius_rsun[0]) * rsun_ref_km
    min_duration = int(np.ceil(radial_span_km / (max_speed_kms * processed.cadence_seconds)))
    max_duration = int(np.ceil(radial_span_km / (min_speed_kms * processed.cadence_seconds)))
    if max_duration_hours > 0.0:
        duration_limit = int(np.floor((max_duration_hours * 3600.0) / processed.cadence_seconds))
        max_duration = min(max_duration, max(1, duration_limit))

    regions: list[dict[str, Any]] = []
    for component_mask in wrapped_masks:
        rows, cols = np.where(component_mask)
        if rows.size < min_area:
            continue
        if rows.min() >= n_angles:
            rows = rows - n_angles
        angle_span = rows.max() - rows.min() + 1
        time_span = cols.max() - cols.min() + 1
        if angle_span < min_width_deg or time_span < min_duration or time_span > max_duration:
            continue

        start_time_index = edge_frames + int(cols.min())
        end_time_index = edge_frames + int(cols.max())
        angle_start = int(rows.min())
        angle_stop = int(rows.max())
        angle_indices = np.arange(angle_start, angle_stop + 1, dtype=int) % n_angles
        aggregated_ht = np.sum(
            np.abs(processed.motion_filtered_cube[angle_indices, :, start_time_index : end_time_index + 1]),
            axis=0,
        ).T
        angle_step = 360.0 / n_angles
        center_index = round((angle_start + angle_stop) / 2.0) % n_angles
        regions.append(
            {
                "start_index": start_time_index,
                "end_index": end_time_index,
                "angle_start": angle_start % n_angles,
                "angle_stop": angle_stop % n_angles,
                "position_angle": float(processed.polar_angles_deg[center_index]),
                "width": float(angle_span * angle_step),
                "height_time": aggregated_ht,
            }
        )
    return regions


def _parabolic_hough_candidates(
    height_time: np.ndarray,
    *,
    cadence_seconds: float,
    km_per_pixel: float,
    min_votes: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    edges = _sobel_magnitude(height_time)
    threshold = np.mean(edges) + 1.75 * np.std(edges)
    binary = edges > threshold
    time_len, radial_len = binary.shape
    aval, root_lookup = _cached_hough_lookup(radial_len)
    accumulator = np.zeros((time_len, aval.size), dtype=int)
    points = np.argwhere(binary)
    if points.size > 0:
        time_points = points[:, 0][:, None]
        radius_points = points[:, 1]
        roots = root_lookup[radius_points]
        t0_values = np.rint(time_points - roots).astype(int)
        valid_idx = (t0_values >= 0) & (t0_values < time_len)
        aval_indices = np.broadcast_to(np.arange(aval.size), t0_values.shape)
        flat_bins = t0_values[valid_idx] * aval.size + aval_indices[valid_idx]
        bincount = np.bincount(flat_bins, minlength=time_len * aval.size)
        accumulator = bincount.reshape(time_len, aval.size)

    if accumulator.max() < min_votes:
        return accumulator, np.empty((0, 2), dtype=int)

    peak_mask = accumulator >= 0.85 * accumulator.max()
    peak_mask = binary_closing(peak_mask, structure=np.ones((5, 5), dtype=bool))
    peak_labels, count = label(peak_mask)
    candidates = []
    for label_id in range(1, count + 1):
        rows, cols = np.where(peak_labels == label_id)
        if rows.size < 8:
            continue
        t0 = int(np.median(rows))
        a_idx = int(np.median(cols))
        candidates.append((t0, a_idx))
    return accumulator, np.asarray(candidates, dtype=int)


@lru_cache(maxsize=16)
def _cached_hough_lookup(radial_len: int) -> tuple[np.ndarray, np.ndarray]:
    aval = np.geomspace(0.01, 10.0, 1000)
    radius_indices = np.arange(radial_len, dtype=float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        root_lookup = np.sqrt(radius_indices / aval[None, :])
    return aval, root_lookup


def _best_parabolic_candidate(
    height_time: np.ndarray,
    *,
    cadence_seconds: float,
    km_per_pixel: float,
    min_votes: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    accumulator, candidates = _parabolic_hough_candidates(
        height_time,
        cadence_seconds=cadence_seconds,
        km_per_pixel=km_per_pixel,
        min_votes=min_votes,
    )
    if candidates.size == 0:
        return accumulator, candidates, None

    scores = np.asarray([accumulator[t0_index, a_index] for t0_index, a_index in candidates], dtype=float)
    best_index = int(np.argmax(scores))
    return accumulator, candidates, tuple(int(value) for value in candidates[best_index])


def _ridge_from_hough_peak(
    height_time: np.ndarray,
    *,
    best_peak: tuple[int, int],
    cadence_seconds: float,
    radial_len: int,
    km_per_pixel: float,
    r_min_rsun: float,
    r_max_rsun: float,
) -> tuple[np.ndarray, np.ndarray]:
    t0_index, a_index = best_peak
    aval, _root_lookup = _cached_hough_lookup(radial_len)
    a_param = float(aval[a_index])
    times = np.arange(height_time.shape[0], dtype=float)
    radii = a_param * (times - t0_index) ** 2
    valid = (radii >= 0.0) & (radii < radial_len) & (times >= float(t0_index))
    if np.count_nonzero(valid) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    ridge_time_seconds = times[valid] * cadence_seconds
    radius_rsun = np.linspace(r_min_rsun, r_max_rsun, radial_len)
    ridge_radius_rsun = np.interp(radii[valid], np.arange(radial_len, dtype=float), radius_rsun)
    _ = km_per_pixel
    return ridge_time_seconds, ridge_radius_rsun


def characterize_region(
    region: dict[str, Any],
    processed: ProcessedSequence,
    sequence: SolarSequence,
    *,
    preset: str | InstrumentPreset | None = None,
) -> CMECandidate | None:
    """Fit parabolic trajectories in a detected height-time region.

    Parameters
    ----------
    region
        Candidate CME region returned by :func:`detect_cme_regions`.
    processed
        Preprocessed sequence products.
    sequence
        Original normalized image sequence.

    Returns
    -------
    CMECandidate or None
        Characterized CME properties, or `None` if the region does not produce
        a usable kinematic estimate.
    """

    resolved_preset = resolve_preset(preset, header=sequence[0].header)
    accumulator, candidates, _best_peak = _best_parabolic_candidate(
        region["height_time"],
        cadence_seconds=processed.cadence_seconds,
        km_per_pixel=processed.km_per_pixel,
        min_votes=resolved_preset.hough_min_votes,
    )
    _ = accumulator
    if candidates.size == 0:
        return None

    height_time = region["height_time"]
    time_len, radial_len = height_time.shape
    aval, _root_lookup = _cached_hough_lookup(radial_len)
    y0 = 0.0
    speeds = []
    accelerations = []

    for t0_index, a_index in candidates:
        a_param = float(aval[a_index])
        times = np.arange(time_len)
        radii = y0 + a_param * (times - t0_index) ** 2
        valid = (radii >= 20) & (radii < radial_len) & (times >= int(np.floor(t0_index + 0.5)))
        if np.count_nonzero(valid) < 3:
            continue
        sample_times = times[valid] * processed.cadence_seconds
        sample_radii = radii[valid] * processed.km_per_pixel
        velocity_fit = np.polyfit(sample_times, sample_radii, 1)
        speed = float(velocity_fit[0])
        acceleration = float(2.0 * a_param * processed.km_per_pixel * 1000.0 / (processed.cadence_seconds**2))
        speeds.append(speed)
        accelerations.append(acceleration)

    if not speeds or not accelerations:
        return None

    start_frame = sequence[min(region["start_index"], len(sequence) - 1)]
    time_value = start_frame.time.isot if start_frame.time is not None else f"frame-{region['start_index']:04d}"
    date_value = time_value[:10]

    return CMECandidate(
        date=date_value,
        start_time=time_value,
        position_angle=float(region["position_angle"]),
        width=float(region["width"]),
        speed=float(np.mean(speeds)),
        del_speed=float(np.std(speeds)),
        speed_min=float(np.min(speeds)),
        speed_max=float(np.max(speeds)),
        acceleration=float(np.mean(accelerations)),
        del_acceleration=float(np.std(accelerations)),
        acceleration_min=float(np.min(accelerations)),
        acceleration_max=float(np.max(accelerations)),
    )


def build_candidate_diagnostic(
    sequence: SolarSequence,
    processed: ProcessedSequence,
    *,
    preset: str | InstrumentPreset | None = None,
    candidate_index: int = 0,
) -> CMEFitDiagnostic:
    """Build a height-time diagnostic product for one detected CME candidate.

    Parameters
    ----------
    sequence
        Original normalized solar image sequence.
    processed
        Preprocessed intermediate products for the same sequence.
    preset
        Optional preset used to configure the Hough threshold.
    candidate_index
        Index of the detected CME region to diagnose.

    Returns
    -------
    CMEFitDiagnostic
        Height-time map together with the best-fit ridge in solar-radius units.
    """

    resolved_preset = resolve_preset(preset, header=sequence[0].header)
    regions = detect_cme_regions(processed, sequence, preset=resolved_preset)
    if not regions:
        raise ValueError("No CME candidate regions were detected, so no height-time diagnostic can be built.")
    if candidate_index < 0 or candidate_index >= len(regions):
        raise IndexError(f"candidate_index {candidate_index} is out of range for {len(regions)} detected candidates.")

    region = regions[candidate_index]
    height_time = region["height_time"]
    time_len, radial_len = height_time.shape
    _accumulator, _candidates, best_peak = _best_parabolic_candidate(
        height_time,
        cadence_seconds=processed.cadence_seconds,
        km_per_pixel=processed.km_per_pixel,
        min_votes=resolved_preset.hough_min_votes,
    )
    if best_peak is None:
        raise ValueError("A CME region was detected, but the parabolic Hough fit did not find a usable ridge.")

    ridge_time_seconds, ridge_radius_rsun = _ridge_from_hough_peak(
        height_time,
        best_peak=best_peak,
        cadence_seconds=processed.cadence_seconds,
        radial_len=radial_len,
        km_per_pixel=processed.km_per_pixel,
        r_min_rsun=float(processed.polar_radius_rsun[0]),
        r_max_rsun=float(processed.polar_radius_rsun[-1]),
    )

    start_frame = sequence[min(region["start_index"], len(sequence) - 1)]
    start_time = start_frame.time.isot if start_frame.time is not None else f"frame-{region['start_index']:04d}"
    time_seconds = np.arange(time_len, dtype=float) * processed.cadence_seconds
    ridge_end_offset = round(float(ridge_time_seconds[-1]) / processed.cadence_seconds) if ridge_time_seconds.size else 0
    end_index = min(region["start_index"] + ridge_end_offset, len(sequence) - 1)
    end_frame = sequence[end_index]
    end_time = end_frame.time.isot if end_frame.time is not None else f"frame-{end_index:04d}"

    return CMEFitDiagnostic(
        position_angle=float(region["position_angle"]),
        width=float(region["width"]),
        start_time=start_time,
        end_time=end_time,
        start_index=int(region["start_index"]),
        end_index=end_index,
        time_seconds=time_seconds,
        radius_rsun=processed.polar_radius_rsun.copy(),
        height_time=height_time.copy(),
        ridge_time_seconds=ridge_time_seconds,
        ridge_radius_rsun=ridge_radius_rsun,
    )


def characterize_cmes(
    sequence: SolarSequence,
    *,
    preset: str | InstrumentPreset | None = None,
    r_min_rsun: float | None = None,
    r_max_rsun: float | None = None,
    theta_samples: int | None = None,
    radial_samples: int | None = None,
    reduce_resolution: bool = True,
    downsample_factor: int | None = None,
    target_max_dim: int = 512,
) -> tuple[list[CMECandidate], ProcessedSequence]:
    """Run the full CIISCO-style pipeline and return detected CME properties.

    Parameters
    ----------
    sequence
        Normalized solar image sequence.
    r_min_rsun, r_max_rsun
        Inner and outer radial bounds of the analysis region in solar radii.
    theta_samples
        Number of angular bins in the polar transform.
    radial_samples
        Number of radial bins in the polar transform.
    reduce_resolution, downsample_factor, target_max_dim
        Accepted for API compatibility. Any downsampling should already happen
        during FITS loading before characterization is called.
    Returns
    -------
    tuple[list[CMECandidate], ProcessedSequence]
        Detected CME candidates and the associated intermediate products.
    """

    _ = (reduce_resolution, downsample_factor, target_max_dim)
    _validate_sequence_for_kinematics(sequence)
    processed = preprocess_sequence(
        sequence,
        preset=preset,
        r_min_rsun=r_min_rsun,
        r_max_rsun=r_max_rsun,
        theta_samples=theta_samples,
        radial_samples=radial_samples,
    )
    regions = detect_cme_regions(processed, sequence, preset=preset)
    candidates = [characterize_region(region, processed, sequence, preset=preset) for region in regions]
    return [candidate for candidate in candidates if candidate is not None], processed


def candidates_to_table(candidates: list[CMECandidate]) -> Table:
    """Convert characterized CME candidates into an Astropy table.

    Parameters
    ----------
    candidates
        List of detected and characterized CME candidates.

    Returns
    -------
    astropy.table.Table
        Table containing the standard CME output columns.
    """

    rows = [
        {
            "date": candidate.date,
            "start_time": candidate.start_time,
            "position_angle": candidate.position_angle,
            "width": candidate.width,
            "speed": candidate.speed,
            "del_speed": candidate.del_speed,
            "speed_min": candidate.speed_min,
            "speed_max": candidate.speed_max,
            "acceleration": candidate.acceleration,
            "del_acceleration": candidate.del_acceleration,
            "acceleration_min": candidate.acceleration_min,
            "acceleration_max": candidate.acceleration_max,
        }
        for candidate in candidates
    ]
    return Table(rows=rows)
