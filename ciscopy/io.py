"""Flexible input loaders for FITS files, arrays, and ndcube objects."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ciscopy.sequence import SolarFrame, SolarSequence, _downsample_header
from ciscopy.sunpy_compat import prepare_legacy_solar_header

try:
    from astropy.io import fits
    from astropy.io.fits import Header
    from astropy.time import Time
    from astropy.wcs import WCS
except ModuleNotFoundError:  # pragma: no cover
    fits = None
    Header = Any
    Time = Any
    WCS = Any


PathLike = str | Path


def _require_astropy(feature: str) -> None:
    if fits is None:  # pragma: no cover
        msg = f"astropy is required for {feature}."
        raise ModuleNotFoundError(msg)


def _is_pathlike_list(data: Any) -> bool:
    return isinstance(data, Sequence) and bool(data) and all(isinstance(item, (str, Path)) for item in data)


def _looks_like_ndcube(data: Any) -> bool:
    return hasattr(data, "data") and hasattr(data, "wcs") and data.__class__.__name__ == "NDCube"


def _parse_time(time_value: Any) -> Time | None:
    if time_value is None:
        return None
    if isinstance(time_value, Time):
        return time_value
    try:
        return Time(time_value)
    except Exception:  # pragma: no cover - falls back on missing/invalid metadata
        return None


def _time_from_header(header: Header | None) -> Time | None:
    if header is None:
        return None
    return _parse_time(header.get("DATE-OBS") or header.get("DATE_OBS"))


def _normalize_header_wcs(
    header: Header | Sequence[Header | None] | None,
    wcs: WCS | Sequence[WCS | None] | None,
    frame_count: int,
) -> tuple[list[Header | None], list[WCS | None]]:
    if isinstance(header, Sequence) and not isinstance(header, (str, bytes, Header)):
        headers = list(header)
    else:
        headers = [header] * frame_count

    if isinstance(wcs, Sequence) and not isinstance(wcs, (str, bytes, WCS)):
        wcs_list = list(wcs)
    else:
        wcs_list = [wcs] * frame_count

    if len(headers) != frame_count or len(wcs_list) != frame_count:
        msg = "Header and WCS inputs must match the number of frames."
        raise ValueError(msg)

    return headers, wcs_list


def load_fits_sequence(
    paths: Sequence[PathLike],
    *,
    ext: int | str | None = None,
    downsample_factor: int | None = None,
    target_max_dim: int | None = None,
) -> SolarSequence:
    """Load a sequence of FITS images into a normalized `SolarSequence`.

    Parameters
    ----------
    paths
        Ordered collection of FITS file paths.
    ext
        FITS extension index or name containing the image data. When omitted,
        the loader first checks the primary HDU and then falls back to the
        first extension containing a 2D image.
    downsample_factor
        Optional integer stride used to downsample large FITS images while
        loading them.
    target_max_dim
        Optional maximum image dimension used to infer a loading-time
        downsampling factor when `downsample_factor` is not given.

    Returns
    -------
    SolarSequence
        Sequence of frames with image data, headers, WCS, and parsed times when
        available.
    """

    _require_astropy("FITS loading")
    frames: list[SolarFrame] = []
    resolved_factor = downsample_factor
    for path in paths:
        with fits.open(path) as hdul:
            selected_ext = _select_image_hdu(hdul, ext)
            if resolved_factor is None and target_max_dim is not None:
                resolved_factor = _infer_loader_downsample_factor(hdul[selected_ext], target_max_dim)
            image = np.asarray(hdul[selected_ext].data, dtype=float)
            if image.ndim != 2:
                msg = f"Expected 2D FITS image in {path!s}, got shape {image.shape!r}."
                raise ValueError(msg)
            header = prepare_legacy_solar_header(hdul[selected_ext].header)
            if resolved_factor is not None and resolved_factor > 1:
                image = image[::resolved_factor, ::resolved_factor]
                header = _downsample_header(header, resolved_factor)
            frames.append(
                SolarFrame(
                    data=image,
                    header=header,
                    wcs=_wcs_from_header(header),
                    meta={"path": str(path), "fits_ext": selected_ext},
                    source=path,
                    time=_time_from_header(header),
                )
            )
    return SolarSequence(frames)


def _infer_loader_downsample_factor(hdu: Any, target_max_dim: int) -> int:
    if target_max_dim <= 0:
        raise ValueError("target_max_dim must be positive.")

    header = getattr(hdu, "header", None)
    naxis1 = None if header is None else header.get("NAXIS1")
    naxis2 = None if header is None else header.get("NAXIS2")
    if naxis1 is None or naxis2 is None:
        data = getattr(hdu, "data", None)
        if data is None:
            return 1
        shape = np.asarray(data).shape
        if len(shape) < 2:
            return 1
        naxis2, naxis1 = shape[-2], shape[-1]

    max_dim = max(int(naxis1), int(naxis2))
    return max(1, int(np.ceil(max_dim / target_max_dim)))


def _select_image_hdu(hdul: Any, ext: int | str | None) -> int | str:
    if ext is not None:
        return ext

    primary_data = getattr(hdul[0], "data", None)
    if primary_data is not None and np.asarray(primary_data).ndim == 2:
        return 0

    for index, hdu in enumerate(hdul[1:], start=1):
        data = getattr(hdu, "data", None)
        if data is not None and np.asarray(data).ndim == 2:
            return index

    msg = "No 2D image HDU was found in the FITS file."
    raise ValueError(msg)


def _frames_from_ndcube_list(data: Sequence[Any]) -> SolarSequence:
    frames = []
    for cube in data:
        meta = dict(getattr(cube, "meta", {}) or {})
        header = meta.get("header")
        frames.append(
            SolarFrame(
                data=np.asarray(cube.data, dtype=float),
                header=header,
                wcs=getattr(cube, "wcs", None),
                meta=meta,
                unit=getattr(cube, "unit", None),
                time=_parse_time(meta.get("DATE-OBS") or meta.get("DATE_OBS") or meta.get("date_obs")),
            )
        )
    return SolarSequence(frames)


def _frames_from_array(
    data: np.ndarray,
    *,
    header: Header | Sequence[Header | None] | None,
    wcs: WCS | Sequence[WCS | None] | None,
    meta: dict[str, Any] | Sequence[dict[str, Any] | None] | None,
    unit: Any | None,
    time: Any | Sequence[Any] | None,
) -> SolarSequence:
    array = np.asarray(data, dtype=float)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        msg = "Array input must be a 2D image or 3D image cube."
        raise ValueError(msg)

    headers, wcs_list = _normalize_header_wcs(header, wcs, array.shape[0])

    if isinstance(meta, Sequence) and not isinstance(meta, (str, bytes, dict)):
        meta_list = list(meta)
    else:
        meta_list = [meta] * array.shape[0]

    if isinstance(time, Sequence) and not isinstance(time, (str, bytes)):
        times = list(time)
    else:
        times = [time] * array.shape[0]

    if len(meta_list) != array.shape[0] or len(times) != array.shape[0]:
        msg = "Meta and time inputs must match the number of frames."
        raise ValueError(msg)

    frames = [
        SolarFrame(
            data=array[index],
            header=headers[index],
            wcs=wcs_list[index] or _wcs_from_header(headers[index]),
            meta=meta_list[index] or {},
            unit=unit,
            time=_parse_time(times[index]) or _time_from_header(headers[index]),
        )
        for index in range(array.shape[0])
    ]
    return SolarSequence(frames)


def _wcs_from_header(header: Header | None) -> WCS | None:
    if header is None:
        return None
    _require_astropy("WCS construction")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(header)


def normalize_input(
    data: Any,
    *,
    header: Header | Sequence[Header | None] | None = None,
    wcs: WCS | Sequence[WCS | None] | None = None,
    meta: dict[str, Any] | Sequence[dict[str, Any] | None] | None = None,
    unit: Any | None = None,
    time: Any | Sequence[Any] | None = None,
    ext: int | str | None = None,
    downsample_factor: int | None = None,
    target_max_dim: int | None = None,
) -> SolarSequence:
    """Normalize supported user inputs to a single `SolarSequence`.

    Parameters
    ----------
    data
        FITS path list, a 2D image, a 3D image cube, an `NDCube`, or a list of
        `NDCube` objects.
    header, wcs, meta, unit, time
        Optional per-input metadata used primarily for NumPy-array inputs.
    ext
        Optional FITS extension to use when `data` is provided as FITS paths.
    downsample_factor
        Optional integer stride used to downsample FITS path inputs during
        loading.
    target_max_dim
        Optional maximum image dimension used to derive an adaptive FITS
        loading downsampling factor.

    Returns
    -------
    SolarSequence
        A consistent internal container used by the rest of the package.
    """

    if _is_pathlike_list(data):
        return load_fits_sequence(
            data,
            ext=ext,
            downsample_factor=downsample_factor,
            target_max_dim=target_max_dim,
        )

    if _looks_like_ndcube(data):
        return _frames_from_ndcube_list([data])

    if isinstance(data, Sequence) and data and all(_looks_like_ndcube(item) for item in data):
        return _frames_from_ndcube_list(data)

    if isinstance(data, np.ndarray):
        return _frames_from_array(data, header=header, wcs=wcs, meta=meta, unit=unit, time=time)

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes, dict)):
        as_list = list(data)
        if _is_pathlike_list(as_list):
            return load_fits_sequence(
                as_list,
                ext=ext,
                downsample_factor=downsample_factor,
                target_max_dim=target_max_dim,
            )
        if as_list and all(_looks_like_ndcube(item) for item in as_list):
            return _frames_from_ndcube_list(as_list)

    msg = (
        "Unsupported input. Provide a list of FITS file paths, a 2D image, a 3D image cube, "
        "an NDCube, or a list of NDCube objects."
    )
    raise TypeError(msg)
