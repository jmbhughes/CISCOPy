"""SunPy compatibility helpers for legacy solar FITS metadata."""

from __future__ import annotations

import warnings
from typing import Any

STANDARD_RSUN_REF_METERS = 695700000.0
STANDARD_RSUN_OBS_ARCSEC = 959.63
EARTH_DSUN_OBS_METERS = 149597870700.0


def _header_value(header: Any, key: str) -> Any:
    if header is None:
        return None
    if hasattr(header, "get"):
        return header.get(key)
    return getattr(header, key, None)


def prepare_legacy_solar_header(header: Any | None) -> Any | None:
    """Fill standard solar metadata for legacy coronagraph headers when needed.

    This currently targets LASCO-style headers that can be sparse compared to
    modern helioprojective FITS-WCS metadata. A warning is emitted when
    standard assumptions are inserted so downstream processing can still
    proceed.
    """

    if header is None:
        return None

    instrument_text = " ".join(
        str(value)
        for value in (
            _header_value(header, "INSTRUME"),
            _header_value(header, "TELESCOP"),
            _header_value(header, "DETECTOR"),
        )
        if value is not None
    ).upper()
    if "LASCO" not in instrument_text:
        return header.copy() if hasattr(header, "copy") else header

    prepared = header.copy() if hasattr(header, "copy") else header
    assumptions: list[str] = []

    if _header_value(prepared, "RSUN_REF") is None:
        prepared["RSUN_REF"] = STANDARD_RSUN_REF_METERS
        assumptions.append("standard photospheric radius")

    if _header_value(prepared, "RSUN_OBS") is None and _header_value(prepared, "RSUN_ARC") is None:
        prepared["RSUN_OBS"] = STANDARD_RSUN_OBS_ARCSEC
        assumptions.append("standard apparent solar radius")

    missing_observer = [
        key
        for key in ("HGLN_OBS", "HGLT_OBS", "DSUN_OBS")
        if _header_value(prepared, key) is None and _header_value(prepared, key.lower()) is None
    ]
    if missing_observer:
        prepared["HGLN_OBS"] = 0.0
        prepared["HGLT_OBS"] = 0.0
        prepared["DSUN_OBS"] = EARTH_DSUN_OBS_METERS
        assumptions.append("Earth-based observer")

    if assumptions:
        warnings.warn(
            "Legacy LASCO-style FITS metadata detected; assuming "
            + ", ".join(assumptions)
            + " to continue processing.",
            UserWarning,
            stacklevel=2,
        )

    return prepared
