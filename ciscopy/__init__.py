"""CISCOPy public package API."""

from ciscopy.background import azimuthal_radial_intensity, minimum_background, uniform_background
from ciscopy.cme import (
    CMECandidate,
    CMEFitDiagnostic,
    candidates_to_table,
    characterize_cmes,
    detect_cme_regions,
    fourier_motion_filter,
    preprocess_sequence,
)
from ciscopy.core import CISCOResult, main, write_table
from ciscopy.diagnostics import write_height_time_map_svg
from ciscopy.geometry import mask_disk, polar_transform
from ciscopy.io import load_fits_sequence, normalize_input
from ciscopy.movie import write_cme_movie
from ciscopy.pipeline import CISCO
from ciscopy.presets import InstrumentPreset, get_instrument_preset, infer_instrument_preset, resolve_preset
from ciscopy.sequence import SolarFrame, SolarSequence, downsample_sequence
from ciscopy.validation import filter_reference_events, patel_2021_reference_table, summarize_reference_coverage

__all__ = [
    "CISCO",
    "CISCOResult",
    "CMECandidate",
    "CMEFitDiagnostic",
    "InstrumentPreset",
    "SolarFrame",
    "SolarSequence",
    "azimuthal_radial_intensity",
    "candidates_to_table",
    "characterize_cmes",
    "detect_cme_regions",
    "downsample_sequence",
    "filter_reference_events",
    "fourier_motion_filter",
    "get_instrument_preset",
    "infer_instrument_preset",
    "load_fits_sequence",
    "main",
    "mask_disk",
    "minimum_background",
    "normalize_input",
    "patel_2021_reference_table",
    "polar_transform",
    "preprocess_sequence",
    "resolve_preset",
    "summarize_reference_coverage",
    "uniform_background",
    "write_cme_movie",
    "write_height_time_map_svg",
    "write_table",
]
