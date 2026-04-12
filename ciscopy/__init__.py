"""CISCOPy public package API."""

from ciscopy.background import azimuthal_radial_intensity
from ciscopy.background import minimum_background
from ciscopy.background import uniform_background
from ciscopy.cme import CMECandidate
from ciscopy.cme import candidates_to_table
from ciscopy.cme import characterize_cmes
from ciscopy.cme import detect_cme_regions
from ciscopy.cme import fourier_motion_filter
from ciscopy.cme import preprocess_sequence
from ciscopy.core import CISCOResult
from ciscopy.core import main
from ciscopy.core import write_table
from ciscopy.geometry import mask_disk, polar_transform
from ciscopy.io import load_fits_sequence, normalize_input
from ciscopy.pipeline import CISCO
from ciscopy.presets import InstrumentPreset
from ciscopy.presets import get_instrument_preset
from ciscopy.presets import infer_instrument_preset
from ciscopy.presets import resolve_preset
from ciscopy.sequence import SolarFrame, SolarSequence
from ciscopy.validation import filter_reference_events
from ciscopy.validation import patel_2021_reference_table
from ciscopy.validation import summarize_reference_coverage

__all__ = [
    "CISCO",
    "CISCOResult",
    "CMECandidate",
    "InstrumentPreset",
    "SolarFrame",
    "SolarSequence",
    "azimuthal_radial_intensity",
    "candidates_to_table",
    "characterize_cmes",
    "detect_cme_regions",
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
    "filter_reference_events",
    "summarize_reference_coverage",
    "uniform_background",
    "write_table",
]
