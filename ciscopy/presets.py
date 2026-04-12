"""Instrument-specific processing presets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class InstrumentPreset:
    """Instrument-specific defaults for the CIISCO pipeline."""

    name: str
    r_min_rsun: float = 1.5
    r_max_rsun: float = 3.0
    theta_samples: int = 360
    radial_samples: int = 256
    positive_fraction: float = 0.05
    disk_mask_rsun: float = 1.05
    threshold_sigma: float = 4.0
    min_width_deg: int = 10
    min_area: int = 50
    min_speed_kms: float = 100.0
    max_speed_kms: float = 2000.0
    hough_min_votes: int = 8


PRESETS: dict[str, InstrumentPreset] = {
    "generic": InstrumentPreset("generic"),
    "aia171": InstrumentPreset("aia171", radial_samples=256, threshold_sigma=3.5, min_area=45),
    "aia304": InstrumentPreset("aia304", radial_samples=256, threshold_sigma=3.0, min_area=40),
    "swap174": InstrumentPreset("swap174", radial_samples=192, threshold_sigma=3.25, min_area=35),
    "euvi304": InstrumentPreset("euvi304", radial_samples=192, threshold_sigma=3.0, min_area=35),
    "lasco_c2": InstrumentPreset("lasco_c2", r_min_rsun=2.0, r_max_rsun=6.0, radial_samples=256, threshold_sigma=3.5, min_area=45),
    "lasco_c3": InstrumentPreset("lasco_c3", r_min_rsun=3.5, r_max_rsun=16.0, radial_samples=320, threshold_sigma=3.5, min_area=45),
    "cor1": InstrumentPreset("cor1", r_min_rsun=1.4, r_max_rsun=4.0, radial_samples=224, threshold_sigma=3.25, min_area=40),
    "cor2": InstrumentPreset("cor2", r_min_rsun=2.5, r_max_rsun=15.0, radial_samples=320, threshold_sigma=3.25, min_area=45),
    "metis": InstrumentPreset("metis", r_min_rsun=1.5, r_max_rsun=6.0, radial_samples=256, threshold_sigma=3.25, min_area=40),
    "suvi": InstrumentPreset("suvi", r_min_rsun=1.2, r_max_rsun=4.0, radial_samples=224, threshold_sigma=3.0, min_area=35),
    "kcor": InstrumentPreset("kcor", r_min_rsun=1.05, r_max_rsun=3.0, radial_samples=224, threshold_sigma=3.0, min_area=35),
}


def get_instrument_preset(name: str | None) -> InstrumentPreset:
    """Return an instrument preset by name."""

    if name is None:
        return PRESETS["generic"]
    key = (
        name.strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )
    aliases = {
        "generic": "generic",
        "aia171": "aia171",
        "aia(171a)": "aia171",
        "aia171a": "aia171",
        "aia304": "aia304",
        "aia(304a)": "aia304",
        "aia304a": "aia304",
        "swap174": "swap174",
        "swap(174a)": "swap174",
        "swap174a": "swap174",
        "euvia304": "euvi304",
        "euvib304": "euvi304",
        "euvi304": "euvi304",
        "euvi-a304": "euvi304",
        "euvi-b304": "euvi304",
        "euvi-a(304a)": "euvi304",
        "euvi-b(304a)": "euvi304",
        "lascoc2": "lasco_c2",
        "lasco_c2": "lasco_c2",
        "lasco c2": "lasco_c2",
        "lasco/c2": "lasco_c2",
        "lascoc3": "lasco_c3",
        "lasco_c3": "lasco_c3",
        "lasco c3": "lasco_c3",
        "lasco/c3": "lasco_c3",
        "stereocor1": "cor1",
        "stereo cor1": "cor1",
        "stereo_cor1": "cor1",
        "stereo/cor1": "cor1",
        "cor1": "cor1",
        "secchicor1": "cor1",
        "stereocor2": "cor2",
        "stereo cor2": "cor2",
        "stereo_cor2": "cor2",
        "stereo/cor2": "cor2",
        "cor2": "cor2",
        "secchicor2": "cor2",
        "solarorbitermetis": "metis",
        "solar orbiter metis": "metis",
        "metis": "metis",
        "goessuvi": "suvi",
        "goes suvi": "suvi",
        "suvi": "suvi",
        "mlsokcoronagraph": "kcor",
        "mlso k-coronagraph": "kcor",
        "mlso kcoronagraph": "kcor",
        "k-coronagraph": "kcor",
        "kcoronagraph": "kcor",
        "kcor": "kcor",
    }
    resolved = aliases.get(key)
    if resolved is None:
        available = ", ".join(sorted(PRESETS))
        msg = f"Unknown preset {name!r}. Available presets: {available}."
        raise ValueError(msg)
    return PRESETS[resolved]


def infer_instrument_preset(header: Any | None) -> InstrumentPreset:
    """Infer an instrument preset from FITS metadata when possible."""

    if header is None:
        return PRESETS["generic"]

    if hasattr(header, "get"):
        instrument = str(header.get("INSTRUME", "") or "").strip().upper()
        wavelength = str(header.get("WAVELNTH", "") or header.get("WAVE_LEN", "") or "").strip()
        detector = str(header.get("DETECTOR", "") or "").strip().upper()
        telescope = str(header.get("TELESCOP", "") or header.get("OBSERVAT", "") or "").strip().upper()
    else:
        instrument = str(getattr(header, "INSTRUME", "") or "").strip().upper()
        wavelength = str(getattr(header, "WAVELNTH", "") or getattr(header, "WAVE_LEN", "") or "").strip()
        detector = str(getattr(header, "DETECTOR", "") or "").strip().upper()
        telescope = str(getattr(header, "TELESCOP", "") or getattr(header, "OBSERVAT", "") or "").strip().upper()

    if instrument == "AIA" and wavelength == "171":
        return PRESETS["aia171"]
    if instrument == "AIA" and wavelength == "304":
        return PRESETS["aia304"]
    if instrument == "SWAP":
        return PRESETS["swap174"]
    if instrument.startswith("EUVI") and wavelength == "304":
        return PRESETS["euvi304"]
    if instrument == "LASCO" and detector == "C2":
        return PRESETS["lasco_c2"]
    if instrument == "LASCO" and detector == "C3":
        return PRESETS["lasco_c3"]
    if instrument in {"COR1", "SECCHI"} and detector == "COR1":
        return PRESETS["cor1"]
    if instrument in {"COR2", "SECCHI"} and detector == "COR2":
        return PRESETS["cor2"]
    if instrument == "METIS" or telescope == "METIS":
        return PRESETS["metis"]
    if instrument == "SUVI" or telescope.startswith("GOES"):
        return PRESETS["suvi"]
    if instrument in {"KCOR", "K-COR"} or telescope in {"KCOR", "K-COR", "MLSO", "K-CORONAGRAPH"}:
        return PRESETS["kcor"]
    return PRESETS["generic"]


def resolve_preset(preset: str | InstrumentPreset | None, *, header: Any | None = None) -> InstrumentPreset:
    """Resolve an explicit or inferred preset."""

    if isinstance(preset, InstrumentPreset):
        return preset
    if preset is not None:
        return get_instrument_preset(preset)
    return infer_instrument_preset(header)
