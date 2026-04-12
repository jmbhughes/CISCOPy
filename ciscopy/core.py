"""Top-level orchestration helpers for the CISCOPy package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astropy.table import Table

from ciscopy.cme import ProcessedSequence
from ciscopy.io import normalize_input
from ciscopy.pipeline import CISCO
from ciscopy.presets import InstrumentPreset


@dataclass(slots=True)
class CISCOResult:
    """Container for the final characterization table and intermediate products."""

    table: Table
    processed: ProcessedSequence


def write_table(
    table: Table,
    path: str | Path,
    *,
    format: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Write a CME result table to a text-based output file.

    Parameters
    ----------
    table
        Astropy table returned by the characterization pipeline.
    path
        Output file path. The suffix may be used to infer the format.
    format
        Optional Astropy ASCII format override. If omitted, the format is
        inferred from the filename suffix. Supported suffixes are `.csv`,
        `.ecsv`, `.tsv`, and `.txt`.
    overwrite
        Whether to overwrite an existing file.

    Returns
    -------
    pathlib.Path
        The resolved output path.
    """

    output_path = Path(path)
    inferred_format = format or _infer_table_format(output_path)
    table.write(output_path, format=inferred_format, overwrite=overwrite)
    return output_path


def main(
    data: Any,
    *,
    header: Any | None = None,
    wcs: Any | None = None,
    meta: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    unit: Any | None = None,
    time: Any | list[Any] | None = None,
    preset: str | InstrumentPreset | None = None,
    r_min_rsun: float | None = None,
    r_max_rsun: float | None = None,
    theta_samples: int | None = None,
    radial_samples: int | None = None,
    output_path: str | Path | None = None,
    output_format: str | None = None,
    overwrite: bool = True,
) -> CISCOResult:
    """Run the end-to-end CISCOPy pipeline from flexible inputs.

    Parameters
    ----------
    data
        Input solar sequence. Supported values are FITS path lists, a 2D image,
        a 3D image cube, an `ndcube.NDCube`, or a list of `NDCube` objects.
    header
        FITS header or per-frame FITS headers associated with array inputs.
    wcs
        WCS object or per-frame WCS objects associated with array inputs.
    meta
        Optional metadata mapping or per-frame metadata mappings.
    unit
        Optional data unit for array or `NDCube` inputs.
    time
        Optional observation time or per-frame times for array inputs.
    preset
        Optional instrument preset name or preset object. When omitted, the
        package attempts to infer a preset from FITS metadata and otherwise
        falls back to a generic configuration. Supported families currently
        include AIA, SWAP, EUVI, LASCO C2/C3, STEREO COR1/COR2, Solar Orbiter
        METIS, GOES/SUVI, and MLSO K-Cor.
    r_min_rsun
        Inner radial bound of the polar processing region in solar radii. When
        omitted, the preset default is used.
    r_max_rsun
        Outer radial bound of the polar processing region in solar radii. When
        omitted, the preset default is used.
    theta_samples
        Number of position-angle samples in the polar transform. When omitted,
        the preset default is used.
    radial_samples
        Number of radial bins in the polar transform. When omitted, the preset
        default is used.
    output_path
        Optional output file path. When provided, the result table is also
        written to disk.
    output_format
        Optional Astropy ASCII writer format. If omitted, the format is inferred
        from the output filename.
    overwrite
        Whether to overwrite an existing output file.

    Returns
    -------
    CISCOResult
        A container with the final Astropy table and intermediate products.
    """

    sequence = normalize_input(data, header=header, wcs=wcs, meta=meta, unit=unit, time=time)
    pipeline = CISCO(sequence)
    table, processed = pipeline.characterize(
        preset=preset,
        r_min_rsun=r_min_rsun,
        r_max_rsun=r_max_rsun,
        theta_samples=theta_samples,
        radial_samples=radial_samples,
    )

    if output_path is not None:
        write_table(table, output_path, format=output_format, overwrite=overwrite)

    return CISCOResult(table=table, processed=processed)


def _infer_table_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "ascii.csv"
    if suffix == ".ecsv":
        return "ascii.ecsv"
    if suffix == ".tsv":
        return "ascii.tab"
    if suffix == ".txt":
        return "ascii.fixed_width"
    msg = "Could not infer output format. Use .csv, .ecsv, .tsv, .txt, or pass output_format explicitly."
    raise ValueError(msg)
