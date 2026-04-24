"""High-level pipeline API for CIISCO/CISCO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ciscopy.cme import candidates_to_table
from ciscopy.cme import characterize_cmes
from ciscopy.io import normalize_input
from ciscopy.presets import InstrumentPreset
from ciscopy.sequence import SolarSequence


@dataclass(slots=True)
class CISCO:
    """Entry point for loading and characterizing CME sequences."""

    sequence: SolarSequence

    @classmethod
    def from_input(
        cls,
        data: Any,
        *,
        header: Any | None = None,
        wcs: Any | None = None,
        meta: dict[str, Any] | list[dict[str, Any] | None] | None = None,
        unit: Any | None = None,
        time: Any | list[Any] | None = None,
        ext: int | str | None = None,
    ) -> "CISCO":
        """Build a pipeline object from supported image-sequence inputs.

        Parameters
        ----------
        data
            FITS paths, a 2D image, a 3D image cube, an `NDCube`, or a list of
            `NDCube` objects. CME characterization requires at least 3
            time-ordered images.
        header, wcs, meta, unit, time
            Optional metadata used when `data` is provided as arrays rather than
            FITS files or `NDCube` objects.
        ext
            Optional FITS extension to use when `data` is a FITS path list. When
            omitted, the loader auto-detects the first 2D image HDU.
        """

        return cls(normalize_input(data, header=header, wcs=wcs, meta=meta, unit=unit, time=time, ext=ext))

    def as_cube(self):
        """Return the normalized time-ordered image cube."""

        return self.sequence.as_cube()

    def characterize(
        self,
        *,
        preset: str | InstrumentPreset | None = None,
        r_min_rsun: float | None = None,
        r_max_rsun: float | None = None,
        theta_samples: int | None = None,
        radial_samples: int | None = None,
        reduce_resolution: bool = True,
        downsample_factor: int | None = None,
        target_max_dim: int = 512,
    ):
        """Run the end-to-end CME characterization pipeline.

        Parameters
        ----------
        r_min_rsun, r_max_rsun
            Inner and outer radial bounds of the analysis region in solar radii.
        theta_samples
            Number of angular samples in the polar representation.
        radial_samples
            Number of radial bins in the polar representation.
        reduce_resolution
            Whether to downsample large images before processing. Enabled by
            default for faster detection of large-scale structures.
        downsample_factor
            Optional explicit integer downsampling factor.
        target_max_dim
            Maximum image dimension targeted by the default adaptive
            downsampling.

        Returns
        -------
        tuple
            A pair containing the output Astropy table and the intermediate
            processed data products. A `ValueError` is raised when fewer than 3
            frames are available.
        """

        candidates, processed = characterize_cmes(
            self.sequence,
            preset=preset,
            r_min_rsun=r_min_rsun,
            r_max_rsun=r_max_rsun,
            theta_samples=theta_samples,
            radial_samples=radial_samples,
            reduce_resolution=reduce_resolution,
            downsample_factor=downsample_factor,
            target_max_dim=target_max_dim,
        )
        return candidates_to_table(candidates), processed
