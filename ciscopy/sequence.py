"""Core data containers used throughout CISCOPy."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np

try:
    from astropy.io.fits import Header
    from astropy.time import Time
    from astropy.wcs import WCS
except ModuleNotFoundError:  # pragma: no cover
    Header = Any
    Time = Any
    WCS = Any


@dataclass(slots=True)
class SolarFrame:
    """Single solar image plus metadata."""

    data: np.ndarray
    header: Header | None = None
    wcs: WCS | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    unit: Any | None = None
    source: str | Path | None = None
    time: Time | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            msg = f"SolarFrame data must be 2D, got shape {self.data.shape!r}."
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the image shape."""

        return self.data.shape

    def to_ndcube(self) -> Any:
        """Return the frame as an `ndcube.NDCube`."""

        try:
            from ndcube import NDCube
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = "ndcube is required to convert SolarFrame to NDCube."
            raise ModuleNotFoundError(msg) from exc

        meta = dict(self.meta)
        if self.header is not None:
            meta.setdefault("header", self.header)
        if self.time is not None:
            meta.setdefault("date_obs", self.time.isot)
        return NDCube(self.data, wcs=self.wcs, meta=meta, unit=self.unit)

    def to_sunpy_map(self) -> Any:
        """Return the frame as a `sunpy.map.GenericMap` when possible."""

        if self.header is None:
            msg = "A FITS header is required to create a SunPy map."
            raise ValueError(msg)
        try:
            import sunpy.map
        except ModuleNotFoundError as exc:  # pragma: no cover
            msg = "sunpy is required to convert SolarFrame to a SunPy map."
            raise ModuleNotFoundError(msg) from exc
        return sunpy.map.Map(self.data, self.header)


@dataclass(slots=True)
class SolarSequence:
    """Ordered collection of solar image frames."""

    frames: list[SolarFrame]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("SolarSequence requires at least one frame.")

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, index: int) -> SolarFrame:
        return self.frames[index]

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the common cube shape when images are stackable."""

        first_shape = self.frames[0].shape
        if any(frame.shape != first_shape for frame in self.frames):
            msg = "Frames have different shapes; use per-frame processing instead of cube access."
            raise ValueError(msg)
        return (len(self.frames), *first_shape)

    @property
    def times(self) -> list[Time | None]:
        """Return times for all frames."""

        return [frame.time for frame in self.frames]

    def as_cube(self) -> np.ndarray:
        """Stack frames into a 3D cube with time on axis 0."""

        return np.stack([frame.data for frame in self.frames], axis=0)

    def headers(self) -> list[Header | None]:
        """Return headers for all frames."""

        return [frame.header for frame in self.frames]

    def wcs_list(self) -> list[WCS | None]:
        """Return WCS objects for all frames."""

        return [frame.wcs for frame in self.frames]

    def to_ndcube_list(self) -> list[Any]:
        """Convert each frame to `ndcube.NDCube`."""

        return [frame.to_ndcube() for frame in self.frames]
