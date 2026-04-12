"""Reference data and helpers for scientific validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.table import Table


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    """Published CME properties used for validation."""

    instrument: str
    date: str
    serial_no: int
    start_time: str
    position_angle: float
    speed: float
    speed_min: float
    speed_max: float
    speed_median: float | None
    acceleration: float
    acceleration_min: float
    acceleration_max: float
    acceleration_median: float | None
    remarks: str | None = None


_PATEL_2021_EVENTS: tuple[ReferenceEvent, ...] = (
    ReferenceEvent("AIA (171 A)", "2012-04-08", 1, "02:00", 115.0, 1742.0, 1361.0, 2449.0, None, 16836.0, 8726.0, 29145.0, None, "False"),
    ReferenceEvent("AIA (171 A)", "2012-06-27", 2, "09:28", 315.0, 79.0, 60.0, 98.0, 324.0, 69.0, 37.0, 100.0, 33.0, None),
    ReferenceEvent("AIA (171 A)", "2012-06-27", 3, "10:34", 315.0, 279.0, 217.0, 507.0, None, 394.0, 219.0, 1210.0, None, "False"),
    ReferenceEvent("AIA (171 A)", "2012-06-27", 4, "11:37", 73.0, 593.0, 389.0, 773.0, None, 1735.0, 754.0, 2693.0, None, "False"),
    ReferenceEvent("AIA (171 A)", "2012-08-31", 5, "19:41", 110.0, 428.0, 417.0, 439.0, 522.0, 803.0, 773.0, 833.0, 896.0, None),
    ReferenceEvent("AIA (304 A)", "2012-04-08", 6, "00:15", 228.0, 175.0, 79.0, 254.0, 724.0, 161.0, 29.0, 298.0, 110.0, None),
    ReferenceEvent("AIA (304 A)", "2012-04-08", 7, "00:41", 228.0, 220.0, 96.0, 249.0, 222.0, 227.0, 169.0, 285.0, 328.0, None),
    ReferenceEvent("AIA (304 A)", "2012-04-08", 8, "01:24", 228.0, 256.0, 244.0, 263.0, None, 309.0, 278.0, 324.0, None, "False"),
    ReferenceEvent("AIA (304 A)", "2014-07-08", 9, "16:15", 63.0, 393.0, 360.0, 487.0, 428.0, 718.0, 601.0, 850.0, -284.0, "Decelerating eruption"),
    ReferenceEvent("EUVI-A (304 A)", "2013-05-13", 10, "05:36", 143.0, 174.0, 130.0, 226.0, 166.0, 40.0, 22.0, 68.0, 20.0, None),
    ReferenceEvent("EUVI-A (304 A)", "2013-05-13", 11, "07:56", 282.0, 154.0, 66.0, 232.0, 142.0, 38.0, 6.0, 70.0, 32.0, None),
    ReferenceEvent("EUVI-A (304 A)", "2013-05-13", 12, "08:26", 81.0, 122.0, 76.0, 238.0, 158.0, 22.0, 8.0, 72.0, 84.0, None),
    ReferenceEvent("EUVI-B (304 A)", "2012-08-31", 13, "07:16", 267.0, 237.0, 171.0, 256.0, None, 72.0, 33.0, 148.0, None, "False"),
    ReferenceEvent("EUVI-B (304 A)", "2012-08-31", 14, "19:26", 249.0, 407.0, 213.0, 906.0, 399.0, 305.0, 55.0, 1004.0, 284.0, None),
    ReferenceEvent("SWAP (174 A)", "2011-12-24", 15, "11:28", 287.0, 102.0, 96.0, 105.0, 99.0, 67.0, 29.0, 35.0, 34.0, None),
    ReferenceEvent("SWAP (174 A)", "2012-04-16", 16, "17:57", 81.0, 279.0, 151.0, 382.0, 234.0, 262.0, 74.0, 457.0, 269.0, None),
    ReferenceEvent("SWAP (174 A)", "2012-04-16", 17, "21:04", 76.0, 349.0, 301.0, 391.0, None, 424.0, 280.0, 547.0, None, "False"),
    ReferenceEvent("SWAP (174 A)", "2013-05-01", 18, "02:23", 76.0, 321.0, 280.0, 405.0, 408.0, 326.0, 253.0, 521.0, 367.0, None),
    ReferenceEvent("SWAP (174 A)", "2013-06-21", 19, "03:04", 110.0, 380.0, 273.0, 575.0, 343.0, 511.0, 234.0, 1264.0, 677.0, None),
    ReferenceEvent("SWAP (174 A)", "2013-06-21", 20, "17:20", 287.0, 586.0, 452.0, 723.0, None, 1188.0, 725.0, 1857.0, None, "False"),
    ReferenceEvent("SWAP (174 A)", "2014-08-24", 21, "12:03", 124.0, 482.0, 456.0, 509.0, 417.0, 712.0, 636.0, 787.0, 760.0, None),
)


def patel_2021_reference_table() -> Table:
    """Return Table 1 from Patel et al. (2021) as an Astropy table."""

    rows = [
        {
            "instrument": event.instrument,
            "date": event.date,
            "serial_no": event.serial_no,
            "start_time": event.start_time,
            "position_angle": event.position_angle,
            "speed": event.speed,
            "speed_min": event.speed_min,
            "speed_max": event.speed_max,
            "speed_median": np.nan if event.speed_median is None else event.speed_median,
            "acceleration": event.acceleration,
            "acceleration_min": event.acceleration_min,
            "acceleration_max": event.acceleration_max,
            "acceleration_median": np.nan if event.acceleration_median is None else event.acceleration_median,
            "remarks": "" if event.remarks is None else event.remarks,
        }
        for event in _PATEL_2021_EVENTS
    ]
    return Table(rows=rows)


def filter_reference_events(
    *,
    instrument: str | None = None,
    date: str | None = None,
) -> Table:
    """Return a filtered subset of the Patel et al. (2021) reference table."""

    table = patel_2021_reference_table()
    mask = np.ones(len(table), dtype=bool)
    if instrument is not None:
        mask &= np.asarray(table["instrument"]) == instrument
    if date is not None:
        mask &= np.asarray(table["date"]) == date
    return table[mask]


def summarize_reference_coverage() -> dict[str, Any]:
    """Summarize the reference-event coverage by instrument."""

    table = patel_2021_reference_table()
    instruments = np.asarray(table["instrument"])
    unique, counts = np.unique(instruments, return_counts=True)
    return {
        "event_count": int(len(table)),
        "instruments": {instrument: int(count) for instrument, count in zip(unique.tolist(), counts.tolist(), strict=True)},
    }
