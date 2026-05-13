"""Diagnostic plotting helpers for CME candidate inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ciscopy.cme import CMEFitDiagnostic, ProcessedSequence, build_candidate_diagnostic
from ciscopy.presets import InstrumentPreset
from ciscopy.sequence import SolarSequence


def write_height_time_map_svg(
    sequence: SolarSequence,
    processed: ProcessedSequence,
    path: str | Path,
    *,
    preset: str | InstrumentPreset | None = None,
    candidate_index: int = 0,
    figure_width: int = 900,
    figure_height: int = 560,
) -> Path:
    """Write a height-time SVG with the best-fit CME ridge overlaid.

    Parameters
    ----------
    sequence
        Original normalized solar image sequence.
    processed
        Preprocessed outputs returned by the pipeline.
    path
        Output SVG path.
    preset
        Optional preset used when re-running the candidate selection for the
        diagnostic panel.
    candidate_index
        Index of the detected candidate to visualize.
    figure_width, figure_height
        Output SVG size in pixels.

    Returns
    -------
    pathlib.Path
        The written SVG path.
    """

    diagnostic = build_candidate_diagnostic(
        sequence,
        processed,
        preset=preset,
        candidate_index=candidate_index,
    )
    output_path = Path(path)
    output_path.write_text(
        _height_time_svg(diagnostic, figure_width=figure_width, figure_height=figure_height),
        encoding="utf-8",
    )
    return output_path


def _height_time_svg(
    diagnostic: CMEFitDiagnostic,
    *,
    figure_width: int,
    figure_height: int,
) -> str:
    margin_left = 80
    margin_right = 24
    margin_top = 64
    margin_bottom = 56
    panel_width = figure_width - margin_left - margin_right
    panel_height = figure_height - margin_top - margin_bottom

    image = np.asarray(diagnostic.height_time, dtype=float)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    vmin = float(np.percentile(finite, 5.0))
    vmax = float(np.percentile(finite, 99.0))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    scaled = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0)

    time_min = float(diagnostic.time_seconds[0]) if diagnostic.time_seconds.size else 0.0
    time_max = float(diagnostic.time_seconds[-1]) if diagnostic.time_seconds.size else 1.0
    radius_min = float(diagnostic.radius_rsun[0]) if diagnostic.radius_rsun.size else 0.0
    radius_max = float(diagnostic.radius_rsun[-1]) if diagnostic.radius_rsun.size else 1.0
    if np.isclose(time_min, time_max):
        time_max = time_min + 1.0
    if np.isclose(radius_min, radius_max):
        radius_max = radius_min + 1.0

    def x_scale(value: float) -> float:
        return margin_left + panel_width * (value - time_min) / (time_max - time_min)

    def y_scale(value: float) -> float:
        return margin_top + panel_height - panel_height * (value - radius_min) / (radius_max - radius_min)

    time_edges = np.linspace(time_min, time_max, scaled.shape[0] + 1)
    radius_edges = np.linspace(radius_min, radius_max, scaled.shape[1] + 1)

    cells: list[str] = []
    for time_index in range(scaled.shape[0]):
        x0 = x_scale(float(time_edges[time_index]))
        x1 = x_scale(float(time_edges[time_index + 1]))
        width = max(1.0, x1 - x0)
        for radius_index in range(scaled.shape[1]):
            y1 = y_scale(float(radius_edges[radius_index]))
            y0 = y_scale(float(radius_edges[radius_index + 1]))
            height = max(1.0, y1 - y0)
            shade = round(scaled[time_index, radius_index] * 255.0)
            cells.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" '
                f'fill="rgb({shade},{shade},{shade})" stroke="none" />'
            )

    ridge_points = " ".join(
        f"{x_scale(float(time_value)):.2f},{y_scale(float(radius_value)):.2f}"
        for time_value, radius_value in zip(diagnostic.ridge_time_seconds, diagnostic.ridge_radius_rsun, strict=False)
    )
    ridge_polyline = (
        f'<polyline fill="none" stroke="#e63946" stroke-width="3" points="{ridge_points}" />'
        if ridge_points
        else ""
    )

    x_ticks = np.linspace(time_min, time_max, 6)
    y_ticks = np.linspace(radius_min, radius_max, 6)
    x_tick_svg = "\n".join(
        (
            f'<line x1="{x_scale(float(value)):.2f}" y1="{margin_top + panel_height:.2f}" '
            f'x2="{x_scale(float(value)):.2f}" y2="{margin_top + panel_height + 6:.2f}" stroke="#222" />'
            f'<text x="{x_scale(float(value)):.2f}" y="{margin_top + panel_height + 22:.2f}" '
            f'font-size="12" text-anchor="middle" fill="#222">{value / 60.0:.1f}</text>'
        )
        for value in x_ticks
    )
    y_tick_svg = "\n".join(
        (
            f'<line x1="{margin_left - 6:.2f}" y1="{y_scale(float(value)):.2f}" '
            f'x2="{margin_left:.2f}" y2="{y_scale(float(value)):.2f}" stroke="#222" />'
            f'<text x="{margin_left - 10:.2f}" y="{y_scale(float(value)) + 4:.2f}" '
            f'font-size="12" text-anchor="end" fill="#222">{value:.2f}</text>'
        )
        for value in y_ticks
    )

    title = (
        f'Height-Time Map | Start {diagnostic.start_time} | '
        f'PA {diagnostic.position_angle:.1f} deg | Width {diagnostic.width:.1f} deg'
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{figure_width}" height="{figure_height}" viewBox="0 0 {figure_width} {figure_height}">
<rect x="0" y="0" width="{figure_width}" height="{figure_height}" fill="#ffffff" />
<text x="{margin_left}" y="30" font-size="20" font-family="Helvetica, Arial, sans-serif" fill="#111">{title}</text>
<text x="{margin_left}" y="48" font-size="13" font-family="Helvetica, Arial, sans-serif" fill="#555">Grayscale: filtered height-time intensity | Red: fitted ridge</text>
<rect x="{margin_left}" y="{margin_top}" width="{panel_width}" height="{panel_height}" fill="#f5f5f5" stroke="#222" stroke-width="1" />
{''.join(cells)}
{ridge_polyline}
<line x1="{margin_left}" y1="{margin_top + panel_height}" x2="{margin_left + panel_width}" y2="{margin_top + panel_height}" stroke="#222" />
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + panel_height}" stroke="#222" />
{x_tick_svg}
{y_tick_svg}
<text x="{margin_left + panel_width / 2:.2f}" y="{figure_height - 16}" font-size="14" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" fill="#111">Time Since Region Start (minutes)</text>
<text x="20" y="{margin_top + panel_height / 2:.2f}" font-size="14" font-family="Helvetica, Arial, sans-serif" fill="#111" transform="rotate(-90 20 {margin_top + panel_height / 2:.2f})">Heliocentric Distance (Rsun)</text>
</svg>
"""
