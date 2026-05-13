"""Movie helpers for compact CME event visualization."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from scipy.ndimage import zoom

from ciscopy.cme import ProcessedSequence, build_candidate_diagnostic
from ciscopy.presets import InstrumentPreset
from ciscopy.sequence import SolarSequence


def write_cme_movie(
    sequence: SolarSequence,
    processed: ProcessedSequence,
    path: str | Path,
    *,
    preset: str | InstrumentPreset | None = None,
    candidate_index: int = 0,
    size_px: int = 256,
    fps: int = 8,
) -> Path:
    """Write an MP4 movie for one detected CME candidate.

    The movie spans the fitted ridge visibility interval and overlays the
    detected central position angle and angular width on each frame.
    """

    diagnostic = build_candidate_diagnostic(
        sequence,
        processed,
        preset=preset,
        candidate_index=candidate_index,
    )
    output_path = Path(path)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to write MP4 movies but was not found on PATH.")

    with tempfile.TemporaryDirectory(prefix="ciscopy-movie-") as tmpdir:
        temp_dir = Path(tmpdir)
        for frame_index in range(diagnostic.start_index, diagnostic.end_index + 1):
            frame = sequence[frame_index]
            image = _resample_frame(frame.data, size_px)
            frame_path = temp_dir / f"frame_{frame_index - diagnostic.start_index:04d}.png"
            _write_frame_png(
                image,
                frame_path,
                position_angle=diagnostic.position_angle,
                width=diagnostic.width,
                time_text=frame.time.isot if frame.time is not None else f"frame-{frame_index:04d}",
                start_time=diagnostic.start_time,
                end_time=diagnostic.end_time,
            )

        command = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir / "frame_%04d.png"),
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True)
    return output_path


def _resample_frame(image: np.ndarray, size_px: int) -> np.ndarray:
    zoom_factors = (size_px / image.shape[0], size_px / image.shape[1])
    return zoom(np.asarray(image, dtype=float), zoom_factors, order=1)


def _write_frame_png(
    image: np.ndarray,
    path: Path,
    *,
    position_angle: float,
    width: float,
    time_text: str,
    start_time: str,
    end_time: str,
) -> None:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        finite = np.array([0.0, 1.0])
    vmin = float(np.percentile(finite, 5.0))
    vmax = float(np.percentile(finite, 99.5))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    center_x = (image.shape[1] - 1) / 2.0
    center_y = (image.shape[0] - 1) / 2.0
    radius = 0.45 * min(image.shape)
    theta_center = (90.0 - position_angle) % 360.0
    theta1 = theta_center - width / 2.0
    theta2 = theta_center + width / 2.0
    wedge = Wedge(
        (center_x, center_y),
        radius,
        theta1,
        theta2,
        width=radius * 0.22,
        facecolor=(1.0, 0.2, 0.2, 0.18),
        edgecolor="#ff4d4d",
        linewidth=2.0,
    )
    ax.add_patch(wedge)
    ax.text(
        6,
        image.shape[0] - 10,
        f"t={time_text}",
        color="white",
        fontsize=7,
        ha="left",
        va="top",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
    )
    ax.text(
        6,
        8,
        f"PA={position_angle:.1f}  W={width:.1f}\n{start_time} to {end_time}",
        color="white",
        fontsize=6,
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
    )
    ax.set_axis_off()
    fig.savefig(path, dpi=100)
    plt.close(fig)
