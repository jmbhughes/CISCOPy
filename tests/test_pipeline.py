"""Tests for flexible input handling and the end-to-end pipeline."""

from __future__ import annotations

import os
import numpy as np
import pytest

os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/ciscopy-sunpy-config")

astropy = pytest.importorskip("astropy")
ndcube = pytest.importorskip("ndcube")

from astropy.io import fits
from astropy.time import Time
from astropy.time import TimeDelta
from astropy.wcs import WCS
from ndcube import NDCube

from ciscopy import CISCO
from ciscopy import SolarSequence
from ciscopy import detect_cme_regions
from ciscopy import downsample_sequence
from ciscopy import filter_reference_events
from ciscopy import fourier_motion_filter
from ciscopy import get_instrument_preset
from ciscopy import infer_instrument_preset
from ciscopy import load_fits_sequence
from ciscopy import main
from ciscopy import normalize_input
from ciscopy import patel_2021_reference_table
from ciscopy import preprocess_sequence
from ciscopy import summarize_reference_coverage
from ciscopy import write_height_time_map_svg
from ciscopy import write_cme_movie
from ciscopy import write_table
from ciscopy.cme import compute_cme_detection_debug
from ciscopy.cme import _parabolic_hough_candidates
from ciscopy.cme import ProcessedSequence
from ciscopy.sunpy_compat import prepare_legacy_solar_header


def _synthetic_headers(frame_count: int, *, shape: tuple[int, int]) -> list[fits.Header]:
    center_y, center_x = np.array(shape) // 2
    start = Time("2024-01-01T00:00:00")
    headers = []
    for index in range(frame_count):
        header = fits.Header()
        header["CRPIX1"] = center_x + 1
        header["CRPIX2"] = center_y + 1
        header["CDELT1"] = 20.0
        header["RSUN_OBS"] = 320.0
        header["RSUN_REF"] = 695700000.0
        header["INSTRUME"] = "AIA"
        header["WAVELNTH"] = 171
        header["DATE-OBS"] = (start + TimeDelta(index * 60, format="sec")).isot
        headers.append(header)
    return headers


def _synthetic_cme_cube(frame_count: int = 40, shape: tuple[int, int] = (128, 128)) -> tuple[np.ndarray, list[fits.Header]]:
    ny, nx = shape
    center_y, center_x = np.array(shape) // 2
    y_coords, x_coords = np.indices(shape)
    radial_distance = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    polar_angle = np.rad2deg(np.arctan2(center_x - x_coords, center_y - y_coords)) % 360.0

    cube = np.zeros((frame_count, ny, nx), dtype=float)
    background = 5.0 + 20.0 / (1.0 + radial_distance / 10.0)
    for index in range(frame_count):
        frame = background.copy()
        radius = 26.0 + 0.18 * (index - 8) ** 2 if index >= 8 else 22.0
        wedge = (polar_angle >= 88.0) & (polar_angle <= 102.0)
        ridge = np.exp(-0.5 * ((radial_distance - radius) / 1.8) ** 2)
        frame += 30.0 * wedge * ridge
        cube[index] = frame

    return cube, _synthetic_headers(frame_count, shape=shape)


def test_normalize_cube_with_single_header() -> None:
    cube = np.arange(24).reshape(3, 4, 2)
    header = fits.Header({"CRPIX1": 2.0, "CRPIX2": 3.0})

    sequence = normalize_input(cube, header=header)

    assert isinstance(sequence, SolarSequence)
    assert len(sequence) == 3
    assert sequence[0].header["CRPIX1"] == 2.0


def test_normalize_ndcube_list() -> None:
    wcs = WCS(naxis=2)
    cubes = [NDCube(np.ones((4, 4)) * idx, wcs=wcs) for idx in range(2)]

    sequence = normalize_input(cubes)

    assert len(sequence) == 2
    assert sequence[1].data[0, 0] == 1


def test_load_fits_sequence(tmp_path) -> None:
    first = tmp_path / "a.fits"
    second = tmp_path / "b.fits"

    for index, path in enumerate((first, second)):
        fits.PrimaryHDU(data=np.ones((5, 5)) * index).writeto(path)

    sequence = load_fits_sequence([first, second])

    assert len(sequence) == 2
    assert sequence.as_cube().shape == (2, 5, 5)


def test_load_fits_sequence_auto_detects_image_extension(tmp_path) -> None:
    primary_path = fits.PrimaryHDU()
    image_hdu = fits.ImageHDU(data=np.arange(25).reshape(5, 5))
    path = tmp_path / "auto_ext.fits"

    with fits.HDUList([primary_path, image_hdu]) as hdul:
        hdul.writeto(path, overwrite=True)

    sequence = load_fits_sequence([path])

    assert len(sequence) == 1
    assert sequence[0].meta["fits_ext"] == 1
    assert sequence[0].data[0, 0] == 0.0


def test_load_fits_sequence_can_downsample_during_loading(tmp_path) -> None:
    path = tmp_path / "downsampled_load.fits"
    header = fits.Header()
    header["CRPIX1"] = 129.0
    header["CRPIX2"] = 129.0
    header["CDELT1"] = 2.0
    header["CDELT2"] = 2.0
    fits.PrimaryHDU(data=np.arange(256 * 256, dtype=float).reshape(256, 256), header=header).writeto(path)

    sequence = load_fits_sequence([path], target_max_dim=128)

    assert sequence[0].data.shape == (128, 128)
    assert sequence[0].header["CDELT1"] == 4.0
    assert sequence[0].header["CRPIX1"] == 64.75


def test_cisco_characterize_returns_cme_table() -> None:
    cube, headers = _synthetic_cme_cube()
    pipeline = CISCO.from_input(cube, header=headers)

    table, processed = pipeline.characterize(r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)

    assert processed.motion_filtered_cube.shape[0] == 360
    assert len(table) >= 1
    row = table[0]
    assert row["date"] == "2024-01-01"
    assert 80.0 <= row["position_angle"] <= 110.0
    assert row["width"] >= 10.0
    assert row["speed"] > 0.0
    assert row["acceleration"] > 0.0


def test_fourier_motion_filter_preserves_shape_and_finite_output() -> None:
    cube = np.zeros((12, 16, 8), dtype=float)
    cube[4:7, 3:10, 2:6] = 1.0

    filtered = fourier_motion_filter(cube)

    assert filtered.shape == cube.shape
    assert np.isfinite(filtered).all()


def test_fourier_motion_filter_prefers_outbound_motion() -> None:
    cube = np.zeros((1, 32, 32), dtype=float)
    for time_index in range(6, 26):
        outbound_radius = 4 + time_index
        inbound_radius = 35 - time_index
        cube[0, outbound_radius, time_index] = 1.0
        cube[0, inbound_radius, time_index] = 1.0

    filtered = fourier_motion_filter(cube)[0]
    outbound_signal = np.mean([abs(filtered[4 + time_index, time_index]) for time_index in range(6, 26)])
    inbound_signal = np.mean([abs(filtered[35 - time_index, time_index]) for time_index in range(6, 26)])

    assert outbound_signal > inbound_signal


def test_parabolic_hough_candidates_detect_synthetic_parabola() -> None:
    height_time = np.zeros((32, 64), dtype=float)
    for time_index in range(10, 24):
        radius_index = int(round(0.4 * (time_index - 10) ** 2))
        if radius_index < height_time.shape[1]:
            height_time[time_index, radius_index] = 10.0

    accumulator, candidates = _parabolic_hough_candidates(
        height_time,
        cadence_seconds=60.0,
        km_per_pixel=1000.0,
    )

    assert accumulator.max() >= 8
    assert len(candidates) >= 1


def test_detect_cme_regions_finds_synthetic_event() -> None:
    cube, headers = _synthetic_cme_cube()
    sequence = normalize_input(cube, header=headers)
    processed = preprocess_sequence(sequence, r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)

    regions = detect_cme_regions(processed, sequence)

    assert len(regions) >= 1
    assert 80.0 <= regions[0]["position_angle"] <= 110.0
    assert regions[0]["width"] >= 10.0


def test_compute_cme_detection_debug_returns_adaptive_threshold_surface() -> None:
    cube, headers = _synthetic_cme_cube()
    sequence = normalize_input(cube, header=headers)
    processed = preprocess_sequence(sequence, r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)

    debug = compute_cme_detection_debug(processed, sequence)

    assert debug.threshold_map.shape == debug.binary_mask.shape
    assert debug.threshold_map.ndim == 2
    assert np.isfinite(debug.threshold_map).all()


def test_detect_cme_regions_discards_persistent_candidates_longer_than_8_hours() -> None:
    cube = np.ones((600, 64, 64), dtype=float)
    headers = _synthetic_headers(600, shape=(64, 64))
    sequence = normalize_input(cube, header=headers)

    processed = preprocess_sequence(sequence, r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)
    processed.motion_filtered_cube[120:135, :, 20:560] = 2.0

    regions = detect_cme_regions(
        processed,
        sequence,
        threshold_sigma=0.5,
        min_width_deg=6,
        min_area=20,
        min_speed_kms=50.0,
        max_speed_kms=2500.0,
        max_duration_hours=8.0,
    )

    assert regions == []


def test_detect_cme_regions_wraps_angles_across_360_boundary() -> None:
    headers = _synthetic_headers(40, shape=(64, 64))
    sequence = normalize_input(np.ones((40, 64, 64), dtype=float), header=headers)
    processed = ProcessedSequence(
        corrected_cube=np.zeros((40, 64, 64), dtype=float),
        polar_cube=np.zeros((40, 360, 32), dtype=float),
        polar_angles_deg=np.arange(360, dtype=float),
        polar_radius_pixels=np.linspace(10.0, 40.0, 32),
        polar_radius_rsun=np.linspace(1.2, 3.0, 32),
        height_time_cube=np.zeros((360, 32, 40), dtype=float),
        motion_filtered_cube=np.zeros((360, 32, 40), dtype=float),
        cadence_seconds=60.0,
        km_per_pixel=1000.0,
    )
    processed.motion_filtered_cube[358:360, :, 5:16] = 20.0
    processed.motion_filtered_cube[0:3, :, 5:16] = 20.0

    regions = detect_cme_regions(
        processed,
        sequence,
        threshold_sigma=0.5,
        min_width_deg=4,
        min_area=20,
        min_speed_kms=50.0,
        max_speed_kms=2500.0,
        max_duration_hours=8.0,
    )

    assert len(regions) == 1
    assert regions[0]["width"] >= 4.0
    assert regions[0]["position_angle"] in {0.0, 359.0, 1.0}


def test_downsample_sequence_updates_shape_and_header() -> None:
    cube, headers = _synthetic_cme_cube(frame_count=4, shape=(256, 256))
    sequence = normalize_input(cube, header=headers)

    reduced = downsample_sequence(sequence, target_max_dim=128)

    assert reduced[0].shape == (128, 128)
    assert reduced[0].header["CDELT1"] == sequence[0].header["CDELT1"] * 2
    assert reduced[0].header["CRPIX1"] < sequence[0].header["CRPIX1"]


def test_main_can_disable_resolution_reduction(tmp_path) -> None:
    cube, headers = _synthetic_cme_cube(frame_count=4, shape=(128, 128))
    output_path = tmp_path / "fullres_results.csv"

    result = main(
        cube,
        header=headers,
        reduce_resolution=False,
        output_path=output_path,
    )

    assert output_path.exists()
    assert result.processed.corrected_cube.shape[1:] == (128, 128)


def test_core_main_writes_csv_output(tmp_path) -> None:
    cube, headers = _synthetic_cme_cube()
    output_path = tmp_path / "cme_results.csv"

    result = main(
        cube,
        header=headers,
        r_min_rsun=1.2,
        r_max_rsun=3.0,
        radial_samples=96,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(result.table) >= 1
    text = output_path.read_text()
    assert "date,start_time,position_angle" in text


def test_write_table_supports_txt_output(tmp_path) -> None:
    cube, headers = _synthetic_cme_cube()
    pipeline = CISCO.from_input(cube, header=headers)
    table, _processed = pipeline.characterize(r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)
    output_path = tmp_path / "cme_results.txt"

    write_table(table, output_path)

    assert output_path.exists()
    text = output_path.read_text()
    assert "start_time" in text


def test_write_height_time_map_svg_outputs_overlay_plot(tmp_path) -> None:
    cube, headers = _synthetic_cme_cube()
    sequence = normalize_input(cube, header=headers)
    processed = preprocess_sequence(sequence, r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)
    output_path = tmp_path / "height_time.svg"

    written = write_height_time_map_svg(sequence, processed, output_path)

    assert written.exists()
    text = written.read_text()
    assert "<svg" in text
    assert "polyline" in text


def test_write_cme_movie_outputs_mp4(tmp_path) -> None:
    cube, headers = _synthetic_cme_cube(frame_count=40)
    sequence = normalize_input(cube, header=headers)
    processed = preprocess_sequence(sequence, r_min_rsun=1.2, r_max_rsun=3.0, radial_samples=96)
    output_path = tmp_path / "cme.mp4"

    written = write_cme_movie(sequence, processed, output_path)

    assert written.exists()
    assert written.stat().st_size > 0


def test_patel_2021_reference_table_contains_expected_rows() -> None:
    table = patel_2021_reference_table()

    assert len(table) == 21
    assert table[0]["instrument"] == "AIA (171 A)"
    assert table[-1]["serial_no"] == 21
    assert table[8]["remarks"] == "Decelerating eruption"


def test_filter_reference_events_and_summary() -> None:
    subset = filter_reference_events(instrument="SWAP (174 A)", date="2013-06-21")
    summary = summarize_reference_coverage()

    assert len(subset) == 2
    assert summary["event_count"] == 21
    assert summary["instruments"]["SWAP (174 A)"] == 7


def test_instrument_preset_lookup_and_inference() -> None:
    header = fits.Header()
    header["INSTRUME"] = "SWAP"
    header["WAVELNTH"] = 174

    explicit = get_instrument_preset("aia171")
    inferred = infer_instrument_preset(header)

    assert explicit.name == "aia171"
    assert inferred.name == "swap174"


def test_extended_preset_aliases_resolve() -> None:
    assert get_instrument_preset("LASCO C2").name == "lasco_c2"
    assert get_instrument_preset("LASCO/C3").name == "lasco_c3"
    assert get_instrument_preset("STEREO COR1").name == "cor1"
    assert get_instrument_preset("STEREO/COR2").name == "cor2"
    assert get_instrument_preset("Solar Orbiter FSI 174").name == "fsi174"
    assert get_instrument_preset("Solar Orbiter FSI 304").name == "fsi304"
    assert get_instrument_preset("Solar Orbiter METIS").name == "metis"
    assert get_instrument_preset("GOES SUVI").name == "suvi"
    assert get_instrument_preset("MLSO K-Coronagraph").name == "kcor"


def test_extended_header_inference_covers_coronagraphs() -> None:
    lasco = fits.Header()
    lasco["INSTRUME"] = "LASCO"
    lasco["DETECTOR"] = "C2"

    cor2 = fits.Header()
    cor2["INSTRUME"] = "SECCHI"
    cor2["DETECTOR"] = "COR2"

    fsi174 = fits.Header()
    fsi174["INSTRUME"] = "EUI"
    fsi174["DETECTOR"] = "FSI"
    fsi174["WAVELNTH"] = 174

    fsi304 = fits.Header()
    fsi304["INSTRUME"] = "EUI"
    fsi304["DETECTOR"] = "FSI"
    fsi304["WAVELNTH"] = 304

    metis = fits.Header()
    metis["TELESCOP"] = "METIS"

    suvi = fits.Header()
    suvi["INSTRUME"] = "SUVI"

    kcor = fits.Header()
    kcor["TELESCOP"] = "MLSO"

    assert infer_instrument_preset(lasco).name == "lasco_c2"
    assert infer_instrument_preset(cor2).name == "cor2"
    assert infer_instrument_preset(fsi174).name == "fsi174"
    assert infer_instrument_preset(fsi304).name == "fsi304"
    assert infer_instrument_preset(metis).name == "metis"
    assert infer_instrument_preset(suvi).name == "suvi"
    assert infer_instrument_preset(kcor).name == "kcor"


def test_unknown_header_stays_generic_for_flexible_inputs() -> None:
    unknown = fits.Header()
    unknown["INSTRUME"] = "CUSTOM"
    unknown["WAVELNTH"] = 999

    assert infer_instrument_preset(unknown).name == "generic"


def test_prepare_legacy_solar_header_warns_for_sparse_lasco_metadata() -> None:
    header = fits.Header()
    header["INSTRUME"] = "LASCO"
    header["DETECTOR"] = "C2"

    with pytest.warns(UserWarning, match="Legacy LASCO-style FITS metadata"):
        prepared = prepare_legacy_solar_header(header)

    assert prepared["RSUN_REF"] > 0.0
    assert prepared["RSUN_OBS"] > 0.0
    assert prepared["DSUN_OBS"] > 0.0


def test_main_accepts_explicit_preset(tmp_path) -> None:
    cube, headers = _synthetic_cme_cube()
    output_path = tmp_path / "preset_results.csv"

    result = main(
        cube,
        header=headers,
        preset="aia171",
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(result.table) >= 1


def test_main_rejects_single_image_input_for_kinematics() -> None:
    image = np.ones((64, 64), dtype=float)
    header = fits.Header()
    header["CRPIX1"] = 33
    header["CRPIX2"] = 33
    header["CDELT1"] = 20.0
    header["RSUN_OBS"] = 320.0
    header["RSUN_REF"] = 695700000.0
    header["DATE-OBS"] = "2024-01-01T00:00:00"

    with pytest.raises(ValueError, match="at least 3 time-ordered images"):
        main(image, header=header)


def test_pipeline_rejects_two_frame_sequence_for_kinematics() -> None:
    cube = np.ones((2, 64, 64), dtype=float)
    headers = _synthetic_headers(2, shape=(64, 64))
    pipeline = CISCO.from_input(cube, header=headers)

    with pytest.raises(ValueError, match="at least 3 time-ordered images"):
        pipeline.characterize()
