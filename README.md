# CISCOPy
[![codecov](https://codecov.io/gh/jmbhughes/CISCOPy/graph/badge.svg?token=mWszAYvBSV)](https://codecov.io/gh/jmbhughes/CISCOPy)

[CIISCO](https://github.com/s0larish/ciisco) in Python with generalizations to more data.

CISCOPy is a Python package for CIISCO/CISCO-style analysis of solar image sequences. It is being
built from the legacy IDL implementation together with the method described by Patel et al. (2021),
using standard scientific Python tools where possible.

## Installation

Standard install:

```bash
python -m pip install .
```

Install from requirements:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Development install:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The package accepts:

- a list of FITS file paths
- a 2D image or 3D image cube as a NumPy array
- an image cube plus FITS headers and/or WCS objects
- a single `ndcube.NDCube` or a list of `NDCube` objects

Implementation choices in this package favor existing, standard tools:

- `sunpy.map` for solar-map aware data handling when FITS headers are available
- SunPy helioprojective-radial reprojection for solar-north-aware Cartesian-to-polar conversion when the map stack is available
- `astropy` for FITS/WCS/time/table support
- `scipy.ndimage` for interpolation, filtering, morphology, and edge detection
- `numpy` for array processing and FFT-based motion filtering

## Current API

```python
import ciscopy

result = ciscopy.main(
    cube,
    header=headers,
    output_path="cme_results.csv",
)
table = result.table
```

Available utilities:

- `ciscopy.main`: top-level entry point that runs the pipeline and optionally writes output
- `ciscopy.write_table`: write results to `.csv`, `.ecsv`, `.tsv`, or `.txt`
- `ciscopy.write_cme_movie`: write a compact MP4 movie for a detected candidate
- `ciscopy.downsample_sequence`: reduce large image sequences before analysis
- `ciscopy.normalize_input`: normalize FITS paths, arrays, headers/WCS, and `ndcube`
- `ciscopy.load_fits_sequence`: load a list of FITS files into a common sequence container
- `ciscopy.mask_disk`: mask the solar disk using header metadata or explicit scaling
- `ciscopy.polar_transform`: convert a Cartesian image into a polar map using `scipy.ndimage.map_coordinates`
- `ciscopy.azimuthal_radial_intensity`: compute an annular radial intensity profile
- `ciscopy.CISCO.from_input(...).characterize()`: run the end-to-end CME characterization pipeline

Preset support includes:

- `AIA` 171 and 304 A
- `Solar Orbiter FSI` 174 and 304 A
- `SWAP` 174 A
- `EUVI` 304 A
- `LASCO` C2 and C3
- `STEREO` COR1 and COR2
- `Solar Orbiter` METIS
- `GOES` SUVI
- `MLSO` K-Coronagraph

Unknown instruments still work through the generic path as long as the data can
be represented with image arrays plus reasonable FITS headers and/or WCS.
For instruments like Solar Orbiter FSI, where the apparent solar size changes
with spacecraft distance, the preset stays in solar-radii units and the actual
pixel-space radial bounds are computed from the image metadata at runtime.
By default, the pipeline also downsamples large images because the algorithm is
designed for large-scale structures; this can be turned off manually.
When FITS paths are used, the loader checks the primary HDU first and then
automatically falls back to the first 2D image extension if the primary HDU is
empty, which is useful for products such as Solar Orbiter EUI/FSI.
For older LASCO-style headers with sparse solar metadata, the package warns and
fills standard solar-radius / Earth-observer assumptions so processing can
continue.

## End-to-End Example

```python
from ciscopy import main

result = main(
    fits_paths,
    preset="lasco_c2",
    output_path="cme_results.csv",
    make_movie=True,
    movie_output_path="cme_event.mp4",
    reduce_resolution=True,
)
table = result.table
processed = result.processed

print(table["date", "start_time", "position_angle", "width", "speed", "acceleration"])
```

Supported text output formats:

- `.csv`
- `.ecsv`
- `.tsv`
- `.txt`

Movie output:

- `.mp4` at compact 256x256 rendering by default
- overlays the detected position angle, angular width, and event time span
- uses the fitted height-time ridge to determine the event end time

The output table contains:

- `date`
- `start_time`
- `position_angle`
- `width`
- `speed`
- `del_speed`
- `speed_min`
- `speed_max`
- `acceleration`
- `del_acceleration`
- `acceleration_min`
- `acceleration_max`

## Scientific Reference

The method is based on Patel et al. 2021:

- [Automated Detection and Characterization of Coronal Mass Ejections in the Inner Corona](https://ui.adsabs.harvard.edu/abs/2021SoPh..296...31P/abstract)

## Pipeline Outline

The current end-to-end pipeline follows the same broad stages as the IDL workflow and paper:

1. Normalize inputs into a common sequence container.
2. Estimate a minimum background and a radially symmetric background.
3. Apply disk masking and intensity normalization.
4. Convert the sequence to polar coordinates, preferring SunPy's
   helioprojective-radial reprojection when available.
5. Build height-time maps and apply Fourier motion filtering.
6. Detect candidate CME regions in angle-time space.
7. Estimate CME kinematics with a parabolic Hough-style characterization step.

Polar-angle convention:

- solar north is `0` degrees
- angles increase counterclockwise
