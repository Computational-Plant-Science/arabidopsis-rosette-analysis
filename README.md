# Smart Plant Growth: Leaf Segmentation & Traits

![CI](https://github.com/Computational-Plant-Science/spg-topdown-traits/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/spg-topdown-traits.svg)](https://badge.fury.io/py/spg-topdown-traits)

Robust, parameter-free leaf segmentation and trait extraction. Developed by **Suxing Liu**.

- Segment and analyze top-view images of individual plants or a whole tray.
- Robust segmentation based on color clustering method.
- Parameters configurable if desired, but not necessary.
- Extract individual plant geometric traits and write output to CSV/Excel files.

![Optional Text](../master/media/image_01.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Contents**

- [Using the Docker image](#using-the-docker-image)
- [Installing the Python package](#installing-the-python-package)
- [Installing from source](#installing-from-source)
- [CLI guide](#cli-guide)
  - [Traits](#traits)
    - [Results](#results)
  - [Tools](#tools)
    - [Enhance](#enhance)
    - [Luminosity](#luminosity)
    - [Crop](#crop)
    - [Skeleton](#skeleton)
    - [Segment](#segment)
  - [Optional parameters](#optional-parameters)
    - [Output directory](#output-directory)
    - [Luminosity threshold](#luminosity-threshold)
    - [Color space](#color-space)
    - [Channels](#channels)
    - [Clusters](#clusters)
    - [Multiprocessing](#multiprocessing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Using the Docker image

The easiest way to run this project is with [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/).

For instance, to pull the `computationalplantscience/spg` image, mount the current working directory, and open a Docker shell:

`docker run -it -v $(pwd):/opt/work -w /opt/work computationalplantscience/spg bash`

With Singularity:

`singularity shell docker://computationalplantscience/spg`

## Installing the Python package

Alternatively, you can install the PyPi package `spg` (e.g., with `pip install spg`) &mdash; however the package is not guaranteed compatible with any particular operating system or dependency configuration. Do this at your own risk.

## Installing from source

To configure a development environment or install this software from the source, first clone the repository with `git clone https://github.com/Computational-Plant-Science/spg.git`.

Then use `pip install -e .` from the project root to install the Python package locally, or run `docker build -t <tag> -f Dockerfile .` to build the Docker container.

## CLI guide

In a properly configured environment, the command `spg` will be available. The top-level CLI provides 2 main sub-commands: `traits` and `tools`.

### Traits

To extract traits from a directory of images with default parameters (suitable for typical use cases), run `spg traits <input file/directory>`. By default, output files will be deposited in the working directory.

This command invokes a parameter-free segmentation and trait extraction pipeline, involving the following steps:

- **luminosity detection**: average luminosity is calculated for each input image, and images with a value below a threshold will be omitted from analysis
- **segmentation**: leaves are segmented via K-means clustering
- **skeletonization**: a skeleton (medial axis) is derived for the entire plant
- **contour detection**: leaves are identified with contour detection
- **geometric trait extraction**: geometric traits are computed for each leaf (including area, solidity, maximum width/height, and average curvature, as well as total number of leaves)

#### Results

The `spg traits` command will produce a number of PNG images as well as 2 CSV files (and 2 corresponding Excel files with identical data, for convenience). Running `ls *.png *.csv *.xlsx` in an output directory should yield the following:

```shell
# for each input image...
<image>.clustered.png
<image>.masked.png
<image>.pie_color.png
<image>.result.1.png
<image>.result.2.png
...
<image>.result.N.png
<image>.curv.png
<image>.euclidean_graph_overlay.png
<image>.excontour.png
<image>.label.png
<image>.leaf_1.png
<image>.leaf_2.png
...
<image>.leaf_N.png
<image>.seg.png
<image>.skeleton.png
...
...
...
luminous_detection.csv
luminous_detection.xlsx
traits.csv
traits.xlsx
```

### Tools

Particular pipeline components can also be invoked independently. To invoke one, use `spg tools <tool> <input file/directory>`. By default, output files will be deposited in the working directory.

Available tools include:

- `luminosity`
- `enhance`
- `crop`
- `skeletonize`
- `segment`

#### Enhance

TODO

#### Luminosity

TODO

#### Crop

TODO

#### Skeleton

TODO

#### Segment

K-means color clustering segmentation. This is achieved by converting the source image to a desired color space and running K-means clustering on only the desired channels, with the pixels grouped into a desired number.

### Optional parameters

A number of parameters can be configured on the `spg traits` command (or on the appropriate `spg tools` command), if desired.

#### Output directory

To provide a path to a custom directory for output files and artifacts, use the `--output_directory` (`-o`) option. This option is available for `spg traits` as well as for all `spg tools`.

#### Luminosity threshold

The `--luminosity_threshold` option sets the lowest permissible average luminosity. For instance, use `-l 0.2` to set a luminosity threshold of 20%. Images darker (on average) than this will not be processed. The default value is `0.1`. This option is available for `spg traits` and `spg tools luminosity`.

#### Color space

The `--color_space` (`-s`) option configures the expected color space for input files. Permissible values are `lab`, `bgr`, `hsv`, or `ycc`. This option is available for `spg traits` and `spg tools segment`.

#### Channels

The `--channels` (`-c`) option selects channels indices to use for clustering, where 0 is the first channel, 1 is the second channel, and so on. For instance, if `bgr` color space is used, `-c 02` selects channels B and R. By default, this option is set to `all`, which is equivalent to `012`. This option is available for `spg traits` and `spg tools segment`.

#### Clusters

The `--num_clusters` (`-n`) option configures the number of K-means clusters to use when computing leaf contours. For instance, use `-n 5` for 5 clusters. The default value is `3`. This option is available for `spg traits` and `spg tools segment`.

#### Multiprocessing

To allow the `extract` command to process images in parallel if multiple cores are available, use the `--multiprocessing` (`-m`) flag. Multiprocessing is disabled by default. This option is available for `spg traits` as well as for all `spg tools`.
