# Smart Plant Growth: Leaf Segmentation & Traits

![CI](https://github.com/Computational-Plant-Science/spg-topdown-traits/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/spg-topdown-traits.svg)](https://badge.fury.io/py/spg-topdown-traits)

Robust, parameter-free leaf segmentation and trait extraction. Developed by Suxing Liu.

- Segment and analyze top-view images of individual plants or a whole tray.
- Robust segmentation based on color clustering method.
- Parameters configurable if desired, but not necessary.
- Extract individual plant geometric traits and write output to CSV/Excel files.

![Optional Text](../master/media/image_01.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Contents**

- [Requirements](#requirements)
- [Usage](#usage)
  - [Parameter-free](#parameter-free)
  - [With (optional) parameters](#with-optional-parameters)
    - [Output directory](#output-directory)
    - [Luminosity threshold](#luminosity-threshold)
    - [Clusters](#clusters)
    - [Multiprocessing](#multiprocessing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements

The easiest way to run this project in a Unix environment is with [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/).

For instance, to pull the `computationalplantscience/spg` image, mount the current working directory, and open a Docker shell:

`docker run -it -v $(pwd):/opt/dev -w /opt/dev computationalplantscience/spg bash`

With Singularity:

`singularity shell docker://computationalplantscience/spg`

## Usage

### Parameter-free

To extract traits from a directory of images with default parameters (suitable for typical use cases), run `spg extract <input directory>`.

By default, output files will be deposited in the working directory.

### With (optional) parameters

A number of parameters can be configured if desired.

#### Output directory

To provide a path to a custom directory for output files and artifacts, use the `--output_directory` (`-o`) option.

#### Luminosity threshold

The `--luminosity_threshold` option sets the lowest permissible average luminosity. For instance, use `-l 0.2` to set a luminosity threshold of 20%. Images darker (on average) than this will not be processed. The default value is `0.1`.

#### Clusters

The `--clusters` (`-c`) option configures the number of K-means clusters to use when computing leaf contours. For instance, use `-c 5` for 8 clusters. The default value is `5`.

#### Multiprocessing

To allow the `extract` command to process images in parallel if multiple cores are available, use the `--multiprocessing` (`-m`) flag. Multiprocessing is disabled by default.
