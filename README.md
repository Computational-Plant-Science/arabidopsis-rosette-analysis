# Smart Plant Growth: Leaf Segmentation & Traits

Author: Suxing Liu

![CI](https://github.com/Computational-Plant-Science/spg-topdown-traits/workflows/CI/badge.svg)
[![PyPI version](https://badge.fury.io/py/spg-topdown-traits.svg)](https://badge.fury.io/py/spg-topdown-traits)

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image, this tool will segment it into individual images.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant gemetrical traits, and write output into excel file.

![Optional Text](../master/media/image_01.png)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Contents**

- [Requirements](#requirements)
- [Usage](#usage)
  - [Multiprocessing](#multiprocessing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements

The easiest way to run this project in a Unix environment is with [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/).

For instance, to pull the `computationalplantscience/spg` image, mount the current working directory, and open a shell:

`docker run -it -v $(pwd):/opt/dev -w /opt/dev computationalplantscience/spg bash`

Singularity users:

`singularity shell docker://computationalplantscience/spg-topdown-traits`

## Usage

A typical use case might look like:

`spg extract <input directory> -o <output directory> -l 0.1 -t /opt/spg-topdown-traits/marker_template.png -m`

#### Output directory

By default, output files will be written to the current working directory. To provide a different path, use the `-o` option.

#### Luminosity threshold

The `-l 0.1` option sets a luminosity threshold of 10%. Images darker than this will not be processed.

#### Marker template

You must provide a marker template image to use `spg-topdown-traits`. By default, an image named `marker_template.png` is expected in the working directory. You can also provide a different image path with the `-t (--template)` argument. A template is provided in the Docker image at `/opt/spg-topdown-traits/marker_template.png`.

#### Multiprocessing

To allow the `extract` command to process images in parallel if multiple cores are available, use the `-m` flag.
