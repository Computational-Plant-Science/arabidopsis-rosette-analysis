"""
plain old Python objects
"""

from pathlib import Path
from typing import List
from typing import TypedDict


class InputImage:
    def __init__(self, input_path):
        path = Path(input_path)
        if not path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        self.path = input_path
        self.name = Path(input_path).name
        self.stem = Path(input_path).stem


class LuminosityResult(TypedDict, total=True):
    name: str
    avg: float
    decision: str


class TraitsResult(TypedDict, total=True):
    name: str
    failed: bool
    area: float
    solidity: float
    max_width: int
    max_height: int
    avg_curvature: float
    leaves: int


class SPGOptions:
    def __init__(self, input_images: List[InputImage], output_directory: str, luminosity_threshold: float, clusters: int, multiprocessing: bool):
        if len(input_images) == 0:
            raise ValueError(f"No input images provided")

        output_path = Path(output_directory)
        if not output_path.exists():
            raise ValueError(f"Output path does not exist: {output_directory}")
        elif not output_path.is_dir():
            raise ValueError(f"Output path is not a directory: {output_directory}")

        if luminosity_threshold <= 0 or luminosity_threshold >= 1:
            raise ValueError(f"Luminosity threshold must be between 0 and 1")

        self.input_images = input_images
        self.output_directory = output_directory
        self.luminosity_threshold = luminosity_threshold
        self.clusters = clusters
        self.multiprocessing = multiprocessing
