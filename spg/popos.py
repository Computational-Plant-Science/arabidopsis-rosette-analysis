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


class LuminosityOptions:
    def __init__(self, threshold: float):
        if threshold <= 0 or threshold >= 1:
            raise ValueError(f"Luminosity threshold must be between 0 and 1")

        self.threshold = threshold


class EnhancementOptions:
    def __init__(self, brightness: int, gamma: int):
        self.brightness = brightness
        self.gamma = gamma


class CroppingOptions:
    def __init__(self, template: str):
        if not Path(template).is_file():
            raise ValueError(f"Template file does not exist: {template}")

        self.template = template


class SkeletonizationOptions:
    def __init__(self):
        pass


class SegmentationOptions:
    def __init__(self, color_space: str, channels: str, num_clusters: int):
        if color_space is None or color_space not in ['lab', 'bgr', 'hsv', 'ycc']:
            raise ValueError(f"Color space must be one of the following: 'lab', 'bgr', 'hsv', 'ycc'")

        if num_clusters < 1:
            raise ValueError(f"Must use at least 1 cluster for K-means clustering")

        self.color_space = color_space
        self.channels = channels
        self.num_clusters = num_clusters


class TraitsOptions:
    def __init__(
            self,
            input_images: List[InputImage],
            output_directory: str,
            multiprocessing: bool,
            luminosity_threshold: float,
            # brightness_enhancement: int,
            # gamma_enhancement: int,
            # marker_template: str,
            color_space: str,
            channels: str,
            num_clusters: int):
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
        self.multiprocessing = multiprocessing
        self.luminosity_options = LuminosityOptions(threshold=luminosity_threshold)
        # self.enhancement_options = EnhancementOptions(brightness=brightness_enhancement, gamma=gamma_enhancement)
        # self.cropping_options = CroppingOptions(template=marker_template)
        self.segmentation_options = SegmentationOptions(color_space=color_space, channels=channels, num_clusters=num_clusters)