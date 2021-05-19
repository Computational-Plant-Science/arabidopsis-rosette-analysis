import cv2
import numpy as np
from tabulate import tabulate

from spg.popos import InputImage, SPGOptions, LuminosityResult
from spg.utils import write_luminosity_results_to_csv, write_luminosity_results_to_excel


def check_luminosity(input_image: InputImage, threshold: float):
    orig = cv2.imread(input_image.path)
    copy = orig.copy()

    # Convert color space to LAB format and extract L channel
    L, A, B = cv2.split(cv2.cvtColor(copy, cv2.COLOR_BGR2LAB))

    # Normalize L channel by dividing all pixel values with maximum pixel value
    normalized = float(np.mean(L / np.max(L)))

    return input_image.name, normalized, "bright" if normalized > threshold else "dark"


def filter_dark_images(options: SPGOptions) -> SPGOptions:
    print(f"Filtering images with average luminosity below {options.luminosity_threshold}")
    results = []

    for input_image in options.input_images:
        name, avg, decision = check_luminosity(input_image, options.luminosity_threshold)
        results.append(LuminosityResult(name=name, avg=avg, decision=decision))

        # luminosity_str is either 'dark' or 'bright'
        if decision == 'dark':
            options.input_images.remove(input_image)

    print(tabulate(results, headers="keys", tablefmt='orgtbl'))
    write_luminosity_results_to_csv(results, options.output_directory)
    write_luminosity_results_to_excel(results, options.output_directory)

    return options
