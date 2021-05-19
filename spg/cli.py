import time
from datetime import timedelta
from glob import glob
from os.path import join
from pathlib import Path

import click
import humanize
from tabulate import tabulate

from spg.extraction import extract_traits
from spg.luminosity import filter_dark_images
from spg.popos import TraitsOptions, InputImage, SegmentationOptions
from spg.utils import write_traits_results_to_csv, write_traits_results_to_excel


@click.group()
def cli():
    pass


if __name__ == '__main__':
    cli()


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-m', '--multiprocessing', is_flag=True)
@click.option('-l', '--luminosity_threshold', required=False, type=float, default=0.1)
@click.option('-s', '--color_space', required=False, type=str, default='lab')
@click.option('-c', '--channels', required=False, type=str, default='1')
@click.option('-n', '--num_clusters', required=False, type=int, default='5')
def traits(source, output_directory, file_types, multiprocessing, luminosity_threshold, color_space, channels, num_clusters):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    start = time.time()

    if Path(source).is_file():
        options = TraitsOptions(
            input_images=[InputImage(input_path=source)],
            output_directory=output_directory,
            multiprocessing=False,
            luminosity_threshold=luminosity_threshold,
            color_space=color_space,
            channels=channels,
            num_clusters=num_clusters)

        filter_dark_images(options)
        results = extract_traits(options)
    elif Path(source).is_dir():
        types = [ft.lower() for ft in file_types.split(',')]  # parse filetypes
        if len(types) == 0: raise ValueError(f"No file types specified")  # must provide at least 1
        if 'jpg' in types: types.append('jpeg')  # support different JPG extensions
        types = types + [t.upper() for t in types]  # extension matching should be case-insensitive
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in types), [])
        print(f"Found {len(files)} input files with extension(s): {', '.join(types)}\n" + '\n'.join(files))

        options = TraitsOptions(
            input_images=[InputImage(input_path=file) for file in files],
            output_directory=output_directory,
            multiprocessing=multiprocessing,
            luminosity_threshold=luminosity_threshold,
            color_space=color_space,
            channels=channels,
            num_clusters=num_clusters)

        filter_dark_images(options)
        results = extract_traits(options)
    else:
        raise ValueError(f"Path does not exist: {source}")

    print(tabulate(results, headers="keys", tablefmt='orgtbl'))
    write_traits_results_to_csv(results, options.output_directory)
    write_traits_results_to_excel(results, options.output_directory)

    complete = time.time()
    print(f"Completed trait extraction in {humanize.naturaldelta(timedelta(seconds=round(complete - start)))}")


@cli.group()
def tools():
    pass


@tools.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-s', '--color_space', required=False, type=str, default='lab')
@click.option('-c', '--channels', required=False, type=str, default='1')
@click.option('-n', '--num_clusters', required=False, type=int, default='3')
@click.option('-m', '--multiprocessing', is_flag=True)
def segment(source, output_directory, file_types, color_space, channels, num_clusters, multiprocessing):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    start = time.time()

    if Path(source).is_file():
        options = SegmentationOptions(color_space=color_space, channels=channels, num_clusters=num_clusters)
        # TODO segment
    elif Path(source).is_dir():
        types = [ft.lower() for ft in file_types.split(',')]  # parse filetypes
        if len(types) == 0: raise ValueError(f"No file types specified")  # must provide at least 1
        if 'jpg' in types: types.append('jpeg')  # support different JPG extensions
        types = types + [t.upper() for t in types]  # extension matching should be case-insensitive
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in types), [])
        print(f"Found {len(files)} input files with extension(s): {', '.join(types)}\n" + '\n'.join(files))

        options = SegmentationOptions(color_space=color_space, channels=channels, num_clusters=num_clusters)
        # TODO segment
    else:
        raise ValueError(f"Path does not exist: {source}")

    complete = time.time()
    print(f"Completed segmentation in {humanize.naturaldelta(timedelta(seconds=round(complete - start)))}")
