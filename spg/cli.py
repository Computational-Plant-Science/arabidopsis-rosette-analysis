import time
from datetime import timedelta
from glob import glob
from os.path import join
from pathlib import Path

import click
import humanize
from tabulate import tabulate

from spg.luminosity import filter_dark_images
from spg.popos import InputImage, SPGOptions
from spg.traits import extract_traits
from spg.utils import write_traits_results_to_csv, write_traits_results_to_excel


@click.group()
def cli():
    pass


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-l', '--luminosity_threshold', required=False, type=float, default=0.1)
@click.option('-c', '--clusters', required=False, type=int, default=5)
@click.option('-m', '--multiprocessing', is_flag=True)
def extract(source, output_directory, file_types, luminosity_threshold, clusters, multiprocessing):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    start = time.time()

    if Path(source).is_file():
        options = SPGOptions([InputImage(input_path=source)], output_directory, luminosity_threshold, clusters, False)
        filter_dark_images(options)
        results = extract_traits(options)
    elif Path(source).is_dir():
        types = [ft.lower() for ft in file_types.split(',')]       # parse filetypes
        if 'jpg' in types:                                        # support different JPG extensions
            types.append('jpeg')
        if len(types) == 0:
            raise ValueError(f"You must specify file types!")
        types = types + [t.upper() for t in types]                # extension matching should also be case-insensitive
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in types), [])
        print(f"Found {len(files)} input files with extension in: {', '.join(types)}: \n" + '\n'.join(files))

        options = SPGOptions([InputImage(input_path=file) for file in files], output_directory, luminosity_threshold, clusters, multiprocessing)
        filter_dark_images(options)
        results = extract_traits(options)
    else:
        raise ValueError(f"Path does not exist: {source}")

    print(tabulate(results, headers="keys", tablefmt='orgtbl'))
    write_traits_results_to_csv(results, options.output_directory)
    write_traits_results_to_excel(results, options.output_directory)

    complete = time.time()
    print(f"Completed in about {humanize.naturaldelta(timedelta(seconds=round(complete - start)))}")


if __name__ == '__main__':
    cli()
