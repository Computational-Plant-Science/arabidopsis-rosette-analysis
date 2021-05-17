from glob import glob
from os.path import join
from pathlib import Path

import click

from spg.luminosity_detection import filter_dark_images
from spg.popos import InputImage, SPGOptions
from spg.extract import trait_extract
from spg.utils import write_traits_results_to_csv


@click.group()
def cli():
    pass


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-l', '--luminosity_threshold', required=False, type=float, default=0.1)
@click.option('-m', '--multiprocessing', is_flag=True)
def extract(source, output_directory, file_types, luminosity_threshold, multiprocessing):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        options = SPGOptions([InputImage(input_path=source)], output_directory, luminosity_threshold, False)

        # run analysis
        filter_dark_images(options)
        traits_results = trait_extract(options)
        write_traits_results_to_csv(traits_results, options.output_directory)
    elif Path(source).is_dir():
        # parse filetypes
        fts = [ft.lower() for ft in file_types.split(',')]

        # support different JPG extensions
        if 'jpg' in fts:
            fts.append('jpeg')
        if len(fts) == 0:
            raise ValueError(f"You must specify file types!")

        # extension matching should also be case-insensitive
        fts = fts + [pattern.upper() for pattern in fts]

        # find input files
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in fts), [])
        inputs = [InputImage(input_path=file) for file in files]
        options = SPGOptions(inputs, output_directory, luminosity_threshold, multiprocessing)
        print(f"Found {len(files)} input files with extensions {', '.join(fts)}: \n" + '\n'.join(files))

        # run analysis
        filter_dark_images(options)
        traits_results = trait_extract(options)
        write_traits_results_to_csv(traits_results, options.output_directory)
    else:
        raise ValueError(f"Path does not exist: {source}")


if __name__ == '__main__':
    cli()
