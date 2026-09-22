"""Command line interface: process images and write a CSV of detections."""

import os
import sys

import base.backend.cli as base_cli
from backend.processing import load_exif_datetime
from backend.settings import parse_species_codes_file
from base.backend.app import path_to_main_module


class CLI(base_cli.CLI):
    # override
    @classmethod
    def create_parser(cls):  # type: ignore
        parser = super().create_parser(
            description='DuckNet',
            default_output='detected_ducks.csv',
        )
        parser.add_argument(
            '--saveboxes',
            action='store_const',
            const=True,
            default=False,
            help='Include boxes of detected ducks in the output',
        )
        return parser

    # override
    @classmethod
    def write_results(cls, results: list, args):
        with open(args.output.as_posix(), 'w') as outputfile:
            outputfile.write(results_to_csv(results, args.saveboxes))


def species_code_for(label: str, species_codes: dict) -> str:
    """Four-letter species code for a model label.

    The model's labels are already codes (see class_list.txt in the packaged
    model), so the label is returned as is unless species_codes.txt maps it
    from a scientific name.
    """
    return species_codes.get(label, label)


def results_to_csv(results, export_boxes=False):
    """Render CLI results as the semicolon-separated CSV from the README.

    Columns: Filename, Date, Time, Class (species code), Confidence level
    (0 to 1) and, with export_boxes, Box as 'x0 y0 x1 y1'. Every line ends
    with a trailing separator, matching the CSV the browser download makes.
    """
    header = ['Filename', 'Date', 'Time', 'Class', 'Confidence level']
    if export_boxes:
        header.append('Box')

    species_codes_file = os.path.join(
        path_to_main_module(), 'species_codes.txt'
    )
    species_codes = parse_species_codes_file(path=species_codes_file)
    csv_data = []
    for r in results:
        filename = os.path.basename(r['filename'])
        result = r['result']

        selectedlabels = result['labels']
        datetime = load_exif_datetime(r['filename'])
        date, time = (
            datetime.split(' ')[:2]
            if datetime is not None and ' ' in datetime
            else ['', '']
        )
        date = date.replace(':', '.')

        if len(selectedlabels) == 0:
            csv_item = [filename, date, time, '', ''] + (
                [''] if export_boxes else []
            )
            csv_data.append(csv_item)

        for i in range(len(selectedlabels)):
            label = selectedlabels[i]
            confidence = result['per_class_scores'][i][label]
            code = species_code_for(label, species_codes)

            csv_item = [filename, date, time, code, f'{confidence:.2f}']
            if export_boxes:
                box = ' '.join([f'{x:.1f}' for x in result['boxes'][i]])
                csv_item.append(box)
            csv_data.append(csv_item)

    if not all(len(item) == len(header) for item in csv_data):
        print('[INTERNAL ERROR] inconsistent CSV data', file=sys.stderr)

    csv_data = [header] + csv_data
    csv_txt = ';\n'.join([';'.join(x) for x in csv_data]) + ';\n'
    return csv_txt
