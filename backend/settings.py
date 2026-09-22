"""DuckNet settings: base settings plus threshold, flag and export options."""

import os

from base.backend.app import path_to_main_module
from base.backend.settings import Settings as BaseSettings


class Settings(BaseSettings):
    # override
    @classmethod
    def get_defaults(cls):
        d = super().get_defaults()
        d.update(
            {
                'confidence_threshold': 50,
                'flag_negatives': True,
                'export_boxes': True,
            }.items()
        )
        return d

    # override
    def get_settings_as_dict(self):
        s = super().get_settings_as_dict()
        s['species_codes'] = parse_species_codes_file()
        return s


DEFAULT_SPECIES_FILE = os.path.join(path_to_main_module(), 'species_codes.txt')


def parse_species_codes_file(path=DEFAULT_SPECIES_FILE) -> dict:
    """Read 'scientific name : CODE' lines into a dict mapping name to code."""
    with open(path) as f:
        lines = f.read().strip().split('\n')
    return dict([map(str.strip, line.split(':')) for line in lines])  # type: ignore
