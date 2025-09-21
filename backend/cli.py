import base.backend.cli as base_cli
from base.backend.app import path_to_main_module

import os, sys
from backend.processing import load_exif_datetime
from backend.settings import parse_species_codes_file

class CLI(base_cli.CLI):

    #override
    @classmethod
    def create_parser(cls): # type: ignore
        parser = super().create_parser(
            description    = '🦆 DuckNet 🦆',
            default_output = 'detected_ducks.csv',
        )
        parser.add_argument('--saveboxes',  action='store_const', const=True, default=False,
                            help='Include boxes of detected ducks in the output')
        return parser

    #override
    @classmethod
    def write_results(cls, results:list, args):
        filename   = args.output.as_posix()
        outputfile = open(filename, 'w')
        outputfile.write(results_to_csv(results, args.saveboxes))

def results_to_csv(results, export_boxes=False):
    header = [
        'Filename', 'Date', 'Time', 'Class', 'Confidence level'
    ]
    if export_boxes:
        header.append('Box')

    species_codes_file = os.path.join(path_to_main_module(), 'species_codes.txt')
    species_codes = parse_species_codes_file(path=species_codes_file)
    csv_data      = []
    for r in results:
        filename       = os.path.basename(r['filename'])
        result         = r['result']

        selectedlabels = result['labels']
        datetime       = load_exif_datetime(r['filename'])
        date,time      = datetime.split(' ')[:2] if datetime is not None and ' ' in datetime else ['',''] 
        date           = date.replace(':','.')

        n              = len(result['labels'])
    

        if n==0:
            csv_item   = [filename, date, time, '', ''] + ([''] if export_boxes else [])
            csv_data.append( csv_item )
        
        for i in range(len(selectedlabels)):
            label      = selectedlabels[i]
            confidence = result['per_class_scores'][i][label]
            code      = species_codes.get(label, '')
                                                             #TODO: custom threshold
            
            confidence_str = f'{confidence*100:.1f}'
            csv_item       = [filename, date, time, code, confidence_str]
            if export_boxes:
                box  = ' '.join( [ f'{x:.1f}' for x in result['boxes'][i] ] )
                csv_item.append(box)
            csv_data.append(csv_item)
        
        #sanity check
        all_ok = all( [len(item)==len(header) for item in csv_data ] )
        if not all_ok:
            print(f'[INTERNAL ERROR] inconsistent CSV data', file=sys.stderr)
            #return   #no return
        
    csv_data = [header] + csv_data
    csv_txt  = ''
    csv_txt  = ';\n'.join( [ ';'.join(x) for x in csv_data ] ) + ';\n'
    return csv_txt