import argparse, pathlib, glob, os, sys

import backend.settings



class CLI:
    '''Command line interface'''

    @staticmethod
    def create_parser(
        description    = 'Command line interface',
        default_output = 'results.csv',
    ):
        parser = argparse.ArgumentParser(
            description = description,
            epilog = 'If used without arguments, the user interface is started.'
        )
        parser.add_argument('--input', type=pathlib.Path, default=None,
                            help='Path to input images (e.g. --input=path/to/*.jpg)')
        parser.add_argument('--output', type=pathlib.Path, default=default_output,
                            help=f'Path to output file (e.g. --output={default_output})')
        parser.add_argument('--model',  type=pathlib.Path,
                            help='Path to model file (default: last used)')
        return parser

    @classmethod
    def process_cli_args(cls, args):
        inputfiles = sorted(glob.glob(args.input.as_posix(), recursive=True))
        if len(inputfiles) == 0:
            print('Could not find any files')
            return
        
        import backend.settings
        settings = backend.settings.Settings()

        if args.model:
            #TODO: code duplication
            modelpath = args.model.as_posix()
            if not os.path.exists(modelpath):
                print(f'[ERROR] File "{modelpath}" does not exist')
                return 1
            model = settings.load_modelfile(modelpath)
            settings.models['detection'] = model
        

        import backend.processing  #FIXME
        print(f'Processing {len(inputfiles)} files')
        results = []
        for i,f in enumerate(inputfiles):
            print(f'[{i:4d} / {len(inputfiles)}] {f}')
            try:
                result       = backend.processing.process_image(f, settings)
            except Exception as e:
                print(f'[ERROR] {e}', file=sys.stderr)
                continue
            results += [{'filename':f, 'result':result}]
        
        if len(results)==0:
            print(f'[ERROR] Unable to process any file', file=sys.stderr)
            return
        
        cls.write_results(results, args)
        print(f'Results written to {args.output.as_posix()}.')

    @classmethod
    def write_results(cls, results:list, args):
        print('[NOT IMPLEMENTED] output to file be implemented downstream')
        #outputfile = open(filename, 'w')
        #outputfile.write(results_to_csv(results, args.saveboxes))
        

    @classmethod
    def run(cls):
        args = cls.create_parser().parse_args()
        if args.input:
            cls.process_cli_args(args)
            return True
        else:
            return False

