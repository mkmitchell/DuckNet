from backend.app import App
from backend.cli import CLI
from waitress import serve
import os
import logging

from backend.app import App
from backend.cli import CLI
from waitress import serve
import os
import logging
import sys

if __name__ == '__main__':
    ok = CLI.run()

    if not ok:
        # Start UI with Waitress production server
        app = App()
        
        # Set production environment
        os.environ['FLASK_ENV'] = 'production'
        
        # Reduce logging noise
        logging.getLogger('waitress').setLevel(logging.WARNING)
        
        serve(
            app, 
            host='0.0.0.0', 
            port=5050,
            threads=4,
            connection_limit=100,
            cleanup_interval=30,
            channel_timeout=120
        )