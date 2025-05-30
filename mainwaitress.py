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
    print("Starting mainwaitress.py...")
    
    try:
        print("Running CLI...")
        ok = CLI.run()
        print(f"CLI result: {ok}")
    except Exception as e:
        print(f"CLI error: {e}")
        ok = False

    if not ok:
        try:
            print("Initializing Flask app...")
            # Start UI with Waitress production server
            app = App()
            print("Flask app initialized successfully")
            
            # Set production environment
            os.environ['FLASK_ENV'] = 'production'
            
            # Reduce logging noise
            logging.getLogger('waitress').setLevel(logging.WARNING)
            
            print("Starting Waitress server on 0.0.0.0:5050...")
            serve(
                app, 
                host='0.0.0.0', 
                port=5050,
                threads=4,
                connection_limit=100,
                cleanup_interval=30,
                channel_timeout=120
            )
        except Exception as e:
            print(f"Error starting server: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("CLI command executed, not starting web server")