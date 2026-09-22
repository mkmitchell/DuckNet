"""Entry point: run the CLI when arguments are given, else serve the UI."""

import logging
import os

from waitress import serve

from backend.app import App
from backend.cli import CLI

# Each browser tab holds one worker thread for its SSE stream, so the pool
# must exceed the tab count: waitress' default 4 plus room for 4 tabs.
DEFAULT_THREADS = 8
# 0.0.0.0 is required inside Docker; publish with -p 127.0.0.1:5050:5050.
DEFAULT_HOST = '0.0.0.0'
PORT = 5050

if __name__ == '__main__':
    ok = CLI.run()

    if not ok:
        app = App()

        os.environ['FLASK_ENV'] = 'production'

        logging.getLogger('waitress').setLevel(logging.WARNING)
        logging.getLogger('waitress.queue').setLevel(logging.ERROR)

        serve(
            app,
            host=os.environ.get('DUCKNET_HOST', DEFAULT_HOST),
            port=PORT,
            threads=int(os.environ.get('DUCKNET_THREADS', DEFAULT_THREADS)),
            connection_limit=100,
            cleanup_interval=30,
            channel_timeout=120,
        )
