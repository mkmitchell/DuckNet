"""DuckNet backend package: overrides of the vendored base backend.

The base app in base/backend/app.py does ``import backend`` at import time
and reads ``backend.settings`` and ``backend.processing`` when a request
arrives. Those attributes are bound when backend/app.py imports the
submodules; importing them here would close an import cycle with
base.backend, so this module only re-exports the shared lock.
"""

import sys

sys.path.append('.')  # so that 'base' resolves when run from the repo root

from base.backend import GLOBALS  # noqa: E402

__all__ = ['GLOBALS']
