"""Vendored DigIT base backend: shared lock plus the base app modules."""

import threading


class GLOBALS:
    processing_lock = threading.RLock()


# These modules import GLOBALS from this package, so they must follow it.
from . import processing, pubsub, settings  # noqa: E402

__all__ = ['GLOBALS', 'processing', 'pubsub', 'settings']
