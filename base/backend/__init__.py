import threading

class GLOBALS:
    processing_lock = threading.RLock()


from . import settings
from . import processing
from . import pubsub
