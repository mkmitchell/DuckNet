"""Image processing entry points for the DuckNet backend."""

import exif

from base.backend import GLOBALS


class ProcessingBusyError(RuntimeError):
    """The detector is busy training or processing another image."""


def process_image(imagepath, settings):
    """Run the active detection model on one image under the processing lock.

    Raises ProcessingBusyError instead of blocking when training or another
    request holds the lock, so a request never waits for a training run that
    may last hours.
    """
    if not GLOBALS.processing_lock.acquire(blocking=False):
        raise ProcessingBusyError(
            'The model is busy (training or processing another image); '
            'try again later.'
        )
    try:
        model = settings.models['detection']
        return model.process_image(imagepath)
    finally:
        GLOBALS.processing_lock.release()


def load_exif_datetime(filename: str) -> str:
    """EXIF capture time as 'YYYY:MM:DD HH:MM:SS', or '' when unavailable."""
    with open(filename, 'rb') as f:
        try:
            exif_f = exif.Image(f)
        except Exception:
            print('Could not load exif')
            return ''

        if exif_f.has_exif:
            if 'datetime_original' in exif_f.list_all():
                return exif_f.datetime_original
            elif 'datetime' in exif_f.list_all():
                return exif_f.datetime
        return ''
