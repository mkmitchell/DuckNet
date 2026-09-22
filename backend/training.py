"""Training orchestration for the DuckNet detector.

Training runs in a background thread so the HTTP request that starts it returns
immediately. Progress and the final outcome are pushed to the browser over the
server-sent-events stream as 'training' events; the final event carries a
'status' of 'done', 'interrupted' or 'failed'.
"""

import os
import threading
import traceback

from base.backend import GLOBALS, pubsub

TRAINING_EVENT = 'training'
# Defaults mirror the Training tab inputs in templates/ducks/training_tab.html.
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.0005

_training_thread = None
_last_status = None


def is_training() -> bool:
    return _training_thread is not None and _training_thread.is_alive()


def training_status() -> dict:
    """Whether a job is running and the last final status event published.

    Lets the browser recover the outcome when the event stream dropped.
    """
    return {'running': is_training(), 'last': _last_status}


def start_training(
    imagefiles, targetfiles, training_options: dict, settings
) -> str:
    """Start a training job in a background thread and return 'STARTED'.

    Raises RuntimeError when a job is already running.
    """
    global _training_thread, _last_status
    if is_training():
        raise RuntimeError('Training is already running.')
    _last_status = None
    _training_thread = threading.Thread(
        target=_run_training_job,
        args=(imagefiles, targetfiles, training_options, settings),
        name='ducknet-training',
        daemon=True,
    )
    _training_thread.start()
    return 'STARTED'


def _run_training_job(
    imagefiles, targetfiles, training_options: dict, settings
) -> None:
    if not GLOBALS.processing_lock.acquire(blocking=False):
        publish_status(
            'failed',
            'Cannot start training: an image is still being processed.',
        )
        return
    try:
        print('Training options: ', training_options)
        model = settings.models['detection']
        # An empty active model name marks the in-memory model as unsaved.
        settings.active_models['detection'] = ''

        if not training_options.get('train_detector'):
            publish_status(
                'done', 'Nothing to train: detector training was not selected.'
            )
            return

        callback = create_training_progress_callback(
            desc='Training detector...'
        )
        callback(0.0)
        ok = model.start_training_detector(
            imagefiles,
            targetfiles,
            classes_of_interest=training_options.get('classes_of_interest'),
            negative_classes=training_options.get('classes_rejected', []),
            num_workers=0,
            callback=callback,
            epochs=training_options.get('epochs', DEFAULT_EPOCHS),
            lr=training_options.get('learning_rate', DEFAULT_LEARNING_RATE),
        )
        if ok:
            publish_status('done', 'Training finished')
        else:
            publish_status('interrupted', 'Training interrupted')
    except Exception as exc:
        traceback.print_exc()
        publish_status('failed', f'Training failed: {exc}')
    finally:
        GLOBALS.processing_lock.release()


def publish_status(status: str, description: str) -> None:
    global _last_status
    message = {'status': status, 'description': description}
    if status == 'done':
        message['progress'] = 1.0
    _last_status = message
    pubsub.PubSub.publish(message, event=TRAINING_EVENT)


def create_training_progress_callback(desc, scale=1, offset=0):
    def callback(progress, logs=None):
        pubsub.PubSub.publish(
            {'progress': progress * scale + offset, 'description': desc},
            event=TRAINING_EVENT,
        )

    return callback


def find_targetfiles(inputfiles):
    """Matching LabelMe json path per image, or None when there is none.

    Both 'image.jpg.json' and 'image.json' are accepted.
    """

    def find_targetfile(imgf):
        no_ext_imgf = os.path.splitext(imgf)[0]
        for f in [f'{imgf}.json', f'{no_ext_imgf}.json']:
            if os.path.exists(f):
                return f

    return list(map(find_targetfile, inputfiles))
