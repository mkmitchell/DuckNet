import os
from base.backend import pubsub
from base.backend import GLOBALS

def start_training(imagefiles, targetfiles, training_options:dict, settings):
    locked = GLOBALS.processing_lock.acquire(blocking=False)
    if not locked:
        raise RuntimeError('Cannot start training. Already processing.')

    print('Training options: ', training_options)

    try:
        model = settings.models['detection']
        # Indicate that the current model is unsaved
        settings.active_models['detection'] = ''
        
        ok = True
        if training_options.get('train_detector'):
            cb = create_training_progress_callback(
                desc='Training detector...', 
                scale=1, 
                offset=0,
            )
            cb(0.0)  # Set to zero at the beginning

            ok = model.start_training_detector(
                imagefiles, 
                targetfiles, 
                negative_classes=training_options.get('classes_rejected', []),
                num_workers=0, 
                callback=cb,
                epochs=training_options.get('epochs', 10),
                lr=training_options.get('learning_rate', 0.001),
            )

        return 'OK' if ok else 'INTERRUPTED'
    finally:
        GLOBALS.processing_lock.release()

def create_training_progress_callback(desc, scale=1, offset=0):
    def callback(progress, logs=None):
        # only drive the front-end
        pubsub.PubSub.publish(
            {'progress': progress*scale + offset, 'description': desc},
            event='training'
        )
    return callback

def find_targetfiles(inputfiles):
    def find_targetfile(imgf):
        no_ext_imgf = os.path.splitext(imgf)[0]
        for f in [f'{imgf}.json', f'{no_ext_imgf}.json']:
            if os.path.exists(f):
                return f
    return list(map(find_targetfile, inputfiles))