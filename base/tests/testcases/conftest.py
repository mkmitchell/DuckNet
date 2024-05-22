import os
os.environ['DO_NOT_RELOAD'] = 'true'

from backend.app import App
App().recompile_static(force=True)


if os.path.exists( os.path.join(os.environ.get('ROOT_PATH'), 'training/base') ):
    #generate dummy models for testing
    from backend.app   import get_models_path
    from training.base import basemodel
    models_path = os.path.join(get_models_path(), 'detection')
    os.makedirs(models_path, exist_ok=True)
    for i in range(3):
        basemodel.Model().save( os.path.join(models_path, f'model_{i}') )

