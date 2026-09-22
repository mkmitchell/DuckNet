"""DuckNet Flask application: overrides of the base DigIT app routes."""

import os

import flask
import numpy as np
from werkzeug.exceptions import HTTPException

import backend.processing
import backend.settings  # binds backend.settings for the base app
import backend.training
from base.backend.app import App as BaseApp
from base.backend.app import safe_cache_path


def json_error(error: HTTPException):
    """Answer HTTP errors with JSON so the browser can show the reason."""
    response = flask.jsonify(
        {'error': error.code, 'description': error.description}
    )
    response.status_code = error.code
    return response


class App(BaseApp):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.route('/read_exif_datetime')(self.read_exif_datetime)
        self.route('/training_status')(self.training_status)
        self.register_error_handler(HTTPException, json_error)

    def training_status(self):
        return flask.jsonify(backend.training.training_status())

    # override
    def process_image(self, imagename):
        full_path = safe_cache_path(imagename)
        if not os.path.exists(full_path):
            flask.abort(404)

        model_name = self.settings.active_models['detection']
        print(f'Processing image {imagename} with model {model_name}')
        try:
            result = backend.processing.process_image(full_path, self.settings)
        except backend.processing.ProcessingBusyError as exc:
            flask.abort(409, description=str(exc))

        jsonresult = {
            'labels': result['per_class_scores'],
            'boxes': np.array(result['boxes']).tolist(),
            'datetime': backend.processing.load_exif_datetime(full_path),
        }
        return flask.jsonify(jsonresult)

    # override
    def training(self):
        requestform = flask.request.get_json(force=True)
        options = requestform['options']
        imagefiles = [safe_cache_path(f) for f in requestform['filenames']]
        if not all(os.path.exists(f) for f in imagefiles):
            flask.abort(
                404, description='Not all training images were uploaded'
            )
        targetfiles = backend.training.find_targetfiles(imagefiles)
        if not all(targetfiles):
            flask.abort(
                404,
                description='Not all training images have an annotation file',
            )

        try:
            return backend.training.start_training(
                imagefiles, targetfiles, options, self.settings
            )
        except RuntimeError as exc:
            flask.abort(409, description=str(exc))

    def read_exif_datetime(self):
        full_path = safe_cache_path(flask.request.args['filename'])
        if not os.path.exists(full_path):
            flask.abort(404)

        return flask.jsonify(
            {
                'datetime': backend.processing.load_exif_datetime(full_path),
            }
        )
