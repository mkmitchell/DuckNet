from base.backend.app import App as BaseApp
import backend.processing
import backend.training

import os
import flask

class App(BaseApp):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.route('/read_exif_datetime')(self.read_exif_datetime)

    #TODO: unify
    #override
    def process_image(self, imagename):
        full_path = os.path.join(self.cache_path, imagename)
        if not os.path.exists(full_path):
            flask.abort(404)
        
        print(f'Processing image with model {self.settings.active_models["detection"]}')
        model = self.settings.models['detection']
        result = model.process_image(full_path)
        
        threshold = self.settings.confidence_threshold/100
        filtered_items = [(i, score) for i, score in enumerate(result['box_scores']) if score > threshold]
        formatted_labels = [{result['labels'][i]: float(score)} for i, score in filtered_items]
        
        filtered_boxes = []
        for i, _ in filtered_items:
            box = result['boxes'][i]
            if hasattr(box, 'tolist'):
                filtered_boxes.append(box.tolist())
            else:
                filtered_boxes.append(list(box))
        
        jsonresult = {
            'labels': formatted_labels,
            'boxes': filtered_boxes,
            'datetime': backend.processing.load_exif_datetime(full_path)
        }
        
        print(jsonresult)
        return flask.jsonify(jsonresult)

    #TODO: unify
    #override
    def training(self):
        requestform  = flask.request.get_json(force=True)
        options      = requestform['options']
        imagefiles   = requestform['filenames']
        imagefiles   = [os.path.join(self.cache_path, f) for f in imagefiles]
        targetfiles  = backend.training.find_targetfiles(imagefiles)
        if not all(targetfiles):
            flask.abort(404)
        ok = backend.training.start_training(imagefiles, targetfiles, options, self.settings)
        return ok
    
    def read_exif_datetime(self):
        filename = flask.request.args['filename']
        full_path = os.path.join(self.cache_path, filename)
        if not os.path.exists(full_path):
            flask.abort(404)
        
        return flask.jsonify({
            'datetime':  backend.processing.load_exif_datetime(full_path),
        })

