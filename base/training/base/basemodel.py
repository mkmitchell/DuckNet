import zipfile, os, time, pickle, pkgutil, sys
import numpy as np
import PIL.Image
import torch


#additional optional modules to import
import importlib
MODULES   = []
#[...]    =  [importlib.reload(importlib.import_module(m)) for m in MODULES]


class Model:
    def __init__(self):
        self.weights = np.sort(np.random.random(4))

    def load_image(self, path):
        return PIL.Image.open(path) / np.float32(255)
    
    def process_image(self, image):
        '''Dummy processing function'''
        if isinstance(image, str):
            image = self.load_image(image)
        result      = np.zeros( image.shape[:2], 'uint8' )
        y0,x0,y1,x1 = (self.weights * (image.shape[:2]+image.shape[:2])).astype(int)
        print(y0,x0,y1,x1)
        result[y0:y1,x0:x1] = 255

        print(f'Simulating image processing')
        for i in range(3):
            #TODO: progress callback
            time.sleep(0.5)
        return result

    def start_training(self, imagefiles, targetfiles, epochs=100, callback=None):
        print(f'Simulating training')
        self.stop_requested = False
        for i in range(3):
            if self.stop_requested:
                print('Stopping training')
                return False
            self.weights = np.sort(np.random.random(4))
            callback( i/3 )
            time.sleep(1)
        callback( 1.0 )
        return True

    def stop_training(self):
        self.stop_requested = True
    
    def save(self, destination):
        if not destination.endswith('.pt.zip'):
            destination += '.pt.zip'
        
        try:
            import torch.package.package_importer as imp
            #re-export
            importer = (imp, torch.package.sys_importer)
        except ImportError as e:
            #first export
            importer = (torch.package.sys_importer,)
        with torch.package.PackageExporter(destination, importer) as pe:
            interns = [__name__.split('.')[-1]]+MODULES
            pe.intern(interns)
            pe.extern('**')
            pe.save_pickle('model', 'model.pkl', self)
        return destination
    

if __name__=='__main__':
    Model().save('./model')
