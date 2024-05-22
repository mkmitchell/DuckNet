import torch, torchvision
import torchvision.transforms.v2 as T
import numpy as np
import typing as tp
import warnings, sys, time, importlib, exif
warnings.simplefilter('ignore') #pytorch is too noisy


if "__torch_package__" in dir():
    # import torch_package_importer
    # import_func = torch_package_importer.import_module

    import_func = lambda m: torch.package.PackageImporter('basemodel.pt.zip').import_module(m)
else:
    #normal
    import importlib
    import_func = lambda m: importlib.reload(importlib.import_module(m))



#import internal modules
MODULES = ['datasets', 'traininglib']
[datasets, traininglib] = [import_func(m) for m in MODULES]



class DuckDetector(torch.nn.Module):
    def __init__(self, classes_of_interest):
        super().__init__()
        self.class_list         = classes_of_interest
        self.detector           = Detector()
        self._device_indicator  = torch.nn.Parameter(torch.zeros(0)) #dummy parameter
    
    def forward(self, x):
        results:tp.List = []
        device           = self._device_indicator.device
        x                = x.to(device)
        
        detector_outputs = self.detector(x)
        if torch.jit.is_scripting():
            detector_outputs = self.detector(x)[1]
        
        for o in detector_outputs:
            results.append(o)
        return results
    
    @staticmethod
    def load_image(filename, to_tensor=False):
        #image =  PIL.Image.open(filename) #do not use, exif-unaware
        image = datasets.load_image(filename) #exif-aware
        if to_tensor:
            image = T.ToImageTensor()(image)
        return image
    

    def process_image(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        width, height = image.size
        x = T.ToImageTensor()(image)
        x = x.unsqueeze(0)
        x = T.ConvertImageDtype(torch.float32)(x)
        x = T.Resize((300,300))(x)
        x = T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(x)
        with torch.no_grad():
            output = self.eval().forward(x)[0]

        boxes = output['boxes']
        # rescale boxes to original image size
        boxes[:, 0] *= width / 300
        boxes[:, 1] *= height / 300
        boxes[:, 2] *= width / 300
        boxes[:, 3] *= height / 300

        return {
            'boxes': boxes,
            'scores': output['scores'],
            'labels': output['labels']
            }
    
    
    # def start_training_detector(
    #         self, 
    #         imagefiles_train,          jsonfiles_train,
    #         imagefiles_test   = None,  jsonfiles_test = None, 
    #         negative_classes  = [],    lr             = 5e-3,
    #         epochs            = 10,    callback       = None,
    #         num_workers       = 'auto',
    # ):
    #     ds_type = datasets.DetectionDataset
        
    #     ds_train = ds_type(imagefiles_train, 
    #                        jsonfiles_train, 
    #                        augment=datasets.get_transforms(train=True),
    #                        negative_classes=negative_classes)
        
    #     ld_train = datasets.create_dataloader(ds_train, batch_size=8, shuffle=True, num_workers=num_workers)
        
    #     ld_test  = None
    #     if imagefiles_test is not None:
    #         ds_test  = datasets.DetectionDataset(imagefiles_test, jsonfiles_test, augment=datasets.get_transforms(train=False), negative_classes=negative_classes)
    #         ld_test  = datasets.create_dataloader(ds_test, batch_size=1, shuffle=False, num_workers=num_workers)
        
    #     task = traininglib.DetectionTask(self.detector, callback=callback, lr=lr)
    #     ret  = task.fit(ld_train, ld_test, epochs=epochs)
    #     return (not task.stop_requested and not ret)
        
    def start_training_detector(
        self, 
        imagefiles_train,           jsonfiles_train,
        # imagefiles_test   = None,   jsonfiles_test = None, 
        lr                = 0.05,   epochs         = 10,     
        callback       = None,      num_workers    = 0,
    ):

        
        ds_train = datasets.DetectionDataset(imagefiles_train, 
                                             jsonfiles_train,
                                             augment=True)
        
        dl_train = datasets.create_dataloader(ds_train, batch_size=8, shuffle=True, num_workers=num_workers)
        
        dl_test  = None
        # if imagefiles_test is not None:
        #     ds_test  = datasets.DetectionDataset(imagefiles_test, jsonfiles_test, augment=False)
            # dl_test  = datasets.create_dataloader(ds_test, batch_size=1, shuffle=False, num_workers=num_workers)
        
        task = traininglib.DetectionTask(self.detector, callback=callback, lr=lr)
        ret  = task.fit(dl_train, dl_test, epochs=epochs)
        return (not task.stop_requested and not ret)



    def stop_training(self):
        traininglib.TrainingTask.request_stop()
    
    def save(self, destination):
        if isinstance(destination, str):
            destination = time.strftime(destination)
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
            pe.extern('**', exclude=['torchvision.**'])
            externs = ['torchvision.ops.**', 'torchvision.datasets.**', 'torchvision.io.**', 'torchvision.models.*']
            pe.intern('torchvision.**', exclude=externs)
            pe.extern(externs, exclude='torchvision.models.detection.**')
            pe.intern('torchvision.models.detection.**')
            
            #force inclusion of internal modules + re-save if importlib.reload'ed
            for inmod in interns:
                if inmod in sys.modules:
                    pe.save_source_file(inmod, sys.modules[inmod].__file__, dependencies=True)
                else:
                    pe.save_source_string(inmod, importer[0].get_source(inmod))
            
            pe.save_pickle('model', 'model.pkl', self)
            pe.save_text('model', 'class_list.txt', '\n'.join(self.class_list))
        return destination
    

#@torch.jit.script  #commented out to make the module cloudpickleable, scripted on the fly
def pad_if_needed(x):
    #pad to batch dimension of at least one
    paddings = [0,0, 0,0, 0,0, 0, max(0, 1 - x.shape[0])]
    x        = torch.nn.functional.pad(x, paddings)
    return x

class Detector(torch.nn.Module):
    def __init__(self, image_size:int=300):
        super().__init__()
        self.basemodel = torchvision.models.detection.ssd300_vgg16()
        self.basemodel.load_state_dict(torch.load('C:/Users/zack/Documents/GitHub/SSD_VGG_PyTorch/ssd300_vgg16_gradientAccumulation_noHen.pth', map_location=torch.device('cpu')))
        self.image_size = image_size
        self._device_indicator = torch.nn.Parameter(torch.zeros(0)) #dummy parameter
        
    
    def forward(self, x, targets:tp.Optional[tp.List[tp.Dict[str, torch.Tensor]]]=None):
        out   = self.basemodel(x, targets)
        return out