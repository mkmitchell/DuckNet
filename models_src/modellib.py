import torch, torchvision
import numpy as np
import typing as tp
import warnings, sys, time, importlib
warnings.simplefilter('ignore') #pytorch is too noisy


if "__torch_package__" in dir():
    import torch_package_importer
    import_func = torch_package_importer.import_module
else:
    #normal
    import importlib
    import_func = lambda m: importlib.reload(importlib.import_module(m))



#internal modules
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
            image = torchvision.transforms.v2.ToImageTensor()(image)
        return image
    

    def process_image(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        x = torchvision.transforms.v2.ToImageTensor()(image)
        x = x.unsqueeze(0)
        x = torchvision.transforms.v2.ConvertImageDtype(torch.float32)(x)
        x = torchvision.transforms.v2.Resize((300,300))(x)
        x = torchvision.transforms.v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])(x)
        with torch.no_grad():
            # output = self.eval().forward(x[np.newaxis])[0]
            self.eval()
            output = self(x)
        
        output = output[0]

        return output
    
    def start_training_detector(
            self, 
            imagefiles_train,          jsonfiles_train,
            imagefiles_test   = None,  jsonfiles_test = None, 
            negative_classes  = [],    lr             = 5e-3,
            epochs            = 10,    callback       = None,
            num_workers       = 'auto',
    ):
        ds_type = datasets.DetectionDataset
        
        ds_train = ds_type(imagefiles_train, jsonfiles_train, augment=datasets.get_transforms(train=True), negative_classes=negative_classes)
        ld_train = datasets.create_dataloader(ds_train, batch_size=8, shuffle=True, num_workers=num_workers)
        
        ld_test  = None
        if imagefiles_test is not None:
            ds_test  = datasets.DetectionDataset(imagefiles_test, jsonfiles_test, augment=datasets.get_transforms(train=False), negative_classes=negative_classes)
            ld_test  = datasets.create_dataloader(ds_test, batch_size=8, shuffle=False, num_workers=num_workers)
        
        task = traininglib.DetectionTask(self.detector, callback=callback, lr=lr)
        ret  = task.fit(ld_train, ld_test, epochs=epochs)
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
    
# class Prediction(tp.NamedTuple):
#     boxes:           torch.Tensor
#     box_scores:          torch.Tensor
#     # probabilities:   torch.Tensor
#     labels:          tp.List[str]
#     # crops:           torch.Tensor

#     def numpy(self):
#         return Prediction(*[x.cpu().numpy() if torch.is_tensor(x) else x for x in self])

class Detector(torch.nn.Module):
    def __init__(self, image_size:int=300):
        super().__init__()
        # self.basemodel = torchvision.models.detection.ssd300_vgg16(weights=torch.load('C:/Users/zack/Documents/GitHub/SSD_VGG_PyTorch/ssd300_vgg16_gradientAccumulation_noHen.pth', map_location=torch.device('cpu'), weights_only=True), progress=False)
        self.basemodel = torchvision.models.detection.ssd300_vgg16()
        self.basemodel.load_state_dict(torch.load('C:/Users/zack/Documents/GitHub/SSD_VGG_PyTorch/ssd300_vgg16_gradientAccumulation_noHen.pth', map_location=torch.device('cpu')))
        self.image_size = image_size
        self._device_indicator = torch.nn.Parameter(torch.zeros(0)) #dummy parameter
        
    
    def forward(self, x, targets:tp.Optional[tp.List[tp.Dict[str, torch.Tensor]]]=None):
        out   = self.basemodel(list(x), targets)
        if torch.jit.is_scripting():
            self.resize_boxes(out[1], 5184, 2916, self.image_size)
        elif not self.training:
            self.resize_boxes(out, 5184, 2916, self.image_size)
        return out

    def resize_boxes(self, inference_output:tp.List[tp.Dict[str, torch.Tensor]], width:tp.List[int], height:tp.List[int], image_size:int=300):
        for o in inference_output:
            o['boxes'][:, 0] *= width / image_size 
            o['boxes'][:, 1] *= height / image_size
            o['boxes'][:, 2] *= width / image_size
            o['boxes'][:, 3] *= height / image_size