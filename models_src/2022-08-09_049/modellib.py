import torch, torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import torchvision.transforms.v2 as T
from torchvision.models._utils import IntermediateLayerGetter
#import PIL.Image  #use datasets.load_image() instead
import numpy as np
from datasets import get_transforms

import typing as tp
import io, warnings, sys, time, importlib
warnings.simplefilter('ignore') #pytorch is too noisy


if "__torch_package__" in dir():
    #inside a torch package
    import torch.package.package_importer
    import_func = torch.package.package_importer.import_module
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
        # self.segmentation_model = UNet()
        # self.classifier         = Ensemble([
        #     Classifier(self.segmentation_model, self.class_list, name) for name in ['mobilenet_v3_large', 'mobilenet_v2', 'shufflenet_v2_x1_0']
        # ])
        self._device_indicator  = torch.nn.Parameter(torch.zeros(0)) #dummy parameter
    
    def forward(self, x, temperature:float=0.75):
        results:tp.List[Prediction] = []
        device           = self._device_indicator.device
        x                = x.to(device)
        
        detector_outputs = self.detector(x)
        if torch.jit.is_scripting():
            detector_outputs = self.detector(x)[1]
        
        for o in detector_outputs:
            boxes            = o['boxes']
            _boxes           = torch.cat([torch.zeros_like(boxes[:,:1]), boxes], -1)
            cropsize         = self.classifier.image_size
            crops            = torchvision.ops.roi_align(x, _boxes, cropsize, sampling_ratio=1)
            if torch.onnx.is_in_onnx_export():
                #pad to batch dimension of at least one
                crops = torch.jit.script(pad_if_needed)(crops)
            if crops.shape[0] > 0:
                probabilities    = self.classifier(crops, T=temperature)
                probabilities    = probabilities[:boxes.shape[0]] #to counter-act padding in onnx
            else:
                probabilities    = torch.zeros([0,len(self.class_list)])
            
            results.append( Prediction(
                boxes          = boxes,
                box_scores     = o['scores'],
                probabilities  = probabilities,
                labels         = [self.class_list[p.argmax(-1)] for p in probabilities],
                crops          = crops,
            ) )
        return results
    
    @staticmethod
    def load_image(filename, to_tensor=False):
        #image =  PIL.Image.open(filename) #do not use, exif-unaware
        image = datasets.load_image(filename) #exif-aware
        if to_tensor:
            image = T.ToImageTensor(image)
        return image
    
    def process_image(self, image, use_onnx=False):
        if isinstance(image, str):
            image = self.load_image(image)
        x = T.ToImageTensor(image)
        with torch.no_grad():
            output = self.eval().forward(x[np.newaxis])[0]
        
        output = output.numpy()
        return {
            'boxes'            : output.boxes,
            'box_scores'       : output.box_scores,
            'cls_scores'       : output.probabilities,
            'per_class_scores' : [ dict(zip( self.class_list, p )) for p in output.probabilities.tolist() ],
            'labels'           : output.labels,
        }
    
    def start_training_detector(
            self, 
            imagefiles_train,          jsonfiles_train,
            imagefiles_test   = None,  jsonfiles_test = None, ood_files = [], 
            negative_classes  = [],    lr             = 5e-3,
            epochs            = 10,    callback       = None,
            num_workers       = 'auto',
    ):
        n_ood   = min(len(imagefiles_train) // 20, len(ood_files) )
        ds_type = datasets.DetectionDataset if (n_ood==0) else datasets.OOD_DetectionDataset
        ood_kw  = {}                        if (n_ood==0) else {'ood_files':ood_files, 'n_ood':n_ood}
        
        ds_train = ds_type(imagefiles_train, jsonfiles_train, augment=get_transforms(train=True), negative_classes=negative_classes, **ood_kw)
        ld_train = datasets.create_dataloader(ds_train, batch_size=8, shuffle=True, num_workers=num_workers)
        
        ld_test  = None
        if imagefiles_test is not None:
            ds_test  = datasets.DetectionDataset(imagefiles_test, jsonfiles_test, augment=get_transforms(train=False), negative_classes=negative_classes)
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



class Prediction(tp.NamedTuple):
    boxes:           torch.Tensor
    box_scores:      torch.Tensor
    probabilities:   torch.Tensor
    labels:          tp.List[str]
    crops:           torch.Tensor
    
    
    
    def numpy(self):
        return Prediction(*[x.cpu().numpy() if torch.is_tensor(x) else x for x in self])


def normalize(x):
    if len(x)==0:
        return x
    if x.ndim==3:
        x = x[None]
    xmax = x.max(1,True)[0].max(2,True)[0].max(3,True)[0]
    return x / torch.clamp_min(xmax, 1e-6)  #range 0...1

class Detector(torch.nn.Module):
    def __init__(self, image_size=300):
        super().__init__()
        self.basemodel = torchvision.models.detection.ssd300_vgg16(weights=None)
        self.resize = T.Resize([image_size]*2)
        self.image_size = image_size
        self._device_indicator = torch.nn.Parameter(torch.zeros(0)) #dummy parameter
    
    def forward(self, x, targets:tp.Optional[tp.List[tp.Dict[str, torch.Tensor]]]=None):
        size0 = [x.shape[-2], x.shape[-1]]
        x     = self.resize(x)
        size1 = [x.shape[-2], x.shape[-1]]
        x     = normalize(x)
        out   = self.basemodel(list(x), targets)
        if torch.jit.is_scripting():
            self.resize_boxes(out[1], size1, size0)
        elif not self.training:
            self.resize_boxes(out, size1, size0)
        return out

    def resize_boxes(self, inference_output:tp.List[tp.Dict[str, torch.Tensor]], from_size:tp.List[int], to_size:tp.List[int]):
        for o in inference_output:
            o['boxes'] = torchvision.models.detection.transform.resize_boxes(o['boxes'], from_size, to_size)
