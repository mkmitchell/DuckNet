import torch
from torchvision import datapoints as dp
import torchvision.transforms.v2 as T
import numpy as np
import PIL.Image, PIL.ImageOps
import json, os
import pandas as pd
from collections import defaultdict

def load_image(path:str) -> PIL.Image:
    '''Load image, rotate according to EXIF orientation'''
    image = PIL.Image.open(path).convert('RGB')
    image = PIL.ImageOps.exif_transpose(image)
    return image


def guess_encoding(x:bytes) -> str:
    try:
        return x.decode('utf8')
    except UnicodeDecodeError:
        return x.decode('cp1250')


def read_json_until_imagedata(jsonfile):
    '''LabelMe JSON are rather large because they contain the whole image additionally to the labels.
       This function reads a jsonfile only up to the imagedata attribute (ignoring everything afterwards) to reduce the loading time.
       Returns a valid JSON string'''
    f = open(jsonfile, 'rb')
    f.seek(0,2); n=f.tell(); f.seek(0,0)
    buffer = b''
    while b'imageData' not in buffer and len(buffer)<n:
        data      = f.read(1024*16)
        buffer   += data
        if len(data)==0:
            return buffer
    buffer   = buffer[:buffer.index(b'imageData')]
    buffer   = buffer[:buffer.rindex(b',')]
    buffer   = buffer+b'}'
    return guess_encoding(buffer)


def get_boxes_from_jsonfile(jsonfile, flip_axes=False):
    '''Reads bounding boxes from a LabeLMe json file and returns them as a (Nx4) array'''
    jsondata = json.loads(read_json_until_imagedata(jsonfile))

    boxes    = [shape['points'] for shape in jsondata['shapes']]
    boxes    = [[min(box[0],box[2]),min(box[1],box[3]),
                 max(box[0],box[2]),max(box[1],box[3])] for box in np.reshape(boxes, (-1,4))]
    boxes    = np.array(boxes)
    boxes    = boxes[:,[1,0,3,2]] if flip_axes else boxes
    return boxes.reshape(-1,4)


def get_labels_from_jsonfile(jsonfile):
    '''Reads a list of labels in a json LabelMe file.'''
    return [ s['label'] for s in json.loads( read_json_until_imagedata(jsonfile) )['shapes'] ]


def get_imagename_from_jsonfile(jsonfile):
    jsondata = json.loads(read_json_until_imagedata(jsonfile))
    return str(jsondata['imagePath'])



def create_dataloader(dataset, batch_size, shuffle:bool, num_workers=0):
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers, 
                                       collate_fn=getattr(dataset, 'collate_fn', None),
                                       pin_memory=True)



#class Dataset(torch.utils.data.Dataset):  #inheriting gives errors on unpickling
class Dataset:
    def __init__(self, jpgfiles, jsonfiles):
        self.jsonfiles = jsonfiles
        self.jpgfiles  = jpgfiles
    
    def __len__(self):
        return len(self.jpgfiles)
    
    def __getitem__(self, i):
        image, target = self.get_item(i)
        return image, target



class DetectionDataset(Dataset):
    """Dataset Loader for Waterfowl Drone Imagery"""
    
    def __init__(self, jpgfiles, jsonfiles, augment:bool, negative_classes:list):
        super().__init__(jpgfiles, jsonfiles)
        self.augment   = augment
        self.negative_classes = negative_classes


    def __getitem__(self, idx):
        jsonfile = self.jsonfiles[idx]
        jpgfile = self.jpgfiles[idx]
        image    = load_image(jpgfile)
        width, height = image.size
        image    = dp.Image(image)

        boxes = get_boxes_from_jsonfile(jsonfile)

        labels = get_labels_from_jsonfile(jsonfile)

        label_dict = {4.0: 'MALL', 1.0: 'AMCO', 3.0: 'GWTE', 6.0: 'NSHO', 2.0: 'GADW', 8.0: 'RNDU', 5.0: 'NOPI', 7.0: 'REDH'} #FIXME: hardcoded
        
        labels = [0 if label not in label_dict.values() else list(label_dict.keys())[list(label_dict.values()).index(label)] for label in labels]

        boxes    = [box for box,label in zip(boxes, labels) if label not in self.negative_classes]
        boxes    = np.array(boxes).reshape(-1,4)

        target = {}
        target['boxes'] = dp.BoundingBox(boxes, format="XYXY", spatial_size=(height, width))
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        augments = []
        if self.augment is not None:
            augments.append(T.RandomZoomOut(fill = defaultdict(lambda: 0, {dp.Image: (255, 20, 147)}),
                                          p = 0.3,
                                          side_range = (1.0, 2.0)))
            augments.append(T.RandomIoUCrop())
            augments.append(T.Resize((300, 300), antialias = True)) # no maintain aspect ratio
            augments.append(T.RandomHorizontalFlip(0.5))
        else:
            augments.append(T.Resize((300, 300), antialias = True)) # no maintain aspect ratio
        augments.append(T.ToImageTensor())
        augments.append(T.ConvertImageDtype(torch.float))
        augments.append(T.SanitizeBoundingBox())
        augments.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # ImageNet mean and std values for normalization
        augments = T.Compose(augments)
        image, target = augments(image, target)

        return image, target


    @staticmethod
    def collate_fn(batchlist):
        images    = [x[0] for x in batchlist]
        images    = torch.stack(images)
        targets   = [x[1] for x in batchlist]
        return images, targets