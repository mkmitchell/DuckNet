import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
import numpy as np
import PIL.Image, PIL.ImageOps
import json
from typing import Any, Callable, List, Tuple, Optional

def load_image(path: str) -> PIL.Image.Image:
    '''Load image, rotate according to EXIF orientation'''
    image: Optional[PIL.Image.Image] = PIL.Image.open(path).convert('RGB')
    image = PIL.ImageOps.exif_transpose(image)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
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

def create_dataloader(dataset, batch_size, shuffle:bool=True, num_workers=0, sampler=None):
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle if sampler is None else False, 
                                       num_workers=num_workers, 
                                       collate_fn=DetectionDataset.collate_fn,
                                       sampler=sampler)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, jpgfiles, jsonfiles):
        self.jsonfiles = jsonfiles
        self.jpgfiles  = jpgfiles
    
    def __len__(self):
        return len(self.jpgfiles)

class DetectionDataset(Dataset):
    """Dataset Loader for Waterfowl Drone Imagery"""

    def __init__(self, jpgfiles, jsonfiles, augment: bool, negative_classes: list, class_list: list):
        super().__init__(jpgfiles, jsonfiles)
        self.augment = augment
        self.negative_classes = negative_classes
        self.class_list = class_list[:]
        self.label_dict = {i + 1: label for i, label in enumerate(self.class_list)}
        self.rev_label_dict = {label: i + 1 for i, label in enumerate(self.class_list)}

    def __getitem__(self, idx):
        jsonfile = self.jsonfiles[idx]
        jpgfile = self.jpgfiles[idx]
        image = load_image(jpgfile)
        image = tv_tensors.Image(image)

        boxes = get_boxes_from_jsonfile(jsonfile)
        labels = get_labels_from_jsonfile(jsonfile)

        new_labels = []
        for label in labels:
            if label in self.rev_label_dict:
                new_labels.append(self.rev_label_dict[label])
            else:
                new_id = max(self.rev_label_dict.values()) + 1 if self.rev_label_dict else 1
                print(f"Label '{label}' not found; adding as new label with ID {new_id}.")
                self.rev_label_dict[label] = new_id
                self.label_dict[new_id] = label
                self.class_list.append(label)
                new_labels.append(new_id)
        labels = new_labels

        boxes = [box for box, label in zip(boxes, labels) if label not in self.negative_classes]
        boxes = torch.as_tensor(np.array(boxes).reshape(-1, 4))

        target = {
            'boxes': tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(image.shape[1], image.shape[2])), # type: ignore
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

        augments_list: List[Callable[[Any, Any], Tuple[Any, Any]]] = [T.ToImage()]

        if self.augment:
            augments_list.extend([
                T.RandomIoUCrop(min_scale=0.5, max_scale=1.5),  # zoom in <1, zoom out >1
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.2, hue=0.02)], p=0.4),
                T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.5, 1.0))], p=0.4),
                T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                T.RandomApply([T.RandomRotation(degrees=(-10, 10), interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
                T.RandomHorizontalFlip(0.5),
                T.ClampBoundingBoxes(),      
                T.SanitizeBoundingBoxes(min_size=1, min_area=1) 
            ])

        augments_list.extend([
            T.Resize(size=(810,), max_size=1440, interpolation=T.InterpolationMode.BILINEAR),
            T.ToDtype(dtype=torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        augments = T.Compose(augments_list)
        image, target = augments(image, target)

        return image, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))