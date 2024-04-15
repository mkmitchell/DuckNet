import sklearn
import torch, torchvision
from torchvision import datapoints as dp
import torchvision.transforms.v2 as T
import numpy as np
import PIL.Image, PIL.ImageOps
import glob, json, os
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


def get_boxes_from_jsonfile(jsonfile, flip_axes=False, normalize=False):
    '''Reads bounding boxes from a LabeLMe json file and returns them as a (Nx4) array'''
    jsondata = json.loads(read_json_until_imagedata(jsonfile))
    boxes    = [shape['points'] for shape in jsondata['shapes']]
    boxes    = [[min(box[0],box[2]),min(box[1],box[3]),
                 max(box[0],box[2]),max(box[1],box[3])] for box in np.reshape(boxes, (-1,4))]
    boxes    = np.array(boxes)
    boxes    = (boxes.reshape(-1,2) / get_imagesize_from_jsonfile(jsonfile)[::-1]).reshape(-1,4) if normalize else boxes
    boxes    = boxes[:,[1,0,3,2]] if flip_axes else boxes
    return boxes.reshape(-1,4)


def get_labels_from_jsonfile(jsonfile):
    '''Reads a list of labels in a json LabelMe file.'''
    return [ s['label'] for s in json.loads( read_json_until_imagedata(jsonfile) )['shapes'] ]


def get_imagesize_from_jsonfile(jsonfile):
    f        = open(jsonfile, 'rb')
    #skip to the last n bytes
    filesize = f.seek(0,2)
    n        = min(192, filesize)
    f.seek(-n, 2)
    buffer   = f.read()
    idx      = buffer.rfind(b"imageHeight")
    if idx<0:
        raise ValueError(f'Cannot get image size: {jsonfile} does not contain image size information')
    jsondata = json.loads( b'{'+buffer[idx-1:] )
    return np.array([jsondata['imageHeight'], jsondata['imageWidth']])


def get_transforms(train:bool):
    transforms = []
    if train:
        transforms.append(T.RandomZoomOut(fill = defaultdict(lambda: 0, {dp.Image: (255, 20, 147)}),
                                        p = 0.3,
                                        side_range = (1.0, 2.0)))
        transforms.append(T.RandomIoUCrop())
        transforms.append(T.Resize((300, 300), antialias = True)) # no maintain aspect ratio
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        transforms.append(T.Resize((300, 300), antialias = True)) # no maintain aspect ratio
    transforms.append(T.ToImageTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    transforms.append(T.SanitizeBoundingBox())
    transforms.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # ImageNet mean and std values for normalization
    return T.Compose(transforms)


def create_dataloader(dataset, batch_size, shuffle=False, num_workers='auto'):
    if num_workers == 'auto':
        num_workers = os.cpu_count()
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, collate_fn=getattr(dataset, 'collate_fn', None),
                                       num_workers=num_workers, pin_memory=True,
                                       worker_init_fn=lambda x: np.random.seed(torch.randint(0,1000,(1,))[0].item()+x) )



#class Dataset(torch.utils.data.Dataset):  #inheriting gives errors on unpickling
class Dataset:
    def __init__(self, jpgfiles, jsonfiles, train:bool):
        self.train   = train
        self.jsonfiles = jsonfiles
        self.jpgfiles  = jpgfiles
    
    def __len__(self):
        return len(self.jpgfiles)
    
    def __getitem__(self, i):
        image, target = self.get_item(i)
        return image, target



class DetectionDataset(Dataset):
    #IGNORE = set(['Duck_hanging'])
    #SIZE   = 300 #px
    
    def __init__(self, jpgfiles, jsonfiles, train:bool, transforms, negative_classes:list, image_size:int=300):
        super().__init__(jpgfiles, jsonfiles, train)
        self.negative_classes = negative_classes
        self.image_size       = image_size
        self.transforms = transforms
    
    def __get_item__(self, i):
        jsonfile = self.jsonfiles[i]
        jpgfile  = self.jpgfiles[i]
    
        image    = load_image(jpgfile)
        #load normalized boxes: 0...1
        boxes    = get_boxes_from_jsonfile(jsonfile, flip_axes=0, normalize=0)
        #duck species labels
        labels   = get_labels_from_jsonfile(jsonfile)

        #remove hanging ducks (spp pre-trained model not trained to detect)
        boxes    = [box for box,label in zip(boxes, labels) if label not in self.negative_classes]
        boxes    = np.array(boxes).reshape(-1,4)

        target = {}
        target['boxes'] = dp.BoundingBox(boxes, format="XYXY", spatial_size=self.image_size)
        target['labels'] = labels
        target['image_id'] = torch.tensor([i])
        target['area'] = torch.tensor([((box[3]-box[1])*(box[2]-box[0])) for box in boxes])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target
    
    @staticmethod
    def collate_fn(batchlist):
        images    = [x[0] for x in batchlist]
        images    = torch.stack(images)
        targets   = [x[1] for x in batchlist]
        return images, targets



class OOD_DetectionDataset(DetectionDataset):
    '''Augments ducks dataset with out-of-distribution images'''
    def __init__(self, *args, ood_files, n_ood, **kwargs):
        super().__init__(*args, **kwargs)
        self.ood_files = ood_files
        self.n_ood     = n_ood
    
    def __len__(self):
        return super().__len__()+self.n_ood
    
    def get_item(self, i):
        if i < super().__len__():
            return super().get_item(i)
        i      = np.random.randint(len(self.ood_files))
        image  = load_image(self.ood_files[i]).resize([self.SIZE]*2)
        return image, {'boxes':torch.as_tensor([]).reshape(-1,4), 'labels':torch.as_tensor([]).long()}


def should_ignore_file(json_file='', ignore_list=[]):
    labels = get_labels_from_jsonfile(json_file) if os.path.exists(json_file) else []
    return any([(l in ignore_list) for l in labels])


def random_wrong_box(imagesize, true_boxes, n=15, max_iou=0.1):
    '''Tries to find a box that does not overlap with other `true_boxes`'''
    for i in range(n):
        center = np.random.random(2)*imagesize
        size   = np.random.uniform(0.05, 0.50, size=2)*imagesize
        box    = np.concatenate([center-size/2, center+size/2])
        ious   = torchvision.ops.box_iou(torch.as_tensor(true_boxes), torch.as_tensor(box)[None] ).numpy()
        if true_boxes is None or ious.max() < max_iou:
            return box
    else:
        return None
