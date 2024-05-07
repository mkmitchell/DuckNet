import torch, torchvision
from torchvision import datapoints as dp
import torchvision.transforms.v2 as T
import numpy as np
import PIL.Image, PIL.ImageOps
import json, os
from collections import defaultdict


def load_image(path:str) -> PIL.Image:
    '''Load image, rotate according to EXIF orientation'''
    image = PIL.Image.open(path).convert('RGB')
    # image = PIL.ImageOps.exif_transpose(image)
    return image

# Comment out the import statement since TensorFlow is not needed for DuckNet.
# import tensorflow as tf
# assert [] == tf.config.list_physical_devices('GPU')


def guess_encoding(x:bytes) -> str:
    try:
        return x.decode('utf8')
    except UnicodeDecodeError:
        return x.decode('cp1250')


def read_json_annotation_data(jsonfile):
    '''This function reads a jsonfile's annotation data in DARWIN JSON format.
       Returns a valid JSON string'''
    f = open(jsonfile, 'rb')
    data = f.read(1024*16)
    return guess_encoding(data)


def get_boxes_from_jsonfile(jsonfile, flip_axes=False, normalize=False):
    '''Reads bounding boxes from a DARWIN json file and returns them as a (Nx4) array'''
    jsondata = json.loads(read_json_annotation_data(jsonfile))
    boxes = []
    for i in range(len(jsondata['annotations'])):
        box = [jsondata['annotations'][i]['bounding_box']['x'], 
                jsondata['annotations'][i]['bounding_box']['y'], 
                jsondata['annotations'][i]['bounding_box']['x']+jsondata['annotations'][i]['bounding_box']['w'], 
                jsondata['annotations'][i]['bounding_box']['y']+jsondata['annotations'][i]['bounding_box']['h']]
        box = np.array(box)
        box.reshape(-1,4)
        boxes.append(box)
    return boxes


def get_labels_from_jsonfile(jsonfile):
    '''Reads a list of labels in a json LabelMe file.'''
    return [ a['name'] for a in json.loads( read_json_annotation_data(jsonfile) )['annotations'] ]


def get_imagesize_from_jsonfile(jsonfile):
    jsondata = json.loads(read_json_annotation_data(jsonfile))
    return np.array([jsondata['item']['slots'][0]['height'], jsondata['item']['slots'][0]['width']])


def get_transforms(train):
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
    # transforms.append(T.SanitizeBoundingBox())
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
    def __init__(self, jpgfiles, jsonfiles, augment=False):
        self.augment   = augment
        self.jsonfiles = jsonfiles
        self.jpgfiles  = jpgfiles
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        if self.augment:
            self.transform.transforms += [
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)
            ]
    
    def __len__(self):
        return len(self.jpgfiles)
    
    def __getitem__(self, i):
        image, target = self.get_item(i)
        # image = self.transform(image)
        return image, target



class DetectionDataset(Dataset):
    #IGNORE = set(['Duck_hanging'])
    #SIZE   = 300 #px
    
    def __init__(self, jpgfiles, jsonfiles, augment:bool, negative_classes:list, image_size:int=300):
        super().__init__(jpgfiles, jsonfiles, augment)
        self.negative_classes = negative_classes
        self.image_size       = image_size
    
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

        if self.augment:
            image, target = T.RandomZoomOut(fill = defaultdict(lambda: 0, {dp.Image: (255, 20, 147)}),
                                    p = 0.3, side_range = (1.0, 2.0))(image)
            image, target = T.RandomIoUCrop()(image)
            image, target = T.Resize((self.image_size, self.image_size), antialias = True)(image) # no maintain aspect ratio
            image, target = T.RandomHorizontalFlip(0.5)(image)
        else:
            image, target = T.Resize((self.image_size, self.image_size), antialias = True)(image)
        image, target = T.ToImageTensor()(image)
        image, target = T.ConvertImageDtype(torch.float)(image)
        image, target = T.SanitizeBoundingBox()(image)
        image, target = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image) # ImageNet mean and std values for normalization

        return image, target
    

    @staticmethod
    def collate_fn(batchlist):
        images    = [x[0] for x in batchlist]
        images    = torch.stack(images)
        targets   = [x[1] for x in batchlist]
        return images, targets



def should_ignore_file(json_file='', ignore_list=[]):
    labels = get_labels_from_jsonfile(json_file) if os.path.exists(json_file) else []
    return any([(l in ignore_list) for l in labels])

def augment_box(box, scale=25, min_size=8):
    new_box  = box + np.random.normal(scale=scale, size=4)
    box_size = new_box[2:] - new_box[:2]
    if any(box_size < min_size):
        new_box = box
    return new_box

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
