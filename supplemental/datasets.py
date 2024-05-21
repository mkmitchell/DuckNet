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
    # image = PIL.ImageOps.exif_transpose(image)
    return image


# read image name from jsonfile
def get_imagename_from_jsonfile(jsonfile):
    with open(jsonfile, 'r') as j:
        jsondata = json.loads(j.read())
    return jsondata['item']['slots'][0]['source_files'][0]['file_name']


def get_boxes_from_jsonfile(jsonfile):
    '''Reads bounding boxes from a DARWIN json file and returns them as a (Nx4) array'''
    with open(jsonfile, 'r') as j:
        jsondata = json.loads(j.read())
    boxes = []
    for i in range(len(jsondata['annotations'])):
        box = [jsondata['annotations'][i]['bounding_box']['x'], 
                jsondata['annotations'][i]['bounding_box']['y'], 
                jsondata['annotations'][i]['bounding_box']['x']+jsondata['annotations'][i]['bounding_box']['w'], 
                jsondata['annotations'][i]['bounding_box']['y']+jsondata['annotations'][i]['bounding_box']['h']]
        box = np.array(box, dtype=float)
        box.reshape(-1,4)
        boxes.append(box)
    return boxes # return as (Nx4) array of bounding


def get_labels_from_jsonfile(jsonfile):
    '''Reads a list of labels in a DARWIN json file.'''
    with open(jsonfile, 'r') as j:
        jsondata = json.loads(j.read())
    return [ a['name'] for a in jsondata['annotations'] ]
 

def get_imagesize_from_jsonfile(jsonfile):
    with open(jsonfile, 'r') as j:
        jsondata = json.loads(j.read())
    return (jsondata['item']['slots'][0]['height'], jsondata['item']['slots'][0]['width'])


def create_df_from_jsonfile(jsonfiles) -> pd.DataFrame:
    """Stores image names, labels, and boxes in a dataframe from a dir of json"""
    df = pd.DataFrame(columns = ['image_name', 'labels', 'boxes'])
    for jsonfile in os.listdir(jsonfiles):
        boxes = np.array(get_boxes_from_jsonfile(jsonfiles + jsonfile))
        labels = list(get_labels_from_jsonfile(jsonfiles + jsonfile))
        img_name = str(get_imagename_from_jsonfile(jsonfiles + jsonfile))
        for i in range(len(labels)):
            df = df.append({'image_name': img_name, 'labels': labels[i], 'boxes': boxes[i]}, ignore_index = True)
    return df

def get_augments(train):
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


def create_dataloader(dataset, batch_size, shuffle:bool, num_workers=0):
    return torch.utils.data.DataLoader(dataset, 
                                       batch_size=batch_size, 
                                       shuffle=shuffle, 
                                       num_workers=num_workers, 
                                       collate_fn=getattr(dataset, 'collate_fn', None),
                                       pin_memory=True)


class DetectionDataset(torch.utils.data.Dataset):
    """Dataset Loader for Waterfowl Drone Imagery"""
    
    def __init__(self, jpgfiles, jsonfiles, augment:bool):
        self.augment   = augment
        self.jsonfiles = jsonfiles
        self.df        = create_df_from_jsonfile(jsonfiles)
        self.jpgfiles  = jpgfiles
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        jsonfile = os.listdir(self.jsonfiles)[idx]

        # load image as torch.datapoints image
        image_path = os.path.join(self.jpgfiles, self.df['image_name'][idx])

        image    = load_image(image_path)
        image    = dp.Image(image)

        # load boxes
        boxes    = self.df[self.df['image_name'] == self.df['image_name'][idx]]['boxes'].values
        boxes = [np.array(box, dtype = np.float32) for box in boxes]
        boxes = torch.as_tensor(boxes, dtype = torch.float32)
        
        #duck species labels
        labels   = self.df['labels'][self.df['image_name'] == self.df['image_name'][idx]]

        label_dict = {4.0: 'MALL', 1.0: 'AMCO', 3.0: 'GWTE', 6.0: 'NSHO', 2.0: 'GADW', 8.0: 'RNDU', 5.0: 'NOPI', 7.0: 'REDH'}
        
        labels = [0 if label not in label_dict.values() else list(label_dict.keys())[list(label_dict.values()).index(label)] for label in labels]

        target = {}
        target['boxes'] = dp.BoundingBox(boxes, format="XYXY", spatial_size=get_imagesize_from_jsonfile(self.jsonfiles + jsonfile))
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = torch.zeros((len(labels),), dtype=torch.int64)
        
        if self.augment is not None:
            image, target = self.augment(image, target)

        return image, target


    def __len__(self):
        return len(self.df['image_name'].unique())


    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))