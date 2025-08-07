from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.cache_size = 0  # disable cache

# Import necessary packages
import os
import json
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion(); # interactive mode
# Function for reading JSON as dictionary
def read_json(filename: str) -> dict:
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception as e:
        raise Exception(f"Reading {filename} file encountered an error: {e}")
    return data

# Function to create a DataFrame from a list of records
def create_dataframe(data: list) -> pd.DataFrame:
    # Normalize the column levels and create a DataFrame
    return pd.json_normalize(data)

# Main function to iterate over files in directory and add to df
def main():
    # Assign directory and empty list for collecting records
    directory = "C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/Annotations/"  # annotation directory
    records = []
    
    # Iterate over files in directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            # Read the JSON file as python dictionary 
            data = read_json(filename=f)
        
            # Create the dataframe for the array items in annotations key 
            df = create_dataframe(data=data['annotations'])
            df.insert(loc=0, column='img_name', value=f'{f[-30:-5]}.JPG')
        
            df.rename(columns={
                "img_name": "img_name",
                "name": "label",
                "bounding_box.h": "bbox_height",
                "bounding_box.w": "bbox_width",
                "bounding_box.x": "bbox_x_topLeft",
                "bounding_box.y": "bbox_y_topLeft",
                "polygon.paths": "polygon_path"
            }, inplace=True)
            
            # Append the records to the list
            records.append(df)
        else:
            print(f"Skipping non-file: {filename}")

    # Concatenate all records into a single DataFrame
    annos_df = pd.concat(records, ignore_index=True)

    # Convert x, y, h, w to xmin, ymin, xmax, ymax
    annos_df['xmin'] = annos_df['bbox_x_topLeft']
    annos_df['ymin'] = annos_df['bbox_y_topLeft']
    annos_df['xmax'] = annos_df['bbox_x_topLeft'] + annos_df['bbox_width']
    annos_df['ymax'] = annos_df['bbox_y_topLeft'] + annos_df['bbox_height']
  
    # Drop unnecessary columns 
    annos_df = annos_df.drop(columns=['bbox_height', 'bbox_width', 'bbox_x_topLeft', 
                                      'bbox_y_topLeft', 'id', 'slot_names', 'polygon_path'])
        
    return annos_df

if __name__ == "__main__":
    df = main()
    print(df.head())
# Get the unique image names
unique_img_names = df['img_name'].unique()

invalid_img_names = []
for img_name in unique_img_names:
    img_path = f'C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/Images/{img_name}'
    img = Image.open(img_path)
    if img.size == (5184, 3888):
        invalid_img_names.append(img_name)

# load curated images list from file
with open('C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/curated_images.txt', 'r') as f:
    curated_images = f.read().splitlines()

# remove any invalid images from curated images list
curated_images = [img_name for img_name in curated_images if img_name not in invalid_img_names]

# filter df to include only curated images
df = df[df['img_name'].isin(curated_images)]

img_classes_to_remove = ['WTDE', 'TURT', 'NUTR', 'ANHI', 'CAGO', 
                         'DCCO', 'GWFG', 'GBHE', 'COGA', 'PBGR'] # remove images with these classes

for class_label in img_classes_to_remove:
    # Get all image names with the class
    images_with_class = df[df['label'] == class_label]['img_name'].unique()

    # Remove all rows for img
    df = df[~df['img_name'].isin(images_with_class)]

# remove images containing only hens
hen_images_no_other_class = df[(df['label'] == 'Hen') & (~df['img_name'].isin(df[df['label'] != 'Hen']['img_name']))]['img_name'].unique()
df = df[~df['img_name'].isin(hen_images_no_other_class)]

# Separate classes with less than 100 instances
class_counts = df['label'].value_counts()
other_classes = class_counts[class_counts < 100].index.tolist()
positive_classes = class_counts[class_counts >= 100].index.tolist()

# print class counts for each label
print("Number of instances per class in cleaned dataset:")
for label in df['label'].unique():
    print(f'{label}: {len(df[df["label"] == label])}')

# print other and positive classes
print()
print(f'Positive classes: {positive_classes}')

# remove images with other classes
for class_label in other_classes:
    # Get all image names with the class
    images_with_class = df[df['label'] == class_label]['img_name'].unique()

    # Remove all rows for img
    df = df[~df['img_name'].isin(images_with_class)]

# encode labels as int (reserve 0 for 'background')
df['target'] = pd.Categorical(df['label']).codes + 1

# filter out images with invalid bounding boxes
df = df.groupby('img_name').filter(lambda x: ((x['xmin'] < x['xmax']) & (x['ymin'] < x['ymax'])).all())

# Create a dictionary using df['label'] as the keys and df['target'] as the values
label_dict = dict(zip(df['target'], df['label']))

# Drop the original 'label' column from df
df = df.drop(['label'], axis=1)

# Rename 'target' column to 'label'
df.rename(columns={'target': 'label'}, inplace=True)

# Save df as csv in directory
df.to_csv('C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/preprocessed_annotations.csv', index=False)
# Store unique img_names in filtered df as array
img_names = df['img_name'].unique().tolist()

# Create a new directory called 'filtered_images'
filtered_dir = 'C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/filtered_images'
if not os.path.exists(filtered_dir):
    os.makedirs(filtered_dir)
else:
    for file in os.listdir(filtered_dir):
        os.remove(os.path.join(filtered_dir, file))

# Copy images in img_names to new directory
for img in img_names:
    shutil.copy2(f'C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/Images/{img}', filtered_dir)

# import necessary packages
import numpy as np
from collections import defaultdict, Counter
import torchvision
torchvision.disable_beta_transforms_warning()
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torchvision import transforms as _transforms, tv_tensors
import torchvision.transforms.v2 as T
import utils

class MAVdroneDataset(torch.utils.data.Dataset):
    """Dataset Loader for Waterfowl Drone Imagery"""

    def __init__(self, csv_file, root_dir, transforms):
        """
        Arguments:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory containing all images.
            transforms (callable): Transformation to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transforms = transforms
        self.unique_image_names = self.df['img_name'].unique()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.unique_image_names[idx]

        # Isolate first row to prevent multiple instances of the same image
        row = self.df[self.df['img_name'] == image_name].iloc[0]

        image_path = os.path.join(self.root_dir, row['img_name'])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.uint8)
        image = torch.from_numpy(image).permute(2, 0, 1)  # Convert to Tensor

        # Bounding boxes and labels
        boxes = self.df[self.df['img_name'] == image_name][['xmin', 'ymin', 'xmax', 'ymax']].values 
        labels = self.df[self.df['img_name'] == image_name]['label'].values

        labels = torch.as_tensor(labels, dtype=torch.int64)  # (n_objects)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Calculate area
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume no crowd annotations
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Create target dictionary
        target = {
            'boxes': tv_tensors.BoundingBoxes(boxes, format=tv_tensors.BoundingBoxFormat.XYXY, canvas_size=(image.shape[1], image.shape[2])),
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.unique_image_names)

def get_transform(train: bool):
    """
    Args:
        train (bool): Whether the transform is for training or validation/testing.
    """
    transforms_list = [T.ToImage()]
    
    if train:
        transforms_list.extend([
        T.RandomIoUCrop(min_scale=0.6, max_scale=1.8), # zoom in < 1, zoom out > 1
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.2, hue=0.02)], p=0.4),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma = (0.5, 1.0))], p=0.4),
        T.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        T.RandomApply([T.RandomRotation(degrees=10, interpolation=T.InterpolationMode.BILINEAR)], p=0.3),
        T.RandomHorizontalFlip(0.5),
        T.ClampBoundingBoxes(), # Clamp bounding boxes to image boundaries
        T.SanitizeBoundingBoxes(min_size=1, min_area=1) # Sanitize bounding boxes
    ])
    
    transforms_list.extend([
        T.Resize(
            size=(810,),
            max_size=1440,
            interpolation=T.InterpolationMode.BICUBIC
        ),
        T.ToDtype(
            dtype=torch.float32,
            scale=True
        ),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return T.Compose(transforms_list)

# classes are values in label_dict
classes = list(label_dict.values())

# reverse label dictionary for mapping predictions to classes
rev_label_dict = {v: k for k, v in label_dict.items()}

# distinct colors 
bbox_colors = [
    "#FF0000",  # Red
    "#00FF00",  # Green
    "#FFFF00",  # Yellow
    "#FF00FF",  # Magenta
    "#00FFFF",  # Cyan
    "#FFC0CB",  # Pink
    "#FFA500",  # Orange
    "#800080",  # Purple
    "#FFFFFF",  # White
    "#FFD700",  # Gold
]

# label color map for plotting color-coded boxes by class
label_color_map = {k: bbox_colors[i] for i, k in enumerate(label_dict.keys())}

# function for reshaping boxes 
def get_box(boxes):
    boxes = np.array(boxes)
    boxes = boxes.astype('float').reshape(-1, 4)
    if boxes.shape[0] == 1 : return boxes
    return np.squeeze(boxes)


# function for plotting image
def img_show(image, ax = None, figsize = (6, 9)):
    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    ax.xaxis.tick_top()
    ax.imshow(image)
    return ax
 

def plot_bbox(ax, boxes, labels):
    # add box to the image and use label_color_map to color-code by bounding box class if exists else 'black'
    ax.add_patch(plt.Rectangle((boxes[:, 0], boxes[:, 1]), boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1],
                    fill = False,
                    color = label_color_map[labels.item()] if labels.item() in label_color_map else 'black', 
                    linewidth = 1.25))
    # add label text to bounding box using label_dict if label exists else labels
    ax.text(boxes[:, 2], boxes[:, 3], 
            (label_dict[labels.item()] if labels.item() in label_dict else labels.item()),
            fontsize = 8,
            bbox = dict(facecolor = 'white', alpha = 0.8, pad = 0, edgecolor = 'none'),
            color = 'black')


# function for plotting all boxes and labels on the image using get_polygon, img_show, and plot_mask functions
def plot_detections(image, boxes, labels, ax = None):
    ax = img_show(image.permute(1, 2, 0), ax = ax)
    for i in range(len(boxes)):
        box = get_box(boxes[i])
        plot_bbox(ax, box, labels[i])

from sklearn.model_selection import StratifiedKFold

# Set random number generator for reproducible data splits
rng = np.random.default_rng(np.random.MT19937(np.random.SeedSequence(6666)))

# Group annotations by image
image_groups = df.groupby('img_name')

# Create a dictionary to store the class distribution for each image
image_class_distribution = {}

# Populate the dictionary with class distributions
for image_name, group in image_groups:
    labels = group['label'].tolist()
    image_class_distribution[image_name] = labels

# Create a list of all image names and their corresponding labels
all_images = list(image_class_distribution.keys())
all_labels = [image_class_distribution[image] for image in all_images]

# Use the most frequent label for each image for stratification
representative_labels = [max(set(labels), key=labels.count) for labels in all_labels]

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.15
test_ratio = 0.05

# Perform stratified split using StratifiedKFold
skf = StratifiedKFold(n_splits=int(1/test_ratio), shuffle=True, random_state=6666)

train_val_indices, test_indices = next(skf.split(all_images, representative_labels))

# Further split train+val into train and validation sets
train_val_images = [all_images[idx] for idx in train_val_indices]
train_val_labels = [representative_labels[idx] for idx in train_val_indices]

skf_val = StratifiedKFold(n_splits=int(1/(val_ratio/(train_ratio + val_ratio))), shuffle=True, random_state=6666)
train_indices, val_indices = next(skf_val.split(train_val_images, train_val_labels))

# Map image names to unique indices
image_to_unique_index = {image: idx for idx, image in enumerate(df['img_name'].unique())}

# Create lists of unique indices for each split
train_indices = [image_to_unique_index[train_val_images[idx]] for idx in train_indices]
val_indices = [image_to_unique_index[train_val_images[idx]] for idx in val_indices]
test_indices = [image_to_unique_index[all_images[idx]] for idx in test_indices]

# Function to get class distribution
def get_class_distribution(images, image_class_distribution):
    class_counts = defaultdict(int)
    for image in images:
        for label in image_class_distribution[image]:
            class_counts[label] += 1
    return class_counts

# Get train, val, and test images
train_images = [all_images[idx] for idx in train_indices]
val_images = [all_images[idx] for idx in val_indices]
test_images = [all_images[idx] for idx in test_indices]

train_class_distribution = get_class_distribution(train_images, image_class_distribution)
val_class_distribution = get_class_distribution(val_images, image_class_distribution)
test_class_distribution = get_class_distribution(test_images, image_class_distribution)

class_indices = {label: [] for label in df['label'].unique()}

for idx, row in df.iterrows():
    class_indices[row['label']].append(idx)

train_class_distribution = {k: v / len(class_indices[k]) for k, v in train_class_distribution.items()}
val_class_distribution = {k: v / len(class_indices[k]) for k, v in val_class_distribution.items()}
test_class_distribution = {k: v / len(class_indices[k]) for k, v in test_class_distribution.items()}

print("Train class distribution:", dict(sorted(train_class_distribution.items())))
print("Validation class distribution:", dict(sorted(val_class_distribution.items())))
print("Test class distribution:", dict(sorted(test_class_distribution.items())))

def calculate_class_weights(labels, hen_label_int, background_label_int):
    # Count the occurrences of each class
    class_counts = Counter(labels)
    
    # Remove the "Hen" class from the counts
    hen_count = class_counts.pop(hen_label_int, None)
    
    # Identify the count for the second most-frequent class
    second_most_frequent_class_count = max(class_counts.values())
    
    # Calculate the weight for the "Hen" class
    hen_weight = second_most_frequent_class_count / hen_count if hen_count else 1.0

    # Assign weights to all classes (non-Hen)
    class_weights = {label: sum(class_counts.values()) / count for label, count in class_counts.items()}
    
    # Add weight for the "Hen" class and background before normalization
    class_weights[hen_label_int] = hen_weight
    class_weights[background_label_int] = 0.1  

    # Normalize all weights (including background) by dividing by the maximum weight
    max_weight = max(class_weights.values())
    class_weights = {label: weight / max_weight for label, weight in class_weights.items()}
    
    return class_weights

# Store train labels for each image
train_labels = [label for image in train_images for label in image_class_distribution[image]]

# Calculate class weights dynamically
hen_label_int = [key for key, value in label_dict.items() if value == 'Hen'][0]  # Get the integer label for "Hen"
background_label_int = 0  # Assuming background is class 0
class_weights = calculate_class_weights(train_labels, hen_label_int, background_label_int)

# Ensure the background label is included for printing
all_labels = sorted(set([background_label_int] + train_labels))

# Convert class weights to a list in the correct order (with background as the first element)
train_class_weights = [class_weights[label] for label in all_labels]
train_class_weights = torch.tensor(train_class_weights, dtype=torch.float32)

# Print class counts and weights (including background)
print("Train class instances and weights: ")
# Print background label explicitly
print(f"Background: weight = {class_weights[background_label_int]}")
for label in all_labels:
    if label == background_label_int:
        continue
    print(f"{label_dict[label]}: count = {train_labels.count(label)}, weight = {class_weights[label]}")
print()

# Calculate sample weights for each image in the training dataset
train_sample_weights = []
for image_name in train_images:
    labels = image_class_distribution[image_name]
    sample_weight = sum(train_class_weights[all_labels.index(label)] for label in labels) / len(labels)
    train_sample_weights.append(sample_weight)

# Create WeightedRandomSampler
train_sampler = torch.utils.data.WeightedRandomSampler(
    weights=train_sample_weights, num_samples=len(train_sample_weights), replacement=True
)

import torch
import torch.nn as nn
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.models.detection._utils as det_utils
from torchvision.ops import sigmoid_focal_loss, FrozenBatchNorm2d
import torchvision.ops.boxes as box_ops
from typing import Callable, Dict, List, Optional, Tuple
from pt_soft_nms import batched_soft_nms


def _sum(x: List[torch.Tensor]) -> torch.Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes, alpha=0.25, gamma_loss=2.0, prior_probability=0.01, 
                 norm_layer: Optional[Callable[..., nn.Module]] = None, dropout_prob=0.25, class_weights=None, label_smoothing=0.1):
        super().__init__(in_channels, num_anchors, num_classes, prior_probability, norm_layer)
        self.alpha = alpha
        self.gamma_loss = gamma_loss
        self.dropout = nn.Dropout(p=dropout_prob)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []

        cls_logits = head_outputs["cls_logits"]

        for i, (targets_per_image, cls_logits_per_image, matched_idxs_per_image) in enumerate(zip(targets, cls_logits, matched_idxs)):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target += self.label_smoothing / (self.num_classes - 1) # smoothing for negative classes
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0 - self.label_smoothing # smoothing for positive classes

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # get the class weights for the valid indices
            if self.class_weights is not None:
                valid_labels = targets_per_image["labels"][matched_idxs_per_image[valid_idxs_per_image]]
                weights = self.class_weights.to(valid_labels.device)[valid_labels]
            else:
                weights = torch.ones(cls_logits_per_image[valid_idxs_per_image].shape[0], dtype=torch.float32, device=cls_logits_per_image.device)

            # compute the classification loss with custom alpha, gamma_loss, and class weights
            losses.append(
                (sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    alpha=self.alpha,
                    gamma=self.gamma_loss,
                    reduction="none",
                ) * weights.unsqueeze(1)).sum() / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)
    
    def forward(self, x):
        all_cls_logits = []
        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.dropout(cls_logits)  # Apply dropout
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)


        return torch.cat(all_cls_logits, dim=1)


class CustomRetinaNetRegressionHead(RetinaNetRegressionHead):
    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., nn.Module]] = None, _loss_type="smooth_l1", beta_loss=0.5, lambda_loss=1.0, dropout_prob=0.25):
        super().__init__(in_channels, num_anchors, norm_layer)
        self._loss_type = _loss_type
        self.beta_loss = beta_loss # beta < 1 helps counter early plateauing
        self.lambda_loss = lambda_loss # lambda > 1 places more emphasis on localization loss
        self.dropout = nn.Dropout(p=dropout_prob)
    
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor], List[torch.Tensor], List[torch.Tensor]) -> torch.Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the loss
            losses.append(
                    det_utils._box_loss(
                    self._loss_type,
                    self.box_coder,
                    anchors_per_image,
                    matched_gt_boxes_per_image,
                    bbox_regression_per_image,
                    cnf={'beta': self.beta_loss}, 
                ) * self.lambda_loss / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))
    
    def forward(self, x):
        all_bbox_regression = []
        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.dropout(bbox_regression)  # Apply dropout
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

class CustomRetinaNet(RetinaNet):
    def __init__(self, backbone, num_classes, min_size, max_size, image_mean, image_std, score_thresh, detections_per_img, 
                 fg_iou_thresh, bg_iou_thresh, topk_candidates, nms_score, nms_sigma):
        super().__init__(backbone, num_classes=num_classes, 
                         min_size=min_size, 
                         max_size=max_size, 
                         image_mean=image_mean, 
                         image_std=image_std, 
                         score_thresh=score_thresh, 
                         nms_thresh=None, 
                         detections_per_img=detections_per_img, 
                         fg_iou_thresh=fg_iou_thresh, 
                         bg_iou_thresh=bg_iou_thresh, 
                         topk_candidates=topk_candidates)
        # Store the new NMS parameters.
        self.nms_score = nms_score
        self.nms_sigma = nms_sigma

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[torch.Tensor]], List[List[torch.Tensor]], List[Tuple[int, int]]) -> List[Dict[str, torch.Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)
        detections: List[Dict[str, torch.Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = det_utils._topk_min(topk_idxs, self.topk_candidates, 0)
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs], anchors_per_level[anchor_idxs]
                )
                boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # First pass: Apply soft-NMS across all classes
            keep = batched_soft_nms(
                image_boxes, 
                image_scores, 
                image_labels, 
                sigma=self.nms_sigma, 
                score_threshold=self.nms_score
            )
            
            # Keep only the best detections after first pass
            keep_boxes = image_boxes[keep]
            keep_scores = image_scores[keep]
            keep_labels = image_labels[keep]
            
            # Sort by score for second pass processing
            score_order = keep_scores.argsort(descending=True)
            keep_boxes = keep_boxes[score_order]
            keep_scores = keep_scores[score_order]
            keep_labels = keep_labels[score_order]
            
            # Remove any remaining overlapping boxes regardless of class
            final_keep = []
            remaining_boxes_mask = torch.ones(len(keep_boxes), dtype=torch.bool, device=keep_boxes.device)
            
            for i in range(len(keep_boxes)):
                if remaining_boxes_mask[i]:
                    final_keep.append(i)
                    
                    # Calculate IoU between this box and all remaining boxes
                    current_box = keep_boxes[i:i+1]
                    ious = box_ops.box_iou(current_box, keep_boxes[remaining_boxes_mask])[0]
                    
                    # Identify boxes with high overlap (adjust this threshold as needed)
                    high_overlap_indices = torch.where(ious > 0.8)[0]  # Stricter overlap threshold for second pass
                    
                    # Convert relative indices to absolute indices in the remaining_boxes_mask
                    absolute_indices = torch.where(remaining_boxes_mask)[0][high_overlap_indices]
                    
                    # Remove the current index from the list of boxes to suppress
                    if len(absolute_indices) > 0 and absolute_indices[0] == i:
                        absolute_indices = absolute_indices[1:]
                    
                    # Set these boxes to be removed
                    remaining_boxes_mask[absolute_indices] = False
            
            final_keep = torch.tensor(final_keep, device=keep_boxes.device, dtype=torch.long)
            
            # Limit to max detections per image
            if len(final_keep) > self.detections_per_img:
                final_keep = final_keep[:self.detections_per_img]
                
            detections.append(
                {
                    "boxes": keep_boxes[final_keep],
                    "scores": keep_scores[final_keep],
                    "labels": keep_labels[final_keep],
                }
            )

        return detections


def get_retinanet_model(depth, num_classes=10, min_size=810, max_size=1440, image_mean=[0, 0, 0], image_std=[1, 1, 1], score_thresh=0.1,
                        detections_per_img=200, fg_iou_thresh=0.6, bg_iou_thresh=0.5, topk_candidates=200, alpha=0.75, gamma_loss=3.0, 
                        class_weights=None, beta_loss=0.5, lambda_loss=1.5, dropout_prob=0.25, nms_score=0.25, nms_sigma=0.5):
    
    trainable_backbone_layers = 0 

    # Create the backbone with FPN
    if depth == 18:
        backbone = resnet_fpn_backbone(backbone_name='resnet18', 
                                       weights=torchvision.models.ResNet18_Weights.DEFAULT, 
                                       trainable_layers=trainable_backbone_layers)
    elif depth == 34:
        backbone = resnet_fpn_backbone(backbone_name='resnet34', 
                                       weights=torchvision.models.ResNet34_Weights.DEFAULT,
                                       trainable_layers=trainable_backbone_layers)
    elif depth == 50:
        backbone = resnet_fpn_backbone(backbone_name='resnet50', 
                                       weights=torchvision.models.ResNet50_Weights.DEFAULT,
                                       trainable_layers=trainable_backbone_layers)
    elif depth == 101:
        backbone = resnet_fpn_backbone(backbone_name='resnet101', 
                                       weights=torchvision.models.ResNet101_Weights.DEFAULT, 
                                       trainable_layers=trainable_backbone_layers)
    elif depth == 152:
        backbone = resnet_fpn_backbone(backbone_name='resnet152', 
                                       weights=torchvision.models.ResNet152_Weights.DEFAULT, 
                                       trainable_layers=trainable_backbone_layers)
    else:
        raise ValueError("Unsupported model depth")

    # Create the RetinaNet model with the custom backbone.
    model = CustomRetinaNet(backbone, 
                            num_classes=num_classes,
                            min_size=min_size,  # same size as resize in transform to keep aspect ratio
                            max_size=max_size,
                            image_mean=image_mean,
                            image_std=image_std,
                            score_thresh=score_thresh, 
                            detections_per_img=detections_per_img,
                            fg_iou_thresh=fg_iou_thresh,
                            bg_iou_thresh=bg_iou_thresh,
                            topk_candidates=topk_candidates,
                            nms_score=nms_score,
                            nms_sigma=nms_sigma
                           )

    # Replace the classification head with the custom one
    in_channels = model.head.classification_head.cls_logits.in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = CustomRetinaNetClassificationHead(in_channels, 
                                                                       num_anchors, 
                                                                       num_classes, 
                                                                       alpha=alpha, 
                                                                       gamma_loss=gamma_loss, 
                                                                       dropout_prob=dropout_prob,
                                                                       class_weights=class_weights)

    # Replace the regression head with the custom one
    model.head.regression_head = CustomRetinaNetRegressionHead(in_channels, 
                                                               num_anchors, 
                                                               _loss_type="smooth_l1",
                                                               beta_loss=beta_loss,
                                                               lambda_loss=lambda_loss,
                                                               dropout_prob=dropout_prob)

    model.anchor_generator = AnchorGenerator(sizes=((24, 32, 40), (48, 64, 80), (96, 128, 160), (192, 256, 320), (472, 536, 600)), 
                                             aspect_ratios=((0.75, 1.15, 1.8), (0.75, 1.15, 1.8), (0.75, 1.15, 1.8), (0.75, 1.15, 1.8), (0.75, 1.15, 1.8)))

    return model

print(get_retinanet_model(depth=50))

del all_images, all_labels, background_label_int, class_counts, class_indices, class_label, class_weights, curated_images, df, file, group, hen_images_no_other_class, hen_label_int, idx, image_class_distribution, image_groups, image_name, images_with_class, img, img_classes_to_remove, img_name, img_names, img_path, invalid_img_names, label, labels, other_classes, positive_classes, representative_labels, row, sample_weight, test_class_distribution, test_images, test_ratio, train_class_distribution, train_images, train_labels, train_ratio, train_sample_weights, train_val_images, train_val_indices, train_val_labels, unique_img_names, val_class_distribution, val_images, val_ratio

from datetime import datetime
import torch
import gc
from pathlib import Path
import ray.cloudpickle as pickle
from concurrent.futures import ThreadPoolExecutor
import random
from torch_lr_finder import LRFinder, TrainDataLoaderIter, ValDataLoaderIter
from engine_gradientAccumulation import train_one_epoch, evaluate
from coco_utils import get_coco_api_from_dataset

# Set random seed for reproducible training
def set_seed(seed):
    import torch, random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def extract_per_class_metrics(coco_eval, coco_gt):
    """
    Extract per-class metrics at different IoU thresholds.
    Returns precision and recall for IoU@[0.5:0.95], IoU@0.5, and IoU@0.75
    """
    per_class_metrics = {}

    # Create a list of category IDs in the order they appear in the evaluation results
    cat_ids = list(coco_gt.cats.keys())
    cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

    for cat_id, idx in cat_id_to_index.items():
        try:
            # Get precision array for different IoU thresholds
            precision_all = coco_eval.coco_eval['bbox'].eval['precision']  # [T, R, K, A, M]
            recall_all = coco_eval.coco_eval['bbox'].eval['recall']      # [T, K, A, M]
            
            # Extract metrics for each IoU threshold
            # For area range 'all' (0) and max detections 100 (2)
            precision_50_95 = precision_all[:, :, idx, 0, 2].mean()  # IoU@[0.5:0.95]
            precision_50 = precision_all[0, :, idx, 0, 2].mean()     # IoU@0.5
            precision_75 = precision_all[5, :, idx, 0, 2].mean()     # IoU@0.75
            
            recall_50_95 = recall_all[:, idx, 0, 2].mean()          # IoU@[0.5:0.95]
            recall_50 = recall_all[0, idx, 0, 2]                    # IoU@0.5
            recall_75 = recall_all[5, idx, 0, 2]                    # IoU@0.75

            per_class_metrics[cat_id] = {
                'precision_50': precision_50,
                'recall_50': recall_50,
                'f1_50': calculate_f1_score(precision_50, recall_50),
                'precision_75': precision_75,
                'recall_75': recall_75,
                'f1_75': calculate_f1_score(precision_75, recall_75),
                'precision_50_95': precision_50_95,
                'recall_50_95': recall_50_95,
                'f1_50_95': calculate_f1_score(precision_50_95, recall_50_95),
            }

        except IndexError as e:
            print(f"IndexError for category ID {cat_id}: {e}")
            continue

    return per_class_metrics

def adjust_trainable_layers(model, trainable_layers):
    """
    Adjust the trainable layers in the RetinaNet backbone (model.backbone.body).
    Unfreeze the last `trainable_layers` residual blocks and replace their FrozenBatchNorm2d layers.
    When trainable_layers=5, also unfreeze conv1 and replace bn1 with trainable BatchNorm2d.
    """
    def convert_frozen_bn(frozen_bn):
        device = frozen_bn.running_mean.device  # Get the device of the frozen BN layer
        num_features = frozen_bn.weight.shape[0]
        bn = torch.nn.BatchNorm2d(num_features)
        bn = bn.to(device)
        # Initialize with existing stats.
        bn.running_mean = frozen_bn.running_mean.clone()
        bn.running_var = frozen_bn.running_var.clone()
        torch.nn.init.normal_(bn.weight, mean=1.0, std=0.02)
        torch.nn.init.constant_(bn.bias, 0)
        return bn

    # Collect backbone blocks.
    backbone_layers = []
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model.backbone.body, layer_name):
            backbone_layers.append(getattr(model.backbone.body, layer_name))
    
    if trainable_layers > 5:
        print(f"Requested trainable_layers ({trainable_layers}) exceeds available layers (5). Using 5 instead.")
        trainable_layers = 5

    # Unfreeze the last `trainable_layers` blocks.
    for block in backbone_layers[-trainable_layers:]:
        for param in block.parameters():
            param.requires_grad = True

    # Replace FrozenBatchNorm2d layers in these blocks.
    for name, module in model.backbone.body.named_modules():
        if isinstance(module, FrozenBatchNorm2d):
            if 'layer' in name:
                layer_num = int(name.split('.')[0][-1])
                if layer_num > (4 - trainable_layers):
                    parent_name = '.'.join(name.split('.')[:-1])
                    module_name = name.split('.')[-1]
                    parent = dict(model.backbone.body.named_modules())[parent_name]
                    setattr(parent, module_name, convert_frozen_bn(module))
            elif trainable_layers == 5 and name == 'bn1':
                model.backbone.body.bn1 = convert_frozen_bn(module)
                model.backbone.body.conv1.weight.requires_grad = True

class BestTrial:
    def __init__(self):
        # Initialize default config first
        self.config = {
            "lr": 0.001181,
            "resnet_depth": 50,
            "momentum": 0.973758,
            "weight_decay": 1.42512e-05,
            "alpha": 0.808413,
            "gamma_loss": 3.46467,
            "dropout": 0.336394,
            "score_thresh": 0.5,
            "fg_iou_thresh": 0.6,
            "bg_iou_thresh": 0.5,
            "beta_loss": 0.6,
            "lambda_loss": 1.2,
            "nms_score": 0.6,
            "nms_sigma": 0.5,
            "class_weights": train_class_weights,
            "train_sampler": train_sampler,
        }

        # self.config["lr"] = self.train_lr_finder()

    def train_lr_finder(self):
        class CustomTrainDataLoaderIter(TrainDataLoaderIter):
            def inputs_labels_from_batch(self, batch_data):
                inputs = [image.to('cuda:0') for image in batch_data[0]]
                labels = [{k: v.to('cuda:0') for k, v in t.items()} for t in batch_data[1]]
                return inputs, labels

        class CustomValDataLoaderIter(ValDataLoaderIter):
            def __iter__(self):
                self._iterator = iter(self.data_loader)
                self.run_counter = 0
                return self

            def __next__(self):
                try:
                    self.run_counter += 1
                    return super().__next__()
                except StopIteration:
                    # Reset if exhausted and then return next batch
                    self._iterator = iter(self.data_loader)
                    self.run_counter = 0
                    return super().__next__()

            def inputs_labels_from_batch(self, batch_data):
                inputs = [image.to("cuda:0") for image in batch_data[0]]
                labels = [{k: v.to("cuda:0") for k, v in t.items()} for t in batch_data[1]]
                return inputs, labels

        dataset_train = MAVdroneDataset(
            csv_file='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/preprocessed_annotations.csv',
            root_dir='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/filtered_images/',
            transforms=get_transform(train=True))
        dataset_train = torch.utils.data.Subset(dataset_train, train_indices)
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8,
                                                        sampler=self.config["train_sampler"],
                                                        collate_fn=utils.collate_fn,
                                                        num_workers=0, pin_memory=True)
        train_iter = CustomTrainDataLoaderIter(data_loader_train)

        dataset_val = MAVdroneDataset(
            csv_file='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/preprocessed_annotations.csv',
            root_dir='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/filtered_images/',
            transforms=get_transform(train=False))
        dataset_val = torch.utils.data.Subset(dataset_val, val_indices)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False,
            collate_fn=utils.collate_fn, num_workers=0, pin_memory=True)
        val_iter = CustomValDataLoaderIter(data_loader_val)

        model = get_retinanet_model(
            depth=self.config["resnet_depth"],
            num_classes=len(self.config["class_weights"]),
            score_thresh=self.config["score_thresh"],
            detections_per_img=200,
            fg_iou_thresh=self.config["fg_iou_thresh"],
            bg_iou_thresh=self.config["bg_iou_thresh"],
            topk_candidates=200,
            alpha=self.config["alpha"],
            gamma_loss=self.config["gamma_loss"],
            class_weights=None,
            beta_loss=self.config["beta_loss"],
            lambda_loss=self.config["lambda_loss"],
            dropout_prob=self.config["dropout"],
            nms_score=self.config["nms_score"],
            nms_sigma=self.config["nms_sigma"]
        ).to('cuda:0')

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=1e-7, momentum=self.config["momentum"], weight_decay=self.config["weight_decay"]
        )

        grad_scaler = torch.GradScaler()

        class CustomLRFinder(LRFinder):
            def __init__(self, model, optimizer, criterion, device=None, amp_backend="native", amp_config=None, grad_scaler=None):
                super().__init__(model, optimizer, criterion, device)
                self.amp_backend = amp_backend
                self.amp_config = amp_config
                self.grad_scaler = grad_scaler or torch.GradScaler()

            def _train_batch(self, train_iter, accumulation_steps, non_blocking_transfer=True):
                self.model.train()
                total_loss = 0

                self.optimizer.zero_grad()
                for _ in range(accumulation_steps):
                    inputs, labels = next(train_iter)
                    inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)

                    with torch.autocast(device_type="cuda:0"):
                        outputs = self.model(inputs, labels)
                        loss = sum(loss for loss in outputs.values())
                    loss /= accumulation_steps
                    self.grad_scaler.scale(loss).backward()
                    total_loss += loss
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                return total_loss.item()

            def _validate(self, val_iter, non_blocking_transfer=True):
                self.model.train()   # FORCE training mode here!
                inputs, labels = next(val_iter)
                inputs, labels = self._move_to_device(inputs, labels, non_blocking=non_blocking_transfer)
                with torch.no_grad(), torch.autocast(device_type="cuda:0"):
                    outputs = self.model(inputs, labels)
                    loss = sum(loss for loss in outputs.values())
                return loss.item()

        lr_finder = CustomLRFinder(model, optimizer, None, device='cuda:0', amp_backend='torch', amp_config=None, grad_scaler=grad_scaler)
        lr_finder.range_test(train_iter, val_iter, end_lr=1, num_iter=450, step_mode='exp', accumulation_steps=1)
        suggested_lr = lr_finder.plot(suggest_lr=True)

        lr_finder.reset()

        # return default if torch lr finder fails
        try:
            if isinstance(suggested_lr, tuple):
                axes, suggested_lr_value = suggested_lr
                return suggested_lr_value
            else:
                raise ValueError(f"Unexpected return type from plot method: {type(suggested_lr)}")
        except ValueError as e:
            print(f"Error during learning rate finding: {e}")
            # Return a default learning rate if an error occurs
            return 5e-4

if __name__ == "__main__":
    best_trial = BestTrial()
    print("Best trial config:")
    for key, value in best_trial.config.items():
        print(f"{key}: {value}")

from torch.utils.tensorboard import SummaryWriter
from coco_utils import get_coco_api_from_dataset
import math

def visualize_predictions(model, data_loader, device, epoch, num_samples=2, label_dict=None, bbox_colors=None, plot=False,
                          output_dir='prediction_visualizations'):
    
    # Define ImageNet normalization parameters
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    def denormalize(img_tensor):
        img = img_tensor.clone().cpu().numpy().transpose(1, 2, 0)
        img = img * imagenet_std + imagenet_mean
        return np.clip(img, 0, 1)
    
    model.eval()
    
    # Get dataset length and generate random indices
    dataset_size = len(data_loader.dataset)
    random_indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    with torch.no_grad():
        for idx in random_indices:
            # Get the batch index and position within batch
            batch_idx = idx // data_loader.batch_size
            pos_in_batch = idx % data_loader.batch_size
            
            for i, (images, targets) in enumerate(data_loader):
                if i == batch_idx:
                    images = [img.to(device) for img in images]
                    outputs = model(images)
                    
                    b = pos_in_batch
                    if b >= len(images):  # Skip if batch is smaller than expected
                        continue
                        
                    img = denormalize(images[b])
                    gt_boxes = targets[b]['boxes'].cpu().numpy()
                    gt_labels = targets[b]['labels'].cpu().numpy()
                    pred_boxes = outputs[b]['boxes'].cpu().numpy()
                    pred_labels = outputs[b]['labels'].cpu().numpy()
                    pred_scores = outputs[b]['scores'].cpu().numpy()
                    
                    # Get original image dimensions for high-res output
                    height, width = img.shape[0], img.shape[1]
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width/50, height/100))  # Scale figure appropriately
                    
                    # Plot ground truth boxes
                    ax1.imshow(img)
                    for box, label in zip(gt_boxes, gt_labels):
                        gt_text = label_dict[label] if label_dict is not None and label in label_dict else str(label)
                        gt_color = bbox_colors[label] if bbox_colors is not None and label < len(bbox_colors) else 'black'
                        rect = plt.Rectangle((box[0], box[1]),
                                          box[2] - box[0],
                                          box[3] - box[1],
                                          linewidth=1.25, edgecolor=gt_color, facecolor='none')
                        ax1.add_patch(rect)
                        ax1.text(box[2], box[3],
                                f'{gt_text}',
                                fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.8, pad=0, edgecolor='none'),
                                color='black')
                    ax1.set_title(f'Ground Truth\nEpoch {epoch}, Image {idx}')
                    
                    # Plot predicted boxes
                    ax2.imshow(img)
                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        pred_text = f"{label_dict[label] if label_dict is not None and label in label_dict else label}: {score:.2f}"
                        pred_color = bbox_colors[label] if bbox_colors is not None and label < len(bbox_colors) else 'black'
                        rect = plt.Rectangle((box[0], box[1]),
                                          box[2] - box[0],
                                          box[3] - box[1],
                                          linewidth=1.25, edgecolor=pred_color, facecolor='none')
                        ax2.add_patch(rect)
                        ax2.text(box[2], box[3],
                                pred_text,
                                fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.8, pad=0, edgecolor='none'),
                                color='black')
                    ax2.set_title(f'Predictions\nEpoch {epoch}, Image {idx}')
                    
                    plt.tight_layout()
                    
                    filename = f"epoch{epoch:03d}_img{idx:05d}.png"
                    filepath = os.path.join(output_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    if plot:
                        plt.show()
                    plt.close()
                    
                    print(f"Saved visualization to: {filepath}")
                    break

# uncomment if running from fresh kernel
def create_coco_datasets(train_dataset, val_dataset, test_dataset):
        with ThreadPoolExecutor(max_workers=3) as executor:
            train_future = executor.submit(get_coco_api_from_dataset, train_dataset)
            val_future = executor.submit(get_coco_api_from_dataset, val_dataset)
            test_future = executor.submit(get_coco_api_from_dataset, test_dataset)
            train_coco_ds = train_future.result()
            val_coco_ds = val_future.result()
            test_coco_ds = test_future.result()
        return train_coco_ds, val_coco_ds, test_coco_ds

def main(best_trial):
    set_seed(6666)
    print(best_trial.config)
    print()

    training_steps = [
        {"step": 0, "batch_size": 8, "print_freq": 25, "accumulation_steps": 1, "trainable_layers": 0, "improvement_threshold": 0.01, "variance_threshold": 1e-4}, 
        {"step": 1, "batch_size": 8, "print_freq": 25, "accumulation_steps": 2, "trainable_layers": 1, "improvement_threshold": 0.008, "variance_threshold": 5e-5}, 
        {"step": 2, "batch_size": 8, "print_freq": 25, "accumulation_steps": 4, "trainable_layers": 2, "improvement_threshold": 0.005, "variance_threshold": 2.5e-5}, 
        {"step": 3, "batch_size": 8, "print_freq": 25, "accumulation_steps": 8, "trainable_layers": 3, "improvement_threshold": 0.003, "variance_threshold": 1e-5}, 
        {"step": 4, "batch_size": 8, "print_freq": 25, "accumulation_steps": 16, "trainable_layers": 4, "improvement_threshold": 0.002, "variance_threshold": 5e-6}, 
        {"step": 5, "batch_size": 8, "print_freq": 25, "accumulation_steps": 32, "trainable_layers": 5, "improvement_threshold": 0.001, "variance_threshold": 2.5e-6}, # bs 256
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f'C:/Users/exx/Documents/GitHub/SSD_VGG_PyTorch/runs/RetinaNet/{current_datetime}')
    checkpoint_dir = Path(f'./checkpoints/{current_datetime}')
    checkpoint_dir.mkdir(exist_ok=True)

    # Dataset setup remains unchanged
    dataset = MAVdroneDataset(
        csv_file='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/preprocessed_annotations.csv',
        root_dir='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/filtered_images/',
        transforms=get_transform(train=True)
    )
    dataset_val = MAVdroneDataset(
        csv_file='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/preprocessed_annotations.csv',
        root_dir='C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/filtered_images/',
        transforms=get_transform(train=False)
    )

    dataset = torch.utils.data.Subset(dataset, train_indices)
    train_dataset_eval = torch.utils.data.Subset(dataset_val, train_indices)
    train_data_loader_eval = torch.utils.data.DataLoader(
        train_dataset_eval, batch_size=1, shuffle=False,
        collate_fn=utils.collate_fn, num_workers=0, pin_memory=True
    )

    dataset_val = torch.utils.data.Subset(dataset_val, val_indices)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False,
        collate_fn=utils.collate_fn, num_workers=0, pin_memory=True
    )

    train_coco_ds, val_coco_ds, test_coco_ds = create_coco_datasets(
        train_dataset=train_dataset_eval, 
        val_dataset=dataset_val, 
        test_dataset=dataset_val
    )

    # Instantiate model and initialize optimizer with default parameters
    model = get_retinanet_model(
        depth=best_trial.config["resnet_depth"],
        num_classes=len(best_trial.config["class_weights"]),
        score_thresh=best_trial.config["score_thresh"],
        detections_per_img=200,
        fg_iou_thresh=best_trial.config["fg_iou_thresh"],
        bg_iou_thresh=best_trial.config["bg_iou_thresh"],
        topk_candidates=200, 
        alpha=best_trial.config["alpha"], 
        gamma_loss=best_trial.config["gamma_loss"],
        dropout_prob=best_trial.config["dropout"],
        beta_loss=best_trial.config["beta_loss"],
        lambda_loss=best_trial.config["lambda_loss"],
        class_weights=None,
        nms_score=best_trial.config["nms_score"],
        nms_sigma=best_trial.config["nms_sigma"]
    )
    model.to(device)

    # Initialize with default parameters - we'll update this in each step
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=best_trial.config["lr"],
        momentum=best_trial.config["momentum"],
        weight_decay=best_trial.config["weight_decay"],
        nesterov=True
    )
    
    # Training loop state
    start_epoch = 0
    current_step = 0
    step_epoch_counter = 0

    # Main training loop
    while current_step < len(training_steps):
        ts = training_steps[current_step]
        batch_size = ts["batch_size"]
        print_freq = ts["print_freq"]
        accumulation_steps = ts["accumulation_steps"]
        backbone_layers = ts["trainable_layers"]
        improvement_threshold = ts["improvement_threshold"]
        variance_threshold = ts["variance_threshold"]

        # Calculate scaled learning rate for this step
        scaled_lr = best_trial.config["lr"] * math.sqrt((batch_size / training_steps[0]["batch_size"]) * accumulation_steps)

        # Adjust the trainable layers for this step
        adjust_trainable_layers(model, backbone_layers)

        # Create new optimizer for the current step
        step_optimizer = optimizer  # Save current optimizer to transfer state
        params_new = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params_new,
            lr=scaled_lr,
            momentum=step_optimizer.param_groups[0]['momentum'],
            weight_decay=step_optimizer.param_groups[0]['weight_decay'],
            nesterov=step_optimizer.param_groups[0]['nesterov']
        )
        
        # Efficiently transfer optimizer state from previous step
        old_state = step_optimizer.state_dict()["state"]
        for group in optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid in old_state:
                    optimizer.state[p] = old_state[pid]

        # Create data loader for this step
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, 
            sampler=best_trial.config["train_sampler"],
            collate_fn=utils.collate_fn, 
            num_workers=0, pin_memory=True
        )

        print(f'Training step: {ts["step"]}, effective batch size: {batch_size * accumulation_steps}, scaled lr: {scaled_lr:.6f}')
        print()
        
        # Early stopping logic
        window_loss = []
        window_f1 = []
        window_size = 5
        minimum_epochs = 15
        alpha = 0.1
        patience = 3 if ts["step"] < 3 else 5
        ema_loss = None
        ema_f1 = None
        non_improving_counter = 0

        # Step training loop
        while True:
            print(f"Epoch {start_epoch}, Step: {ts['step']}, Memory: {torch.cuda.memory_allocated(device)} bytes")
            print()

            train_metric_logger, val_metric_logger = train_one_epoch(
                model, optimizer, data_loader, device, start_epoch,
                print_freq, accumulation_steps, data_loader_val, step_epoch_counter
            )
            print()
            
            train_coco_evaluator, val_coco_evaluator = evaluate(
                model, data_loader_val, val_coco_ds, device, train_data_loader_eval, train_coco_ds
            )
            print()

            # Process metrics
            train_class_metrics = extract_per_class_metrics(train_coco_evaluator, train_coco_ds)
            val_class_metrics = extract_per_class_metrics(val_coco_evaluator, val_coco_ds)
            train_class_metrics = {label_dict[k]: v for k, v in train_class_metrics.items()}
            val_class_metrics = {label_dict[k]: v for k, v in val_class_metrics.items()}

            # Print per-class metrics
            print("Training Class Metrics:")
            for name, m in train_class_metrics.items():
                print()
                print(name)
                print(f"AP@0.50: {m['precision_50']:.4f},\n AP@0.75: {m['precision_75']:.4f},\n AP@[0.50:0.95]: {m['precision_50_95']:.4f}")
                print(f"AR@0.50: {m['recall_50']:.4f},\n AR@0.75: {m['recall_75']:.4f},\n AR@[0.50:0.95]: {m['recall_50_95']:.4f}")
                print(f"F1@0.50: {m['f1_50']:.4f},\n F1@0.75: {m['f1_75']:.4f},\n F1@[0.50:0.95]: {m['f1_50_95']:.4f}")
            print()
            print("\nValidation Class Metrics:")
            for name, m in val_class_metrics.items():
                print()
                print(name)
                print(f"AP@0.50: {m['precision_50']:.4f},\n AP@0.75: {m['precision_75']:.4f},\n AP@[0.50:0.95]: {m['precision_50_95']:.4f}")
                print(f"AR@0.50: {m['recall_50']:.4f},\n AR@0.75: {m['recall_75']:.4f},\n AR@[0.50:0.95]: {m['recall_50_95']:.4f}")
                print(f"F1@0.50: {m['f1_50']:.4f},\n F1@0.75: {m['f1_75']:.4f},\n F1@[0.50:0.95]: {m['f1_50_95']:.4f}")
            print()

            # Visualize predictions periodically
            if start_epoch % 5 == 0:
                visualize_predictions(model, data_loader_val, device, start_epoch, num_samples=3, 
                                      label_dict=label_dict, bbox_colors=bbox_colors, plot=False,
                                      output_dir=f'C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/prediction_visualizations/{current_datetime}')

            # Calculate current metrics
            current_loss = val_metric_logger.loss.avg
            current_f1 = calculate_f1_score(val_coco_evaluator.coco_eval['bbox'].stats[0],
                                           val_coco_evaluator.coco_eval['bbox'].stats[8])
            
            # Update metric windows for EMA calculation
            window_loss.append(current_loss)
            window_f1.append(current_f1)
            if len(window_loss) > window_size:
                window_loss.pop(0)
            if len(window_f1) > window_size:
                window_f1.pop(0)

            # Save checkpoint to disk
            checkpoint = {
                "epoch": start_epoch,
                "current_step": current_step,
                "step_epoch_counter": step_epoch_counter,
                "train_loss": train_metric_logger.loss.avg,
                "val_loss": current_loss,
                "train_bbox_loss": train_metric_logger.bbox_regression.avg,
                "val_bbox_loss": val_metric_logger.bbox_regression.avg,
                "train_class_loss": train_metric_logger.classification.avg,
                "val_class_loss": val_metric_logger.classification.avg,
                "train_mAP": train_coco_evaluator.coco_eval['bbox'].stats[0],
                "train_mAR": train_coco_evaluator.coco_eval['bbox'].stats[8],
                "val_mAP": val_coco_evaluator.coco_eval['bbox'].stats[0],
                "val_mAR": val_coco_evaluator.coco_eval['bbox'].stats[8],
                "train_f1": calculate_f1_score(train_coco_evaluator.coco_eval['bbox'].stats[0],
                                            train_coco_evaluator.coco_eval['bbox'].stats[8]),
                "val_f1": current_f1,
                "config": best_trial.config
            }

            # Save checkpoint metrics
            checkpoint_metrics_path = checkpoint_dir / f"{current_datetime}_epoch{start_epoch}_metrics.pth"
            torch.save(checkpoint, checkpoint_metrics_path)

            # Save model state
            model_path = checkpoint_dir / f"{current_datetime}_epoch{start_epoch}_model.pth"
            torch.save({
                'epoch': start_epoch,
                'current_step': current_step,
                'step_epoch_counter': step_epoch_counter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

            # Calculate train/val ratios for monitoring
            train_val_ratio = {
                'loss': checkpoint["train_loss"] / (checkpoint["val_loss"] if checkpoint["val_loss"] != 0 else 1.0),
                'mAP': checkpoint["train_mAP"] / (checkpoint["val_mAP"] if checkpoint["val_mAP"] != 0 else 1.0),
                'mAR': checkpoint["train_mAR"] / (checkpoint["val_mAR"] if checkpoint["val_mAR"] != 0 else 1.0),
                "f1": checkpoint["train_f1"] / (checkpoint["val_f1"] if checkpoint["val_f1"] != 0 else 1.0)
            }

            # TensorBoard logging
            writer.add_scalar('Loss/Train', float(checkpoint["train_loss"]), start_epoch)
            writer.add_scalar('Loss/Val', float(checkpoint["val_loss"]), start_epoch)
            writer.add_scalar('Box Loss/Train', float(checkpoint["train_bbox_loss"]), start_epoch)
            writer.add_scalar('Box Loss/Val', float(checkpoint["val_bbox_loss"]), start_epoch)
            writer.add_scalar('Class Loss/Train', float(checkpoint["train_class_loss"]), start_epoch)
            writer.add_scalar('Class Loss/Val', float(checkpoint["val_class_loss"]), start_epoch)
            writer.add_scalar('mAP/Train', float(checkpoint["train_mAP"]), start_epoch)
            writer.add_scalar('mAP/Val', float(checkpoint["val_mAP"]), start_epoch)
            writer.add_scalar('mAR/Train', float(checkpoint["train_mAR"]), start_epoch)
            writer.add_scalar('mAR/Val', float(checkpoint["val_mAR"]), start_epoch)
            writer.add_scalar('F1/Train', float(checkpoint["train_f1"]), start_epoch)
            writer.add_scalar('F1/Val', float(checkpoint["val_f1"]), start_epoch)
            writer.add_scalar('Ratios/loss', train_val_ratio['loss'], start_epoch)
            writer.add_scalar('Ratios/mAP', train_val_ratio['mAP'], start_epoch)
            writer.add_scalar('Ratios/mAR', train_val_ratio['mAR'], start_epoch)
            writer.add_scalar('Ratios/f1', train_val_ratio['f1'], start_epoch)

            # Print current metrics
            print(f"Epoch {start_epoch}: Current Loss = {current_loss:.4f},", end=" ")

            # Early stopping check
            if step_epoch_counter >= minimum_epochs and len(window_loss) == window_size:
                if ema_loss is None:
                    ema_loss = current_loss
                    ema_f1 = current_f1
                    relative_improvement = 1.0
                    relative_f1_improvement = 1.0
                else:
                    # Update EMAs
                    prev_ema = ema_loss
                    prev_f1_ema = ema_f1
                    ema_loss = alpha * current_loss + (1 - alpha) * prev_ema
                    ema_f1 = alpha * current_f1 + (1 - alpha) * prev_f1_ema
                    
                    # Calculate improvements
                    relative_improvement = (prev_ema - ema_loss) / prev_ema
                    relative_f1_improvement = (ema_f1 - prev_f1_ema) / prev_f1_ema
                    
                    # Check both metrics for improvement
                    if (relative_improvement < improvement_threshold and 
                        relative_f1_improvement < improvement_threshold):
                        non_improving_counter += 1
                    else:
                        non_improving_counter = 0

                loss_variance = np.var(window_loss)
                f1_variance = np.var(window_f1)
                
                # Check both metrics for plateau
                should_break = ((non_improving_counter >= patience) or 
                              (loss_variance < variance_threshold and f1_variance < variance_threshold))
                
                print(f"EMA Loss = {ema_loss:.4f}, Loss Improvement = {relative_improvement:.4f},", end=" ")
                print(f"EMA F1 = {ema_f1:.4f}, F1 Improvement = {relative_f1_improvement:.4f},", end=" ")
                print(f"Loss Var = {loss_variance:.6f}, F1 Var = {f1_variance:.6f}, Non-improvement Count = {non_improving_counter}")
            else:
                should_break = False
                print("")

            if start_epoch % 5 == 0:
                # Memory cleanup
                del train_metric_logger, val_metric_logger, train_coco_evaluator, val_coco_evaluator
                torch.cuda.empty_cache()
                print(f"Epoch {start_epoch} GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
                print(f"Epoch {start_epoch} GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
                gc.collect()
            
            # Check if we should move to the next step
            if should_break:
                print("Plateau reached; moving to next training step.\n")
                break
                
            # Update counters
            start_epoch += 1
            step_epoch_counter += 1

        # Move to next step and reset step epoch counter
        current_step += 1
        step_epoch_counter = 0

    print('All Training Steps Complete!')
    writer.close()
    # return current_datetime, checkpoint_dir
    return current_datetime, checkpoint_dir, test_coco_ds 

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    gc.collect()
    
    # current_datetime, checkpoint_dir = main(train_coco_ds, val_coco_ds, best_trial)
    current_datetime, checkpoint_dir, test_coco_ds = main(best_trial)

# load all checkpoint metrics from current run, add to df
checkpoint_dir = Path(f'./checkpoints/{current_datetime}')

# if file in checkpoint_dir end in '_metrics.pth', load it
def load_checkpoints(checkpoint_dir):
    checkpoints = []
    for file in checkpoint_dir.glob('*_metrics.pth'):
        checkpoint = torch.load(file)
        checkpoints.append(checkpoint)
    return checkpoints

checkpoints = load_checkpoints(checkpoint_dir)
checkpoints_df = pd.DataFrame(checkpoints)

# best train epoch is epoch with max val_f1
checkpoints_df = checkpoints_df.sort_values(by='val_f1', ascending=False)
best_train_epoch = checkpoints_df.iloc[0]
epoch = best_train_epoch['epoch']
print(f"Best train epoch: {epoch}")

# initialize model with best trial config
model = get_retinanet_model(depth=best_trial.config["resnet_depth"],
                            num_classes=len(best_trial.config["class_weights"]),
                            score_thresh=best_trial.config["score_thresh"],
                            detections_per_img=200,
                            fg_iou_thresh=best_trial.config["fg_iou_thresh"],
                            bg_iou_thresh=best_trial.config["bg_iou_thresh"],
                            topk_candidates=200, 
                            alpha=best_trial.config["alpha"], 
                            gamma_loss=best_trial.config["gamma_loss"], 
                            dropout_prob=best_trial.config["dropout"],
                            beta_loss=best_trial.config["beta_loss"],
                            lambda_loss=best_trial.config["lambda_loss"],
                            class_weights=None,
                            nms_score=best_trial.config["nms_score"],
                            nms_sigma=best_trial.config["nms_sigma"])

# reference best_train_epoch['epoch'] to load model checkpoint from ./checkpoints/{current_datetime}/{current_datetime}_{epoch}.pth
checkpoint_path = f'./checkpoints/{current_datetime}/{current_datetime}_epoch{epoch}_model.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# save best model weights to .pth file
torch.save(model.state_dict(), f'./checkpoints/{current_datetime}/RetinaNet_ResNet50_FPN_DuckNet.pth')

# copy checkpoints and remove model and optimizer state dicts
checkpoints_copy = checkpoints.copy()

# save checkpoints list to text file in checkpoint_dir
checkpoint_list_path = checkpoint_dir / f"{current_datetime}_checkpoints.txt"
with open(checkpoint_list_path, 'w') as f:
    for checkpoint in checkpoints_copy:
        checkpoint.pop('model_state_dict', None)
        checkpoint.pop('optimizer_state_dict', None)
        f.write(f"{checkpoint}\n")
dataset_test = MAVdroneDataset(csv_file = 'C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/preprocessed_annotations.csv',
                                root_dir = 'C:/Users/exx/Deep Learning/UAV_Waterfowl_Detection/RetinaNet/filtered_images/', 
                                transforms = get_transform(train = False))

# subset test dataset using test_indices
dataset_test = torch.utils.data.Subset(dataset_test, test_indices)

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = 1, shuffle = False,
                                               collate_fn = utils.collate_fn, num_workers = 0,
                                               pin_memory = True)
model.to('cuda:0')
test_performance = evaluate(model, data_loader_test, test_coco_ds, device='cuda:0', train_data_loader=None, train_coco_ds=None)

# Overall metrics
mAP_50_95 = test_performance.coco_eval["bbox"].stats[0]  # AP at IoU=0.50:0.95
mAP_50 = test_performance.coco_eval["bbox"].stats[1]     # AP at IoU=0.50
mAP_75 = test_performance.coco_eval["bbox"].stats[2]     # AP at IoU=0.75

# For recall at specific IoU thresholds, we need to access the raw recall array
recall_array = test_performance.coco_eval["bbox"].eval["recall"]  # shape: [T, K, A, M]
mAR_50_95 = test_performance.coco_eval["bbox"].stats[8]  # AR at IoU=0.50:0.95
mAR_50 = np.mean(recall_array[0, :, 0, -1])  # AR at IoU=0.50 (first threshold)
mAR_75 = np.mean(recall_array[5, :, 0, -1])  # AR at IoU=0.75 (sixth threshold)

# Print overall results
print("Overall Detection Performance:")
print(f"mAP@0.50 = {mAP_50:.4f}")
print(f"mAR@0.50 = {mAR_50:.4f}")  
print(f"F1@0.50 = {calculate_f1_score(mAP_50, mAR_50):.4f}")
print()
print(f"mAP@0.75 = {mAP_75:.4f}")
print(f"mAR@0.75 = {mAR_75:.4f}")
print(f"F1@0.75 = {calculate_f1_score(mAP_75, mAR_75):.4f}")
print()
print(f"mAP@[0.50:0.95] = {mAP_50_95:.4f}")
print(f"mAR@[0.50:0.95] = {mAR_50_95:.4f}")
print(f"F1@[0.50:0.95] = {calculate_f1_score(mAP_50_95, mAR_50_95):.4f}")
print()

# Get per-class metrics
test_class_metrics = extract_per_class_metrics(test_performance, test_coco_ds)
test_class_metrics = {label_dict[k]: v for k, v in test_class_metrics.items()}

# Print per-class results
print("Per-Class Detection Performance:")
for class_name, metrics in test_class_metrics.items():
    print(f"\nClass: {class_name}")
    print(f"AP@0.50 = {metrics['precision_50']:.4f}")
    print(f"AR@0.50 = {metrics['recall_50']:.4f}")
    print(f"F1@0.50 = {calculate_f1_score(metrics['precision_50'], metrics['recall_50']):.4f}")
    print()
    print(f"AP@0.75 = {metrics['precision_75']:.4f}")
    print(f"AR@0.75 = {metrics['recall_75']:.4f}")
    print(f"F1@0.75 = {calculate_f1_score(metrics['precision_75'], metrics['recall_75']):.4f}")
    print()
    print(f"AP@[0.50:0.95] = {metrics['precision_50_95']:.4f}")
    print(f"AR@[0.50:0.95] = {metrics['recall_50_95']:.4f}") 
    print(f"F1@[0.50:0.95] = {calculate_f1_score(metrics['precision_50_95'], metrics['recall_50_95']):.4f}")