import torch, torchvision
import torchvision.transforms.v2 as T
from torch._jit_internal import is_scripting
import warnings, importlib
warnings.simplefilter('ignore') #pytorch is too noisy
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torchvision.models.detection._utils as det_utils
from torchvision.ops import sigmoid_focal_loss
import torchvision.ops.boxes as box_ops
from typing import Callable, Dict, List, Optional, NamedTuple
from pt_soft_nms import batched_soft_nms
import time, sys
from collections import Counter, defaultdict
import numpy as np

if "__torch_package__" in dir():
    import torch_package_importer # type: ignore
    import_func = torch_package_importer.import_module
else:
    import importlib
    import_func = lambda m: importlib.reload(importlib.import_module(m))

MODULES = ['datasets', 'traininglib']
[datasets, traininglib] = [import_func(m) for m in MODULES]

def _sum(x: List[torch.Tensor]) -> torch.Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res

class Prediction(NamedTuple):
    boxes:           torch.Tensor
    box_scores:      torch.Tensor
    probabilities:   torch.Tensor
    labels:          List[str]
        
    def numpy(self):
        return Prediction(*[x.cpu().numpy() if torch.is_tensor(x) else x for x in self]) # type: ignore

class DuckDetector(torch.nn.Module):
    def __init__(self, classes_of_interest):
        super().__init__()
        self.class_list         = classes_of_interest
        self.detector           = Detector()
        self._device_indicator  = torch.nn.Parameter(torch.zeros(0))  # dummy parameter

    def forward(self, x):
        results = []
        device = self._device_indicator.device
        x = x.to(device)
        
        self.eval()
        with torch.no_grad():
            detector_outputs = self.detector(x)
            if is_scripting():
                detector_outputs = detector_outputs[1] 
        
        for o in detector_outputs:
            boxes = o['boxes']
            scores = o['scores']
            labels_idx = o['labels']
            
            probabilities = torch.zeros((len(boxes), len(self.class_list)), device=device)
            string_labels = []
            
            for i, (label_idx, score) in enumerate(zip(labels_idx, scores)):
                class_idx = label_idx.item() - 1  # Convert 1-indexed to 0-indexed
                probabilities[i, class_idx] = score.item()
                string_labels.append(self.class_list[class_idx])
            
            boxes_tensor: torch.Tensor = torch.as_tensor(boxes, device=device)
            scores_tensor: torch.Tensor = torch.as_tensor(scores, device=device)
            probabilities_tensor: torch.Tensor = torch.as_tensor(probabilities, device=device)
            
            results.append(Prediction(
                boxes=boxes_tensor,
                box_scores=scores_tensor,
                probabilities=probabilities_tensor,
                labels=string_labels,
            ))
        
        return results
    
    @staticmethod
    def load_image(filename, to_tensor=False):
        image = datasets.load_image(filename)  # exif-aware
        if to_tensor:
            image = T.ToImage()(image)
        return image

    def process_image(self, image):
        if isinstance(image, str):
            image = self.load_image(image)
        width, height = image.size
        
        x = T.ToTensor()(image).unsqueeze(0)
        x = T.Resize(size=(810,), max_size=1440, interpolation=T.InterpolationMode.BILINEAR)(x)
        resized_height, resized_width = x.shape[-2:]
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])(x)
        
        self.eval()
        with torch.no_grad():
            output = self.forward(x)[0]  
        
        boxes = output.boxes.clone()
        boxes[:, 0] *= width / resized_width
        boxes[:, 1] *= height / resized_height
        boxes[:, 2] *= width / resized_width
        boxes[:, 3] *= height / resized_height
        
        boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
        scores_np = output.box_scores.cpu().numpy() if torch.is_tensor(output.box_scores) else output.box_scores
        probabilities = output.probabilities.cpu().numpy() if torch.is_tensor(output.probabilities) else output.probabilities
        labels = output.labels  
        
        return {
            'boxes': boxes_np,
            'box_scores': scores_np,
            'cls_scores': probabilities,
            'per_class_scores': [dict(zip(self.class_list, p)) for p in probabilities.tolist()],
            'labels': labels,
        }
    
    def update_class_list(self, jsonfiles_train: list[str]) -> None:
        existing_classes = self.class_list.copy()
        unique_classes = set()
        for jf in jsonfiles_train:
            labels = datasets.get_labels_from_jsonfile(jf)
            unique_classes.update(labels)
        new_classes = [cls for cls in unique_classes if cls not in existing_classes]
        
        if new_classes:
            updated_classes = existing_classes + new_classes
            print(f"Found new classes: {new_classes}")
            self.class_list = updated_classes

    def start_training_detector(self, imagefiles_train, jsonfiles_train,
        imagefiles_test=None, jsonfiles_test=None,
        classes_of_interest=None, negative_classes=[], lr=0.0005,
        epochs=10, callback=None, num_workers=0,
        use_weighted_sampling=True, validation_split=0.2): 

        if imagefiles_test is None and jsonfiles_test is None and validation_split > 0:
            print(f"No validation data provided. Auto-splitting training data ({int((1-validation_split)*100)}% train, {int(validation_split*100)}% validation)")
            
            imagefiles_train, imagefiles_test, jsonfiles_train, jsonfiles_test = self._stratified_split(
                imagefiles_train, jsonfiles_train, test_size=validation_split, random_state=666
            )
            
            print(f"Split: {len(imagefiles_train)} training files, {len(imagefiles_test)} validation files")

        original_class_list = self.class_list.copy()

        if classes_of_interest is not None:
            self.class_list = classes_of_interest.copy()

        all_json_files = jsonfiles_train + (jsonfiles_test or [])
        unique_classes = set()
        for jf in all_json_files:
            labels = datasets.get_labels_from_jsonfile(jf)
            unique_classes.update(labels)

        new_classes = [cls for cls in unique_classes 
                      if cls not in self.class_list and cls not in negative_classes]
        new_classes.sort()

        if new_classes:
            print(f"Found new classes: {new_classes}")
            self.class_list.extend(new_classes)

        if original_class_list != self.class_list:
            print(f"Class list changed. Reinitializing detector with {len(self.class_list)} classes.")
            num_classes = int(len(self.class_list) + 1)  # +1 for background class
            
            old_detector = self.detector
            self.detector = Detector(num_classes=num_classes, pretrained_detector=old_detector)

        object.__setattr__(self.detector, 'class_list', self.class_list.copy())
        print(f"Final class_list for training: {self.class_list}")

        filtered_train_images = []
        filtered_train_jsons = []
        
        for img_file, json_file in zip(imagefiles_train, jsonfiles_train):
            labels = datasets.get_labels_from_jsonfile(json_file)
            if all(label in self.class_list for label in labels):
                filtered_train_images.append(img_file)
                filtered_train_jsons.append(json_file)

        if len(filtered_train_images) < len(imagefiles_train):
            print(f"Filtered training set: {len(imagefiles_train)} → {len(filtered_train_images)} files")
            imagefiles_train = filtered_train_images
            jsonfiles_train = filtered_train_jsons

        if imagefiles_test is not None and jsonfiles_test is not None:
            filtered_image_files = []
            filtered_json_files = []
            
            for img_file, json_file in zip(imagefiles_test, jsonfiles_test):
                labels = datasets.get_labels_from_jsonfile(json_file)
                if all(label in self.class_list for label in labels):
                    filtered_image_files.append(img_file)
                    filtered_json_files.append(json_file)
            
            if len(filtered_image_files) < len(imagefiles_test):
                print(f"Filtered validation set: {len(imagefiles_test)} → {len(filtered_image_files)} files")
                imagefiles_test = filtered_image_files
                jsonfiles_test = filtered_json_files

        self._print_class_distribution(imagefiles_train, jsonfiles_train, imagefiles_test, jsonfiles_test)

        ds_type = datasets.DetectionDataset
        ds_train = ds_type(imagefiles_train, jsonfiles_train,
                        augment=True,
                        negative_classes=negative_classes,
                        class_list=self.class_list)

        if use_weighted_sampling:
            _, sample_weights = self._calculate_class_weights(
                jsonfiles_train, negative_classes, original_class_list
            )
            
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            dl_train = datasets.create_dataloader(
                ds_train, batch_size=2, sampler=sampler, num_workers=num_workers
            )
        else:
            dl_train = datasets.create_dataloader(ds_train, batch_size=2, shuffle=True, num_workers=num_workers)

        dl_test = None
        if imagefiles_test is not None:
            ds_test = ds_type(imagefiles_test, jsonfiles_test,
                            augment=False,
                            negative_classes=negative_classes,
                            class_list=self.class_list)
            dl_test = datasets.create_dataloader(ds_test, batch_size=1, shuffle=False, num_workers=num_workers)

        task = traininglib.DetectionTask(self.detector, callback=callback, lr=lr)
        ret = task.fit(dl_train, dl_test, epochs=epochs)
        return (not task.stop_requested and not ret)
    
    def _print_class_distribution(self, train_images, train_jsons, test_images, test_jsons):
        """Print class distribution after filtering"""
        image_class_distribution = {}

        for img_file, json_file in zip(train_images, train_jsons):
            labels = datasets.get_labels_from_jsonfile(json_file)
            image_class_distribution[img_file] = labels

        for img_file, json_file in zip(test_images, test_jsons):
            labels = datasets.get_labels_from_jsonfile(json_file)
            image_class_distribution[img_file] = labels
        
        train_class_counts = Counter()
        test_class_counts = Counter()
        
        for img in train_images:
            train_class_counts.update(image_class_distribution[img])
        for img in test_images:
            test_class_counts.update(image_class_distribution[img])
        
        table_str = "\nPost-filtering class distribution:\n"
        table_str += f"{'Class':<10} {'Train':<8} {'Test':<8} {'Total':<8} {'Train %':<8}\n"
        table_str += "-" * 50 + "\n"
        
        all_classes = sorted(set(train_class_counts.keys()) | set(test_class_counts.keys()))
        for cls in all_classes:
            train_count = train_class_counts[cls]
            test_count = test_class_counts[cls]
            total = train_count + test_count
            train_pct = train_count / total * 100 if total > 0 else 0
            table_str += f"{cls:<10} {train_count:<8} {test_count:<8} {total:<8} {train_pct:.1f}%\n"

        print(table_str, flush=True)

    def _stratified_split(self, imagefiles, jsonfiles, test_size=0.2, random_state=666):
        """Stratified split that ensures all classes appear in both train and test sets"""
        from sklearn.model_selection import train_test_split

        image_class_distribution = {}
        for img_file, json_file in zip(imagefiles, jsonfiles):
            labels = datasets.get_labels_from_jsonfile(json_file)
            image_class_distribution[img_file] = labels if labels else ['background']

        class_to_images = defaultdict(list)
        for img, classes in image_class_distribution.items():
            for cls in classes:
                class_to_images[cls].append(img)

        train_images = set()
        test_images = set()

        for cls, images in class_to_images.items():
            if len(images) == 1:
                train_images.add(images[0])
                test_images.add(images[0])
            else:
                np.random.seed(random_state + hash(cls) % 10000)
                shuffled = np.random.permutation(images).tolist()
                train_images.add(shuffled[0])
                test_images.add(shuffled[1 % len(shuffled)])

        remaining_images = [img for img in imagefiles 
                        if img not in train_images or img not in test_images]
        
        if remaining_images:
            remaining_class_distribution = {img: image_class_distribution[img] 
                                        for img in remaining_images}
            
            class_counts = Counter()
            for labels in remaining_class_distribution.values():
                class_counts.update(labels)

            top_classes = [cls for cls, _ in class_counts.most_common(3)]
            
            stratification_features = []
            for img in remaining_images:
                img_labels = set(remaining_class_distribution[img])
                feature = ''.join(['1' if cls in img_labels else '0' for cls in top_classes])
                stratification_features.append(feature)

            target_test_size = int(len(imagefiles) * test_size)
            current_test_size = len(test_images)
            remaining_test_size = max(0, target_test_size - current_test_size)
            adjusted_test_size = remaining_test_size / len(remaining_images)
            
            train_remaining, test_remaining = train_test_split(
                remaining_images,
                test_size=adjusted_test_size,
                stratify=stratification_features,
                random_state=random_state
            )
            
            train_images.update(train_remaining)
            test_images.update(test_remaining)

        train_images = list(train_images - test_images)
        test_images = list(test_images)

        target_test_count = int(len(imagefiles) * test_size)
        if len(test_images) < target_test_count:
            move_count = target_test_count - len(test_images)
            np.random.seed(random_state)
            to_move = np.random.choice(train_images, size=move_count, replace=False)
            for img in to_move:
                train_images.remove(img)
                test_images.append(img)
        elif len(test_images) > target_test_count:
            move_count = len(test_images) - target_test_count
            np.random.seed(random_state)
            to_move = np.random.choice(test_images, size=move_count, replace=False)
            for img in to_move:
                test_images.remove(img)
                train_images.append(img)

        train_json = [jsonfiles[imagefiles.index(img)] for img in train_images]
        test_json = [jsonfiles[imagefiles.index(img)] for img in test_images]
        
        return train_images, test_images, train_json, test_json
        
    def _calculate_class_weights(self, jsonfiles_train: list[str], known_negative_classes: list[str], original_classes: list[str]):
        """Calculate class weights with Hen fixed at 0.3, other originals 0.5-1.0, new classes 1.5-2.0"""
        
        class_counts = Counter()
        for jf in jsonfiles_train:
            labels = datasets.get_labels_from_jsonfile(jf)
            class_counts.update(labels)
        
        print("Calculating class weights for balanced sampling...")
        
        beta = 0.99 
        class_weights = {}
        
        # Calculate effective number weights for all classes
        for class_name in self.class_list:
            count = class_counts.get(class_name, 0)
            if count > 0:
                effective_num = (1.0 - beta**count) / (1.0 - beta)
                class_weights[class_name] = 1.0 / effective_num
            else:
                class_weights[class_name] = 1.0
        
        # Handle negative classes
        for cls in known_negative_classes:
            if cls in class_weights:
                class_weights[cls] = 0.1
        
        # Separate original and new classes
        original_weights = {cls: class_weights[cls] for cls in original_classes 
                        if cls in class_weights and cls not in known_negative_classes}
        
        new_classes = [cls for cls in self.class_list 
                    if cls not in original_classes and cls not in known_negative_classes]

        # Calculate weights for original classes: range 0.5-1.0, Hen fixed at 0.3
        if original_weights:
            if 'Hen' in original_weights:
                class_weights['Hen'] = 0.3
                
            non_hen_original = {cls: weight for cls, weight in original_weights.items() if cls != 'Hen'}
            
            if non_hen_original:
                min_weight = min(non_hen_original.values()) 
                max_weight = max(non_hen_original.values())  
                
                for cls, weight in non_hen_original.items():
                    if max_weight > min_weight:
                        normalized = (weight - min_weight) / (max_weight - min_weight)
                        class_weights[cls] = 0.5 + (normalized * 0.5)  
                    else:
                        class_weights[cls] = 0.75 
                         
        # Calculate weights for new classes: range 1.5-2.0
        if new_classes:
            new_counts = {cls: class_counts.get(cls, 0) for cls in new_classes}
            
            if len(new_counts) > 1:
                min_count = min(new_counts.values())  
                max_count = max(new_counts.values())  

                for cls, count in new_counts.items():
                    if max_count > min_count:
                        normalized = (count - min_count) / (max_count - min_count)
                        inverted = 1.0 - normalized 
                        class_weights[cls] = 1.5 + (inverted * 0.5)
                    else:
                        class_weights[cls] = 1.75 
            else:
                class_weights[new_classes[0]] = 1.75
        
        class_weights['background'] = 0.1
        
        # Calculate sample weights for WeightedRandomSampler
        sample_weights = []
        for jf in jsonfiles_train:
            labels = datasets.get_labels_from_jsonfile(jf)
            if len(labels) == 0:
                sample_weights.append(class_weights['background'])
            else:
                label_counts = Counter(labels)
                total_instances = sum(label_counts.values())
                weighted_sum = sum(class_weights.get(label, 1.0) * count for label, count in label_counts.items())
                sample_weights.append(weighted_sum / total_instances)
        
        # Print weights
        print(f"  Background: {class_weights['background']:.3f}")
        
        printed_classes = set()
        for cls in self.class_list:
            if cls in class_weights and cls not in printed_classes:
                count = class_counts.get(cls, 0)
                print(f"  {cls}: class count={count}, class weight={class_weights[cls]:.3f}")
                printed_classes.add(cls)
        
        return class_weights, sample_weights
    
    def stop_training(self):
        traininglib.TrainingTask.request_stop()
    
    def save(self, destination):
        if isinstance(destination, str):
            destination = time.strftime(destination)
            if not destination.endswith('.pt.zip'):
                destination += '.pt.zip'
        
        try:
            import torch_package_importer as imp # type: ignore
            importer = (imp, torch.package.sys_importer) # type: ignore
        except ImportError:
            importer = (torch.package.sys_importer,) # type: ignore
        
        with torch.package.PackageExporter(destination, importer) as pe: # type: ignore
            current_module = __name__.split('.')[-1]
            interns = [current_module] + MODULES
            
            pe.extern([
                'torchvision.**',
                'torchvision.ops.**',
                'torchvision.models.**',
                'pt_soft_nms',
                'pt_soft_nms.**',
            ])
            pe.intern(interns)
            pe.extern('**', exclude=interns)
            
            for inmod in interns:
                if inmod in sys.modules:
                    pe.save_source_file(inmod, sys.modules[inmod].__file__, dependencies=True)
                else:
                    pe.save_source_string(inmod, importer[0].get_source(inmod))
            
            pe.save_pickle('model', 'model.pkl', self)
            pe.save_text('model', 'class_list.txt', '\n'.join(self.class_list))
        
        return destination

class CustomRetinaNetClassificationHead(RetinaNetClassificationHead):
    def __init__(self, in_channels, num_anchors, num_classes, alpha=0.25, gamma_loss=2.0, prior_probability=0.01, 
                norm_layer: Optional[Callable[..., torch.nn.Module]] = None, dropout_prob=0.25, class_weights=None, label_smoothing=0.1):
        super().__init__(in_channels, num_anchors, num_classes, prior_probability, norm_layer)
        self.alpha = alpha
        self.gamma_loss = gamma_loss
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        cls_logits = head_outputs["cls_logits"]

        for i, (targets_per_image, cls_logits_per_image, matched_idxs_per_image) in enumerate(zip(targets, cls_logits, matched_idxs)):
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target += self.label_smoothing / (self.num_classes - 1) # smoothing for negative classes
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][matched_idxs_per_image[foreground_idxs_per_image]],
            ] = 1.0 - self.label_smoothing # smoothing for positive classes

            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS
            if self.class_weights is not None:
                valid_labels = targets_per_image["labels"][matched_idxs_per_image[valid_idxs_per_image]]
                weights = self.class_weights.to(valid_labels.device)[valid_labels]
            else:
                weights = torch.ones(cls_logits_per_image[valid_idxs_per_image].shape[0], 
                                   dtype=torch.float32, device=cls_logits_per_image.device)

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
            cls_logits = self.dropout(cls_logits) 
            cls_logits = self.cls_logits(cls_logits)

            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)

class CustomRetinaNetRegressionHead(RetinaNetRegressionHead):
    def __init__(self, in_channels, num_anchors, norm_layer: Optional[Callable[..., torch.nn.Module]] = None, 
                 _loss_type="smooth_l1", beta_loss=0.5, lambda_loss=1.0, dropout_prob=0.25):
        super().__init__(in_channels, num_anchors, norm_layer)
        self._loss_type = _loss_type
        self.beta_loss = beta_loss # beta < 1 helps counter early plateauing
        self.lambda_loss = lambda_loss # lambda > 1 places more emphasis on localization loss
        self.dropout = torch.nn.Dropout(p=dropout_prob)
    
    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        losses = []
        bbox_regression = head_outputs["bbox_regression"]

        for targets_per_image, bbox_regression_per_image, anchors_per_image, matched_idxs_per_image in zip(
            targets, bbox_regression, anchors, matched_idxs
        ):
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            matched_gt_boxes_per_image = targets_per_image["boxes"][matched_idxs_per_image[foreground_idxs_per_image]]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

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

            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)

class CustomRetinaNet(RetinaNet):
    def __init__(
        self,
        backbone,
        num_classes,
        min_size,
        max_size,
        image_mean,
        image_std,
        score_thresh,
        detections_per_img,
        fg_iou_thresh,
        bg_iou_thresh,
        topk_candidates,
        nms_score,
        nms_sigma,
    ):
        super().__init__(
            backbone,
            num_classes=num_classes,
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std,
            score_thresh=score_thresh,
            detections_per_img=detections_per_img,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            topk_candidates=topk_candidates,
        )
        self.nms_score = nms_score
        self.nms_sigma = nms_sigma

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
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

            for (
                box_regression_per_level,
                logits_per_level,
                anchors_per_level,
            ) in zip(box_regression_per_image, logits_per_image, anchors_per_image):
                # logits_per_level shape: (num_anchors, num_classes)
                scores_per_level = torch.sigmoid(logits_per_level)  # (N, num_classes)
                scores_per_level, labels_per_level = scores_per_level.max(dim=-1)  # (N,), (N,)

                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                labels_per_level = labels_per_level[keep_idxs]

                if scores_per_level.numel() == 0:
                    continue

                num_topk = det_utils._topk_min(
                    torch.nonzero(keep_idxs).flatten(), self.topk_candidates, 0
                )
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                labels_per_level = labels_per_level[idxs]

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[keep_idxs][idxs],
                    anchors_per_level[keep_idxs][idxs],
                )
                boxes_per_level = box_ops.clip_boxes_to_image(
                    boxes_per_level, image_shape
                )

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            if not image_boxes:
                detections.append(
                    {
                        "boxes": torch.empty((0, 4)),
                        "scores": torch.empty((0,)),
                        "labels": torch.empty((0,), dtype=torch.int64),
                    }
                )
                continue

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # Apply soft-NMS
            keep = batched_soft_nms(
                image_boxes,
                image_scores,
                image_labels,
                sigma=self.nms_sigma,
                score_threshold=self.nms_score,
            )
            keep_boxes = image_boxes[keep]
            keep_scores = image_scores[keep]
            keep_labels = image_labels[keep]
            score_order = keep_scores.argsort(descending=True)
            keep_boxes = keep_boxes[score_order]
            keep_scores = keep_scores[score_order]
            keep_labels = keep_labels[score_order]

            final_keep = []
            remaining_boxes_mask = torch.ones(
                len(keep_boxes), dtype=torch.bool, device=keep_boxes.device
            )
            for i in range(len(keep_boxes)):
                if remaining_boxes_mask[i]:
                    final_keep.append(i)
                    current_box = keep_boxes[i : i + 1]
                    ious = box_ops.box_iou(current_box, keep_boxes[remaining_boxes_mask])[0]
                    high_overlap_indices = torch.where(ious > 0.8)[0]
                    absolute_indices = torch.where(remaining_boxes_mask)[0][
                        high_overlap_indices
                    ]
                    if len(absolute_indices) > 0 and absolute_indices[0] == i:
                        absolute_indices = absolute_indices[1:]
                    remaining_boxes_mask[absolute_indices] = False
            final_keep = torch.tensor(
                final_keep, device=keep_boxes.device, dtype=torch.long
            )
            if len(final_keep) > self.detections_per_img:
                final_keep = final_keep[: self.detections_per_img]
            detections.append(
                {
                    "boxes": keep_boxes[final_keep],
                    "scores": keep_scores[final_keep],
                    "labels": keep_labels[final_keep],
                }
            )
        return detections

class Detector(torch.nn.Module):
    def __init__(self, num_classes: int = 10, pretrained_detector=None):
        super().__init__()
        self.basemodel = self._get_custom_retinanet_model(num_classes)
        
        if pretrained_detector is not None:
            self._load_compatible_weights(pretrained_detector)
            
        self._device_indicator = torch.nn.Parameter(torch.zeros(0))  # dummy parameter
    
    def _load_compatible_weights(self, pretrained_detector):
        """Load weights from pretrained detector, skipping incompatible layers (classification/regression heads)"""
        pretrained_state = pretrained_detector.basemodel.state_dict()
        current_state = self.basemodel.state_dict()
        
        compatible_weights = {}
        
        for key, value in pretrained_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_weights[key] = value
        
        self.basemodel.load_state_dict(compatible_weights, strict=False)
    
    def _get_custom_retinanet_model(self, num_classes: int):
        trainable_backbone_layers = 2 
        backbone = resnet_fpn_backbone(
            'resnet50',
            weights=torchvision.models.ResNet50_Weights.DEFAULT,
            trainable_layers=trainable_backbone_layers
        )
        model = CustomRetinaNet(
            backbone,
            num_classes=num_classes,
            min_size=810,
            max_size=1440,
            image_mean=[0, 0, 0],
            image_std=[1, 1, 1],
            score_thresh=0.5,
            detections_per_img=200,
            fg_iou_thresh=0.6,
            bg_iou_thresh=0.5,
            topk_candidates=200,
            nms_score=0.6,
            nms_sigma=0.5
        )
        in_channels = model.head.classification_head.cls_logits.in_channels
        num_anchors = model.head.classification_head.num_anchors

        model.head.classification_head = CustomRetinaNetClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            alpha=0.5,
            gamma_loss=2.0,
            prior_probability=0.01,
            dropout_prob=0.25
        )
        model.head.regression_head = CustomRetinaNetRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            _loss_type="smooth_l1",
            beta_loss=0.6,
            lambda_loss=0.75,
            dropout_prob=0.25
        )
        model.anchor_generator = AnchorGenerator( # sizes and ratios calculated from dataset resized to 810x1440
            sizes=((24, 32, 40), (48, 64, 80), (96, 128, 160), (192, 256, 320), (472, 536, 600)),
            aspect_ratios=((0.75, 1.15, 1.8),) * 5
        )
        return model
    
    def forward(self, x, targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        out = self.basemodel(x, targets)
        return out