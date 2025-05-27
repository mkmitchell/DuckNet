import torch, torchvision
import torchvision.transforms.v2 as T
from torch._jit_internal import is_scripting
import warnings, importlib
from collections import Counter
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
            for i, (label, score) in enumerate(zip(labels_idx, scores)):
                class_idx = label.item() - 1  
                if 0 <= class_idx < len(self.class_list):
                    probabilities[i, class_idx] = score
            
            string_labels = [self.class_list[idx.item()-1] for idx in labels_idx]
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
    
    def update_class_list_for_training(self, jsonfiles_train: list[str]) -> None:
        existing_classes = self.class_list.copy()
        unique_classes = set()
        for jf in jsonfiles_train:
            labels = datasets.get_labels_from_jsonfile(jf)
            unique_classes.update(labels)
        new_classes = [cls for cls in unique_classes if cls not in existing_classes]
        
        if new_classes:
            updated_classes = existing_classes + new_classes
            print(f"Found new classes: {new_classes}")
            print(f"Updating class_list for training: {updated_classes}")
            self.class_list = updated_classes
        else:
            print(f"No new classes found. Using existing class list: {existing_classes}")

    def start_training_detector(self, imagefiles_train, jsonfiles_train,
                            imagefiles_test=None, jsonfiles_test=None,
                            negative_classes=[], lr=0.001,
                            epochs=10, callback=None, num_workers=0,
                            use_weighted_sampling=True, validation_split=0.2): 
    
        # Auto-split training data if no validation data provided
        if imagefiles_test is None and jsonfiles_test is None and validation_split > 0:
            from sklearn.model_selection import train_test_split
            print(f"No validation data provided. Auto-splitting training data ({int((1-validation_split)*100)}% train, {int(validation_split*100)}% validation)")
            
            imagefiles_train, imagefiles_test, jsonfiles_train, jsonfiles_test = train_test_split(
                imagefiles_train, jsonfiles_train,
                test_size=validation_split,
                random_state=42
            )
            
            print(f"Split: {len(imagefiles_train)} training files, {len(imagefiles_test)} validation files")
        
        original_class_list = self.class_list.copy()
        self.update_class_list_for_training(jsonfiles_train)

        if original_class_list != self.class_list:
            print(f"New classes found. Reinitializing detector with {len(self.class_list)} classes.")
            num_classes = int(len(self.class_list) + 1)
            self.detector = Detector(num_classes=num_classes)
        
        self.detector.class_list = self.class_list
        
        ds_type = datasets.DetectionDataset
        ds_train = ds_type(imagefiles_train, jsonfiles_train,
                        augment=True,
                        negative_classes=negative_classes,
                        class_list=self.class_list)
        
        if use_weighted_sampling:
            all_labels = []
            class_indices = {class_name: idx+1 for idx, class_name in enumerate(self.class_list)}
            print("Calculating class weights for balanced sampling...")
            for json_file in jsonfiles_train:
                labels = datasets.get_labels_from_jsonfile(json_file)
                label_indices = [class_indices.get(label, 0) for label in labels if label in class_indices]
                all_labels.extend(label_indices)
            
            if all_labels:
                class_counts = Counter(all_labels)       
                raw_inv = {}
                for cls_idx, cnt in class_counts.items():
                    if cls_idx == 0: 
                        continue
                    if self.class_list[cls_idx-1] == "Hen":
                        continue
                    raw_inv[cls_idx] = max(class_counts.values()) / cnt

                min_inv, max_inv = min(raw_inv.values()), max(raw_inv.values())
                scaled = {ci: 1.0 + (inv - min_inv) / (max_inv - min_inv) for ci, inv in raw_inv.items()}

                # set background to 0.1
                class_weights = {0: 0.1}
                class_weights.update(scaled)

                hen_idx = class_indices.get("Hen")
                if hen_idx is not None:
                    other_max = max(cnt for ci, cnt in class_counts.items()
                                    if ci != hen_idx and ci != 0)
                    hen_count = class_counts[hen_idx]
                    class_weights[hen_idx] = other_max / hen_count

                sample_weights = []
                for json_file in jsonfiles_train:
                    labels = datasets.get_labels_from_jsonfile(json_file)
                    idxs = [class_indices.get(l, 0) for l in labels if l in class_indices]
                    if idxs:
                        w = sum(class_weights.get(i, 1.0) for i in idxs) / len(idxs)
                    else:
                        w = 1.0
                    sample_weights.append(w)

                print(f"  Background: {class_weights[0]:.3f}")
                for cls_name, idx in class_indices.items():
                    count = class_counts.get(idx, 0)
                    print(f"  {cls_name}: count={count}, weight={class_weights.get(idx, 1.0):.3f}")

                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )

                dl_train = datasets.create_dataloader(
                    ds_train, batch_size=2, sampler=sampler, num_workers=num_workers
                )
            else:
                print("Warning: No labels found for weighted sampling, using random sampling instead.")
                dl_train = datasets.create_dataloader(ds_train, batch_size=2, shuffle=True, num_workers=num_workers)
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
    
    def stop_training(self):
        traininglib.TrainingTask.request_stop()
    
    def save(self, destination):
        if isinstance(destination, str):
            destination = time.strftime(destination)
            if not destination.endswith('.pt.zip'):
                destination += '.pt.zip'
        
        try:
            import torch_package_importer as imp
            importer = (imp, torch.package.sys_importer)
        except ImportError:
            importer = (torch.package.sys_importer,)
        
        with torch.package.PackageExporter(destination, importer) as pe:
            current_module = __name__.split('.')[-1]
            interns = [current_module] + MODULES
            
            # Extern problematic modules/operations first
            pe.extern([
                'torchvision.**',
                'torchvision.ops.**',
                'torchvision.models.**',
                'pt_soft_nms',
                'pt_soft_nms.**',
            ])
            
            # Intern your modules
            pe.intern(interns)
            
            # Extern everything else
            pe.extern('**', exclude=interns)
            
            # Force inclusion of internal modules
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

            # Remove overlapping boxes regardless of class
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
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.basemodel = self._get_custom_retinanet_model(num_classes)
        model_path = "S:/Savanna Institute/Deep Learning/DuckNet/RetinaNet/RetinaNet_ResNet50_FPN_DuckNet.pth"
        self.basemodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self._device_indicator = torch.nn.Parameter(torch.zeros(0))  # dummy parameter
    
    def _get_custom_retinanet_model(self, num_classes: int):
        trainable_backbone_layers = 0  # only update classification and regression heads
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
            gamma_loss=3.0,
            prior_probability=0.01,
            dropout_prob=0.25
        )
        model.head.regression_head = CustomRetinaNetRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            _loss_type="smooth_l1",
            beta_loss=0.6,
            lambda_loss=1.2,
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