import math
import sys
import time
import torch, torchvision
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils


def train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq, accumulation_steps, val_data_loader=None, step_epoch_counter=None):
    model.train()
    train_metric_logger = utils.MetricLogger(delimiter="  ")
    train_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_header = 'Epoch: [{}] Training'.format(epoch)

    lr_scheduler = None
    # Apply warmup only at the start of each training step
    if step_epoch_counter == 0 or step_epoch_counter == 1:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_data_loader) // accumulation_steps - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # Initialize gradient scaler for mixed precision training
    scaler = torch.GradScaler(device)

    for i, (images, targets) in enumerate(train_metric_logger.log_every(train_data_loader, print_freq, train_header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the gradients before the forward pass
        if (i % accumulation_steps == 0):
            optimizer.zero_grad()

        # Use autocast for mixed precision
        with torch.autocast(device_type=device):
            train_loss_dict = model(images, targets)
            train_losses = sum(loss for loss in train_loss_dict.values()) / accumulation_steps

        # Reduce losses over all GPUs for logging purposes
        train_loss_dict_reduced = utils.reduce_dict(train_loss_dict)
        train_losses_reduced = sum(loss for loss in train_loss_dict_reduced.values())
        train_loss_value = train_losses_reduced.item()

        if not math.isfinite(train_loss_value):
            print(f"Loss is {train_loss_value}, stopping training")
            print(train_loss_dict_reduced)
            sys.exit(1)

        # Scale the loss and backward pass
        scaler.scale(train_losses).backward()

        # Update weights after accumulation steps
        if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(train_data_loader)):
            scaler.step(optimizer)
            scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

        train_metric_logger.update(loss=train_losses_reduced, **train_loss_dict_reduced)
        train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # validation loop
    if val_data_loader is not None:
        val_metric_logger = utils.MetricLogger(delimiter="  ")
        val_metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        val_header = 'Epoch: [{}] Validation'.format(epoch)

        with torch.no_grad():
            for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 2, val_header):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.autocast(device_type=device):
                    val_loss_dict = model(images, targets)

                val_loss_dict_reduced = utils.reduce_dict(val_loss_dict)
                val_losses_reduced = sum(loss for loss in val_loss_dict_reduced.values())
                val_loss_value = val_losses_reduced.item()

                if not math.isfinite(val_loss_value):
                    print(f"Loss is {val_loss_value}, stopping validation")
                    print(val_loss_dict_reduced)
                    sys.exit(1)

                val_metric_logger.update(loss=val_losses_reduced, **val_loss_dict_reduced)
                val_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if val_data_loader is not None:
        return train_metric_logger, val_metric_logger
    else:
        return train_metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, val_data_loader, val_coco_ds, device, train_data_loader=None, train_coco_ds=None):
    """
    Evaluate model on validation dataset and return validation accuracy to metric logger. If training dataset is provided,
    also return training accuracy to metric logger.
    :param model: torch.nn.Module
    :param val_data_loader: torch.utils.data.DataLoader
    :param device: torch.device
    :param train_data_loader: torch.utils.data.DataLoader (default=None)
    :param val_coco_ds: COCO dataset object for validation (precomputed)
    :param train_coco_ds: COCO dataset object for training (precomputed)
    :return: train_coco_evaluator: CocoEvaluator, 
             val_coco_evaluator: CocoEvaluator (if train_data_loader is not None else val_coco_evaluator)
    """
    
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    val_metric_logger = utils.MetricLogger(delimiter="  ")
    train_metric_logger = utils.MetricLogger(delimiter="  ") if train_data_loader is not None else None

    iou_types = _get_iou_types(model)

    train_coco_evaluator = None
    if train_data_loader is not None:
        # Use precomputed train_coco_ds if provided, otherwise create it
        if train_coco_ds is None:
            train_coco_ds = get_coco_api_from_dataset(train_data_loader.dataset)
        train_coco_evaluator = CocoEvaluator(train_coco_ds, iou_types)
        
        # Calculate training accuracy
        for images, targets in train_metric_logger.log_every(train_data_loader, len(train_data_loader) // 3, 'Training Accuracy: '):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.autocast(device_type=device):
                outputs = model(images)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            train_coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            train_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # Gather the stats from all processes
        train_metric_logger.synchronize_between_processes()
        print("Training Performance: ", train_metric_logger)
        print()
        train_coco_evaluator.synchronize_between_processes()

        # Accumulate predictions from all images
        train_coco_evaluator.accumulate()
        train_coco_evaluator.summarize()
    
    # Use precomputed val_coco_ds
    val_coco_evaluator = CocoEvaluator(val_coco_ds, iou_types)

    # Calculate validation accuracy
    for images, targets in val_metric_logger.log_every(val_data_loader, len(val_data_loader) // 2, 'Validation Accuracy: '):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.autocast(device_type=device):
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        val_coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        val_metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # Gather the stats from all processes
    val_metric_logger.synchronize_between_processes()
    print("Validation Performance: ", val_metric_logger)
    print()
    val_coco_evaluator.synchronize_between_processes()
    val_coco_evaluator.accumulate()
    val_coco_evaluator.summarize()

    torch.set_num_threads(n_threads)

    if train_data_loader is not None:
        return train_coco_evaluator, val_coco_evaluator
    else:
        return val_coco_evaluator