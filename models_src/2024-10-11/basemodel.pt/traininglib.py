import torch, torchvision
import torch.distributed as dist
import numpy as np
import scipy.optimize
from collections import defaultdict, deque
import datetime
import time

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class TrainingTask(torch.nn.Module):
    stop_requested = False
    
    def __init__(self, basemodule, epochs=10, lr=0.001, callback=None):
        super().__init__()
        self.basemodule = basemodule
        self.epochs = epochs
        self.lr = lr
        self.progress_callback = callback
        self._stopping = False
    
    def training_step(self, batch):
        raise NotImplementedError()
    
    def validation_step(self, batch):
        raise NotImplementedError()
    
    def validation_epoch_end(self, logs):
        raise NotImplementedError()

    def warmup_lr_scheduler(self, optimizer, num_batches_in_epoch):
        """Creates a scheduler that warms up linearly from 10% to 100% over initial epoch"""
        def f(x):
            epoch = x // num_batches_in_epoch
            batch_in_epoch = x % num_batches_in_epoch           
            if epoch >= 1: 
                return 1.0
            warmup_factor = 0.1 + 0.9 * (batch_in_epoch / num_batches_in_epoch)
            return warmup_factor
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
    
    def configure_optimizers(self, loader):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
        sched = self.warmup_lr_scheduler(optim, len(loader)) 
        return optim, sched
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def is_stop_requested(self):
        """Check if training should stop - use direct class reference"""
        return TrainingTask.stop_requested or self._stopping
    
    def train_one_epoch(self, loader, optimizer, scheduler=None):
        self.basemodule.train()
        
        for i, batch in enumerate(loader):
            if self.is_stop_requested():
                print("Training stopped during batch processing")
                return True 
                
            loss, logs = self.training_step(batch)
            logs['lr'] = optimizer.param_groups[0]['lr']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
                
            self.callback.on_batch_end(logs, i, len(loader))

            if self.is_stop_requested():
                print("Training stopped after batch processing")
                return True
                
        return False  

    def eval_one_epoch(self, loader):
        if self.is_stop_requested():
            print("Training stopped before validation")
            return {}
            
        all_outputs = []
        self.basemodule.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if self.is_stop_requested():
                    print("Training stopped during validation")
                    return {}
                    
                outputs, logs = self.validation_step(batch)
                all_outputs.append(outputs)

        logs = self.validation_epoch_end(all_outputs)
        
        if hasattr(self, 'callback'):
            self.callback.logs = logs

        return logs

    def fit(self, loader_train, loader_valid=None, epochs='auto'):
        self.epochs = epochs
        if epochs == 'auto':
            self.epochs = max(15, 50 // len(loader_train))
        self.epochs = int(self.epochs)
            
        if self.progress_callback is not None:
            self.callback = TrainingProgressCallback(self.progress_callback, self.epochs)
        else:
            self.callback = PrintMetricsCallback()
        
        self.train().requires_grad_(True)
        optim, sched = self.configure_optimizers(loader_train)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.cuda.empty_cache()

        TrainingTask.stop_requested = False
        self._stopping = False
        
        try:
            self.to(device)
            for e in range(self.epochs):
                if self.is_stop_requested():
                    print(f"Training stopped before epoch {e}")
                    break

                epoch_terminated = self.train_one_epoch(loader_train, optim, sched)
                if epoch_terminated:
                    print(f"Training epoch {e} terminated early")
                    break

                if self.is_stop_requested():
                    print(f"Training stopped after epoch {e} before validation")
                    break

                if loader_valid:
                    self.eval_one_epoch(loader_valid)
                
                self.callback.on_epoch_end(e)

                if self.is_stop_requested():
                    print(f"Training stopped after epoch {e} completed")
                    break
                    
        except KeyboardInterrupt:
            print('\nTraining interrupted by keyboard')
            self._stopping = True
        except Exception as e:
            print(f"Exception during training: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return e
        finally:
            self.zero_grad(set_to_none=True)
            self.eval().cpu().requires_grad_(False)
            torch.cuda.empty_cache()
            
        return TrainingTask.stop_requested or self._stopping
     
    @classmethod
    def request_stop(cls):
        """Request training stop using direct class name"""
        print("Stop training requested")
        TrainingTask.stop_requested = True

class DetectionTask(TrainingTask):
    def training_step(self, batch):
        x, y = batch
        x = [img.to(self.device) for img in x]
        y = [dict([(k, yy[k].to(self.device)) for k in ['boxes', 'labels']]) for yy in y]
        lossdict = self.basemodule(x, y)
        loss = torch.stack([*lossdict.values()]).sum()
        logs = dict([(k, v.item()) for k, v in lossdict.items()] + [('loss', loss.item())])
        return loss, logs
    
    def validation_step(self, batch):
        x, y_true = batch
        x = [img.to(self.device) for img in x]
    
        self.basemodule.eval()
        with torch.no_grad():
            y_pred = self.basemodule(x)
        
        return {'y_pred': y_pred, 'y_true': y_true}, {}
    
    def validation_epoch_end(self, logs):
        boxes_true = [b for B in [[o['boxes'].cpu() for o in O['y_true']] for O in logs] for b in B]
        boxes_pred = [b for B in [[o['boxes'].cpu() for o in O['y_pred']] for O in logs] for b in B]
        scores_pred = [s for S in [[o['scores'].cpu() for o in O['y_pred']] for O in logs] for s in S]
        labels_true = [l for L in [[o['labels'].cpu() for o in O['y_true']] for O in logs] for l in L]
        labels_pred = [l for L in [[o['labels'].cpu() for o in O['y_pred']] for O in logs] for l in L]
        
        if not scores_pred or not any(len(s) > 0 for s in scores_pred):
            return {}
        
        class_list = self.basemodule.class_list
        iou_range = np.arange(0.5, 1.0, 0.05)

        metrics = {}
        per_class_metrics = {}
        
        for class_idx, class_name in enumerate(class_list):
            class_idx_numeric = class_idx + 1  # Classes are 1-indexed
            
            class_boxes_true = []
            class_boxes_pred = []
            class_scores_pred = []
            
            for i, (b_true, l_true) in enumerate(zip(boxes_true, labels_true)):
                mask = l_true == class_idx_numeric
                class_boxes_true.append(b_true[mask])
                
            for i, (b_pred, l_pred, s_pred) in enumerate(zip(boxes_pred, labels_pred, scores_pred)):
                mask = l_pred == class_idx_numeric
                class_boxes_pred.append(b_pred[mask])
                class_scores_pred.append(s_pred[mask])
            
            if len(class_boxes_true) > 0 and any(len(b) > 0 for b in class_boxes_true):
                try:
                    class_metrics_cache = precompute_metrics(class_boxes_true, class_boxes_pred, class_scores_pred)
                    
                    class_AP_50 = cached_average_precision(class_metrics_cache, iou_threshold=0.5)
                    class_AR_50 = cached_max_recall(class_metrics_cache, iou_threshold=0.5)
                    class_F1_50 = calculate_f1_score(class_AP_50, class_AR_50)
                    
                    class_AP_75 = cached_average_precision(class_metrics_cache, iou_threshold=0.75)
                    class_AR_75 = cached_max_recall(class_metrics_cache, iou_threshold=0.75)
                    class_F1_75 = calculate_f1_score(class_AP_75, class_AR_75)
                    
                    class_AP_range = np.mean([cached_average_precision(class_metrics_cache, float(iou)) for iou in iou_range])
                    class_AR_range = np.mean([cached_max_recall(class_metrics_cache, float(iou)) for iou in iou_range])
                    class_F1_range = calculate_f1_score(class_AP_range, class_AR_range)
                    
                    per_class_metrics[class_name] = {
                        'AP@IoU=0.5': class_AP_50,
                        'AR@IoU=0.5': class_AR_50,
                        'F1@IoU=0.5': class_F1_50,
                        'AP@IoU=0.75': class_AP_75,
                        'AR@IoU=0.75': class_AR_75,
                        'F1@IoU=0.75': class_F1_75,
                        'AP@IoU=0.5:0.95': class_AP_range,
                        'AR@IoU=0.5:0.95': class_AR_range,
                        'F1@IoU=0.5:0.95': class_F1_range,
                    }
                    
                    metrics[f'{class_name}_AP@IoU=0.5'] = class_AP_50
                    metrics[f'{class_name}_AP@IoU=0.75'] = class_AP_75
                    metrics[f'{class_name}_AP@IoU=0.5:0.95'] = class_AP_range
                    
                except Exception as e:
                    print(f"Error calculating metrics for class {class_name}: {e}")
                    continue

        if per_class_metrics:
            metrics['AP@IoU=0.5'] = np.mean([m['AP@IoU=0.5'] for m in per_class_metrics.values()])
            metrics['AR@IoU=0.5'] = np.mean([m['AR@IoU=0.5'] for m in per_class_metrics.values()])
            metrics['F1@IoU=0.5'] = np.mean([m['F1@IoU=0.5'] for m in per_class_metrics.values()])
            
            metrics['AP@IoU=0.75'] = np.mean([m['AP@IoU=0.75'] for m in per_class_metrics.values()])
            metrics['AR@IoU=0.75'] = np.mean([m['AR@IoU=0.75'] for m in per_class_metrics.values()])
            metrics['F1@IoU=0.75'] = np.mean([m['F1@IoU=0.75'] for m in per_class_metrics.values()])
            
            metrics['AP@IoU=0.5:0.95'] = np.mean([m['AP@IoU=0.5:0.95'] for m in per_class_metrics.values()])
            metrics['AR@IoU=0.5:0.95'] = np.mean([m['AR@IoU=0.5:0.95'] for m in per_class_metrics.values()])
            metrics['F1@IoU=0.5:0.95'] = np.mean([m['F1@IoU=0.5:0.95'] for m in per_class_metrics.values()])

        metrics['per_class'] = per_class_metrics
        
        return metrics

def precompute_metrics(boxes_true, boxes_pred, scores_pred, N=51):
    """
    Precompute IoU matrices and error metrics at all confidence thresholds
    This avoids recalculating for different IoU thresholds
    """
    cache = {}
    confidence_thresholds = np.linspace(0, 1, N)
    
    for i, (b_true, b_pred, s_pred) in enumerate(zip(boxes_true, boxes_pred, scores_pred)):
        if len(b_true) == 0 or len(b_pred) == 0:
            continue
            
        iou_matrix = torchvision.ops.box_iou(
            torch.as_tensor(b_true), 
            torch.as_tensor(b_pred)
        ).cpu().numpy()
        
        assignment_cache = {}
        for threshold in confidence_thresholds:
            if isinstance(s_pred, torch.Tensor):
                mask = s_pred >= threshold
                has_valid_values = mask.any().item()
            else:
                mask = s_pred >= threshold
                has_valid_values = np.any(mask)
                
            if not has_valid_values:
                assignment_cache[threshold] = {
                    'matches': [],
                    'pred_count': 0,
                    'gt_count': len(b_true)
                }
                continue
                
            filtered_matrix = iou_matrix[:, mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask]
            pred_count = int(mask.sum().item() if isinstance(mask, torch.Tensor) else np.sum(mask))
            
            if filtered_matrix.size > 0:
                ixs0, ixs1 = scipy.optimize.linear_sum_assignment(filtered_matrix, maximize=True)
                matches = [(i, j, filtered_matrix[i, j]) for i, j in zip(ixs0, ixs1)]
            else:
                matches = []
                
            assignment_cache[threshold] = {
                'matches': matches,
                'pred_count': pred_count,
                'gt_count': len(b_true)
            }
            
        cache[i] = assignment_cache
        
    return cache

def cached_precision_recall(metrics_cache, iou_threshold=0.5):
    """
    Calculate precision and recall from cached matches at specified IoU threshold
    """
    N = 51  # Number of confidence thresholds
    precision = np.zeros(N)
    recall = np.zeros(N)
    
    confidence_thresholds = np.linspace(0, 1, N)
    
    TP = np.zeros(N)
    FP = np.zeros(N)
    FN = np.zeros(N)
    
    # Aggregate TP, FP, FN across all images at each confidence level
    for i, threshold_idx in enumerate(range(N)):
        threshold = confidence_thresholds[threshold_idx]
        
        for batch_id, batch_cache in metrics_cache.items():
            if threshold not in batch_cache:
                continue
            
            cache_entry = batch_cache[threshold]
            valid_matches = sum(1 for _, _, iou in cache_entry['matches'] if iou >= iou_threshold)
            
            TP[i] += valid_matches
            FP[i] += cache_entry['pred_count'] - valid_matches
            FN[i] += cache_entry['gt_count'] - valid_matches
    
    denom_p = TP + FP
    denom_r = TP + FN
    precision = np.divide(TP, denom_p, out=np.zeros_like(TP), where=denom_p > 0)
    recall = np.divide(TP, denom_r, out=np.zeros_like(TP), where=denom_r > 0)
    
    return precision, recall

def cached_average_precision(metrics_cache, iou_threshold=0.5):
    """Calculate AP using cached metrics"""
    precision, recall = cached_precision_recall(metrics_cache, iou_threshold)
    
    # AP is the area under the precision-recall curve
    ap = 0
    for r in np.arange(0, 1.01, 0.01):
        max_precision = np.max(precision[recall >= r]) if np.any(recall >= r) else 0
        ap += max_precision / 101
    
    return ap

def cached_max_recall(metrics_cache, iou_threshold=0.5):
    """Calculate max recall using cached metrics"""
    _, recall = cached_precision_recall(metrics_cache, iou_threshold)
    return np.max(recall) if len(recall) > 0 else 0

def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall values"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

class PrintMetricsCallback:
    '''Prints metrics after each training epoch in a compact table'''
    def __init__(self):
        self.epoch = 0
        self.metric_logger = MetricLogger(delimiter="  ")
        self.logs = {}  
        
    def on_epoch_end(self, epoch):
        overall_metrics = {k: v for k, v in self.logs.items() 
                        if isinstance(v, (float, int)) and k.startswith(('AP@', 'AR@', 'F1@'))}
        if overall_metrics:
            print("\nOverall metrics:")
            for metric in ['AP@IoU=0.5', 'AR@IoU=0.5', 'F1@IoU=0.5', 
                        'AP@IoU=0.75', 'AR@IoU=0.75', 'F1@IoU=0.75',
                        'AP@IoU=0.5:0.95', 'AR@IoU=0.5:0.95', 'F1@IoU=0.5:0.95']:
                if metric in overall_metrics:
                    print(f"  {metric}: {overall_metrics[metric]:.4f}")

        if 'per_class' in self.logs:
            print("\nPer-class metrics:")
            per_class = self.logs['per_class']
            for class_name, metrics in per_class.items():
                print(f"\n{class_name}:")
                print(f"  AP@IoU=0.5: {metrics['AP@IoU=0.5']:.4f}, AR@IoU=0.5: {metrics['AR@IoU=0.5']:.4f}, F1@IoU=0.5: {metrics['F1@IoU=0.5']:.4f}")
                print(f"  AP@IoU=0.75: {metrics['AP@IoU=0.75']:.4f}, AR@IoU=0.75: {metrics['AR@IoU=0.75']:.4f}, F1@IoU=0.75: {metrics['F1@IoU=0.75']:.4f}")
        print(f"\nEpoch {self.epoch} completed")
        
        self.epoch = epoch + 1
        self.metric_logger = MetricLogger(delimiter="  ")
    
    def clear_logs(self):
        """Separate method to clear logs after metrics are transferred"""
        self.logs = {}
    
    def on_batch_end(self, logs, batch_i, n_batches):
        filtered_logs = {}
        for k, v in logs.items():
            if k != 'per_class' and isinstance(v, (float, int)) and not np.isnan(v):
                filtered_logs[k] = v
            elif k == 'per_class':
                self.logs['per_class'] = v 
                
        if 'lr' in filtered_logs:
            filtered_logs['lr'] = float(filtered_logs['lr'])
            
        self.metric_logger.update(**filtered_logs)
        
        print_interval = max(1, n_batches // 10)
        if batch_i % print_interval == 0 or batch_i == 0 or batch_i == n_batches - 1:
            header = f'Epoch: [{self.epoch}]'
            print(f"{header} {batch_i+1}/{n_batches} {self.metric_logger}")

class TrainingProgressCallback:
    def __init__(self, callback_fn, epochs):
        self.n_epochs    = epochs
        self.epoch       = 0
        self.callback_fn = callback_fn
        self.logs        = {}
        self._printer    = PrintMetricsCallback()

    def on_batch_end(self, logs, batch_i, n_batches):
        self._printer.on_batch_end(logs, batch_i, n_batches)
        percent = (self.epoch * n_batches + (batch_i + 1)) / (n_batches * self.n_epochs)
        self.callback_fn(percent, logs)

    def on_epoch_end(self, epoch):
        if hasattr(self, 'logs') and self.logs:
            for k, v in self.logs.items():
                if isinstance(v, (float, int)) or k == 'per_class':
                    self._printer.logs[k] = v
        
        self._printer.on_epoch_end(epoch)
        
        self._printer.clear_logs()
        
        self.epoch = epoch + 1
        self.callback_fn(self.epoch/self.n_epochs, self.logs)