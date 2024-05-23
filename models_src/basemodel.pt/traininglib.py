import torch, torchvision
import numpy as np
import scipy.optimize

class TrainingTask(torch.nn.Module):
    def __init__(self, basemodule, epochs=10, lr=0.05, callback=None):
        super().__init__()
        self.basemodule        = basemodule
        self.epochs            = epochs
        self.lr                = lr
        self.progress_callback = callback
    
    def training_step(self, batch):
        raise NotImplementedError()
    def validation_step(self, batch):
        raise NotImplementedError()
    def validation_epoch_end(self, logs):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size = 3, gamma=0.1)
        return optim, sched
    
    @property
    def device(self):
        return next(self.parameters()).device
    

    # def train_one_epoch(self, data_loader, optimizer, device, epoch, scheduler=None):
    #     self.train()
    #     print_freq = 10
    #     metric_logger = utils.MetricLogger(delimiter="  ")
    #     metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    #     header = 'Epoch: [{}]'.format(epoch)

    #     lr_scheduler = scheduler
    #     if epoch == 0:
    #         warmup_factor = 1. / 1000
    #         warmup_iters = min(1000, len(data_loader) - 1)

    #         lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    #     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    #         images = list(image.to(device) for image in images)
    #         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    #         loss_dict = self(images, targets)

    #         losses = sum(loss for loss in loss_dict.values())

    #         # reduce losses over all GPUs for logging purposes
    #         loss_dict_reduced = utils.reduce_dict(loss_dict)
    #         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

    #         loss_value = losses_reduced.item()

    #         if not math.isfinite(loss_value):
    #             print("Loss is {}, stopping training".format(loss_value))
    #             print(loss_dict_reduced)
    #             sys.exit(1)

    #         optimizer.zero_grad()
    #         losses.backward()
    #         optimizer.step()

    #         if lr_scheduler is not None:
    #             lr_scheduler.step()

    #         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
    #         metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    #     return metric_logger
            
    def train_one_epoch(self, loader, optimizer, scheduler=None):
        for i,batch in enumerate(loader):
            if self.__class__.stop_requested:
                break
            loss,logs  = self.training_step(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.callback.on_batch_end(logs, i, len(loader))
        if scheduler:
            scheduler.step()

    # @torch.no_grad()
    # def eval_one_epoch(self, data_loader, device):
    #     n_threads = torch.get_num_threads()
    #     torch.set_num_threads(1)
    #     cpu_device = torch.device("cpu")
    #     self.eval()
    #     metric_logger = utils.MetricLogger(delimiter="  ")
    #     header = 'Test:'

    #     coco = utils.get_coco_api_from_dataset(data_loader.dataset)
    #     iou_types = _get_iou_types(self)
    #     coco_evaluator = utils.CocoEvaluator(coco, iou_types)

    #     for images, targets in metric_logger.log_every(data_loader, 100, header):
    #         images = list(img.to(device) for img in images)

    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         model_time = time.time()
    #         outputs = self(images)

    #         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    #         model_time = time.time() - model_time

    #         res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
    #         evaluator_time = time.time()
    #         coco_evaluator.update(res)
    #         evaluator_time = time.time() - evaluator_time
    #         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    #     # gather the stats from all processes
    #     metric_logger.synchronize_between_processes()
    #     print("Averaged stats:", metric_logger)
    #     coco_evaluator.synchronize_between_processes()

    #     # accumulate predictions from all images
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    #     torch.set_num_threads(n_threads)
    #     return coco_evaluator

    def eval_one_epoch(self, loader):
            all_outputs = []
            for i,batch in enumerate(loader):
                outputs, logs  = self.validation_step(batch)
                self.callback.on_batch_end(logs, i, len(loader))
                all_outputs   += [outputs]
            logs = self.validation_epoch_end(all_outputs)
            self.callback.on_batch_end(logs, i, len(loader))


    def fit(self, loader_train, loader_valid=None, epochs='auto'):
        self.epochs = epochs
        if epochs == 'auto':
            self.epochs = max(15, 50 // len(loader_train))
            
        if self.progress_callback is not None:
            self.callback = TrainingProgressCallback(self.progress_callback, self.epochs)
        else:
            self.callback = PrintMetricsCallback()

        self.train().requires_grad_(True)
        optim, sched  = self.configure_optimizers()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.cuda.empty_cache()
        try:
            self.to(device)
            self.__class__.stop_requested = False
            for e in range(self.epochs):
                if self.__class__.stop_requested:
                    break
                self.train().requires_grad_(True)
                # self.train_one_epoch(loader_train, optim, device, epoch, sched)
                self.train_one_epoch(loader_train, optim, sched)
                
                self.eval().requires_grad_(False)
                if loader_valid:
                    # self.eval_one_epoch(loader_valid, device)
                    self.eval_one_epoch(loader_valid)

                self.callback.on_epoch_end(e)
        except KeyboardInterrupt:
            print('\nInterrupted')
        except Exception as e:
            #prevent the exception getting to ipython (memory leak)
            import traceback
            traceback.print_exc()
            return e
        finally:
            self.zero_grad(set_to_none=True)
            self.eval().cpu().requires_grad_(False)
            torch.cuda.empty_cache()
        
    #XXX: class method to avoid boiler code
    @classmethod
    def request_stop(cls):
        cls.stop_requested = True

def _get_iou_types(self):
        model_without_ddp = self
        if isinstance(self, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.module
        iou_types = ["bbox"]
        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")
        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")
        return iou_types

class DetectionTask(TrainingTask):
    def training_step(self, batch):
        x,y         = batch
        x           = x.to(self.device)
        y           = [dict([(k,yy[k].to(self.device)) for k in ['boxes', 'labels']])  for yy in y]
        lossdict    = self.basemodule(x,y)
        loss        = torch.stack( [*lossdict.values()] ).sum()
        logs        = dict([(k,v.item())  for k,v in lossdict.items()] + [('loss', loss.item())])
        return loss, logs
    
    def validation_step(self, batch):
        x,y_true    = batch
        x           = x.to(self.device)
        y_pred      = self.basemodule(x)
        return {'y_pred':y_pred, 'y_true':y_true}, {}
    
    def validation_epoch_end(self, outputs):
        boxes_true  = [b for B in [ [o['boxes'].cpu()  for o in O['y_true']] for O in outputs] for b in B]
        boxes_pred  = [b for B in [ [o['boxes'].cpu()  for o in O['y_pred']] for O in outputs] for b in B]
        scores_pred = [s for S in [ [o['scores'].cpu() for o in O['y_pred']] for O in outputs] for s in S]
        return {
            'Precision@98Recall' : precision_at_recall(boxes_true, boxes_pred, scores_pred, target_recall=0.98),
            'Precision@95Recall' : precision_at_recall(boxes_true, boxes_pred, scores_pred, target_recall=0.95),
            'Precision@90Recall' : precision_at_recall(boxes_true, boxes_pred, scores_pred, target_recall=0.90),
        }


def error_metrics_at_score(boxes, predicted_boxes, scores, score_threshold, iou_threshold=0.5):
    '''Calculates true positives, false positives and false negatives.
       Disregards predictions whose confidence level (scores) is lower than threshold.
       A prediction counts as a true positive if there is a matching label with at least the specified iou value'''
    #filter all predictions below the threshold, they will not be counted as predictions any more
    predicted_boxes = predicted_boxes[scores >= score_threshold]
    ious      = torchvision.ops.box_iou( torch.as_tensor(boxes), torch.as_tensor(predicted_boxes) ).cpu().numpy()
    ixs0,ixs1 = scipy.optimize.linear_sum_assignment(ious, maximize=True)
    TP        = len(ixs0)
    FP        = len(predicted_boxes) - TP
    FN        = len(boxes)           - TP
    return TP,FP,FN

def precision_recall(boxes, predicted_boxes, scores, iou_threshold=0.5, N=51):
    '''Calculates precision and recall values at different confidence thresholds.
       labels: list of batched boxes
       predictions: list of batched boxes
       scores: list of batched scalars'''
    EM = np.zeros((N,3))
    for b,p,s in zip(boxes, predicted_boxes, scores):
        EM += np.array([error_metrics_at_score(b, p, s, th, iou_threshold) for th in np.linspace(0,1,N)])
    
    TP,FP,FN  = EM.T
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    return precision, recall

def precision_at_recall(boxes, predicted_boxes, scores, target_recall=0.95, iou_threshold=0.25):
    precision, recall = precision_recall(boxes, predicted_boxes, scores, iou_threshold)
    #XXX: assuming recall is always monotonically decreasing (is it?)
    precision_at      = np.interp(target_recall, recall[::-1], precision[::-1], right=np.nan)
    return precision_at



# def dice_score(ypred, ytrue, eps=1):
#     '''Per-image dice score'''
#     d = torch.sum(ytrue, dim=[2,3]) + torch.sum(ypred, dim=[2,3]) + eps
#     n = 2* torch.sum(ytrue * ypred, dim=[2,3] ) +eps
#     return torch.mean(n/d, dim=1)

# def dice_loss(ypred, ytrue):
#     return 1 - dice_score(ypred,ytrue)

# def dice_entropy_loss(ypred, ytrue, alpha=0.1, weight_map=1):
#     return (  dice_loss(ypred, ytrue)[:,np.newaxis, np.newaxis]*alpha 
#             + torch.nn.functional.binary_cross_entropy(ypred, ytrue, reduction='none')*(1-alpha)*weight_map ).mean()


# def smoothed_cross_entropy(ypred, ytrue, alpha=0.01, ignore_index=-100):
#     mask  = (ytrue != ignore_index )
#     ypred = ypred[mask]
#     ytrue = ytrue[mask]
    
#     ypred   = torch.nn.functional.log_softmax(ypred, 1)
#     alpha_i = alpha / ypred.size(-1)
#     loss    = -(  ypred.gather(dim=-1, index=ytrue[:,np.newaxis]) * (1-alpha)
#                 + ypred.sum(dim=-1, keepdim=True)*alpha_i)
#     return torch.nan_to_num(loss.mean())

# def low_confidence_loss(ypred, ytrue, index=-100):
#     mask  = ytrue == index
#     loss  = torch.nn.functional.mse_loss(ypred, torch.zeros_like(ypred), reduction='none')
#     return (loss*mask[:, None]).mean()


class PrintMetricsCallback:
    '''Prints metrics after each training epoch in a compact table'''
    def __init__(self):
        self.epoch = 0
        self.logs  = {}
        
    def on_epoch_end(self, epoch):
        self.epoch = epoch + 1
        self.logs  = {}
        print() #newline
    
    def on_batch_end(self, logs, batch_i, n_batches):
        self.accumulate_logs(logs)
        percent     = ((batch_i+1) / n_batches)
        metrics_str = ' | '.join([f'{k}:{float(np.mean(v)):>9.5f}' for k,v in self.logs.items()])
        print(f'[{self.epoch:04d}|{percent:.2f}] {metrics_str}', end='\r')
    
    def accumulate_logs(self, newlogs):
        for k,v in newlogs.items():
            self.logs[k] = self.logs.get(k, []) + [v]

class TrainingProgressCallback:
    '''Passes training progress as percentage to a custom callback function'''
    def __init__(self, callback_fn, epochs):
        self.n_epochs    = epochs
        self.epoch       = 0
        self.callback_fn = callback_fn
    
    def on_batch_end(self, logs, batch_i, n_batches):
        percent     = ((batch_i+1) / (n_batches*self.n_epochs))
        percent    += self.epoch / self.n_epochs
        self.callback_fn(percent)
    
    def on_epoch_end(self, epoch):
        self.epoch = epoch + 1
