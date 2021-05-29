import copy
import os
import six

import torch
from torch import nn
from torch import Tensor
from torch.utils.data.distributed import DistributedSampler

from aw_nas import utils
from aw_nas.utils.common_utils import nullcontext
from aw_nas.utils.exception import expect
from aw_nas.utils.torch_utils import calib_bn
from aw_nas.final.cnn_trainer import CNNFinalTrainer
from aw_nas.final.base import FinalTrainer


try:
    from torch.nn import SyncBatchNorm
    convert_sync_bn = SyncBatchNorm.convert_sync_batchnorm
except ImportError:
    utils.getLogger("mar_trainer").warn(
        "Import convert_sync_bn failed! SyncBatchNorm might not work!")
    convert_sync_bn = lambda m: m


def cal_distance(logits: Tensor, targets: Tensor) -> list:
    l1_dis = torch.sum(torch.abs(logits - targets))
    l2_dis = sum([(logit - target).norm(2) for (logit, target) in zip(logits, targets)])
    return [l1_dis.item() / len(logits), l2_dis.item() / len(logits)]


def _warmup_update_lr(optimizer, epoch, init_lr, warmup_epochs):
    """
    update learning rate of optimizers
    """
    lr = init_lr * epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class MARFinalTrainer(CNNFinalTrainer):
    NAME = "mar_trainer"
    
    def evaluate_split(self, split):
        if len(self.gpus) >= 2:
            self._forward_once_for_flops(self.model)
        assert split in {"train", "test"}
        queue = self.valid_queue if split == "test" else self.train_queue

        l1, l2, obj, perfs = self.infer_epoch(queue, self.parallel_model, self.device)
        self.logger.info("l1 %f ; l2 %f; obj %f ; performance: %s", l1, l2, obj,
                         "; ".join(
                             ["{}: {:.3f}".format(n, v) for n, v in perfs.items()]))
        return l1, l2, obj

    def infer_epoch(self, valid_queue, model, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        l1_distance = utils.AverageMeter()
        l2_distance = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        all_perfs = []
        model.eval()
        origin_l1, origin_l2, l1, l2, origin_rmse, rmse = [], [], [], [], [], []

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for step, (inputs, targets) in enumerate(valid_queue):
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)

                loss = self._obj_loss(inputs, logits, targets, model,
                                      add_evaluator_regularization = self.add_regularization)
                perfs = self._perf_func(inputs, logits, targets, model)
                all_perfs.append(perfs)
                n = inputs.size(0)

                l1_dis, l2_dis = cal_distance(logits, targets)
                l1_distance.update(l1_dis, n)
                l2_distance.update(l2_dis, n)
                
                objs.update(loss.item(), n)
                del loss
                
                if step % self.report_every == 0:
                    all_perfs_by_name = list(zip(*all_perfs))
                    obj_perfs = {
                        k: self.objective.aggregate_fn(k, False)(v)
                        for k, v in zip(self._perf_names, all_perfs_by_name)
                    }
                    self.logger.info("valid %03d %e %f %f %s", step, objs.avg, l1_distance.avg, l2_distance.avg, \
                            "; ".join(["{}: {:.3f}".format(perf_n, v) for perf_n, v in obj_perfs.items()]))

        all_perfs_by_name = list(zip(*all_perfs))
        obj_perfs = {
            k: self.objective.aggregate_fn(k, False)(v)
            for k, v in zip(self._perf_names, all_perfs_by_name)
        }
        return l1_distance.avg, l2_distance.avg, objs.avg, obj_perfs

    def train_epoch(self, train_queue, model, optimizer, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        l1_distance = utils.AverageMeter()
        l2_distance = utils.AverageMeter()
        model.train()

        for step, (inputs, targets) in enumerate(train_queue):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = self._obj_loss(inputs, logits, targets, model, 
                                  add_evaluator_regularization = self.add_regularization)

            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            
            optimizer.step()

            l1_dis, l2_dis = cal_distance(logits, targets)
            n = inputs.size(0)
            objs.update(loss.item(), n)
            l1_distance.update(l1_dis, n)
            l2_distance.update(l2_dis, n)
            del loss

            if step % self.report_every == 0:
                self.logger.info("train %03d %.3f; %.2f; %.2f",
                                 step, objs.avg, l1_distance.avg, l2_distance.avg)

        return l1_distance.avg, l2_distance.avg, objs.avg

    def train(self):
        # save the model.log
        if self.train_dir is not None:
            with open(os.path.join(self.train_dir, "model.log"),"w") as f:
                f.write(str(self.model))
        
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.epoch = epoch
            self.on_epoch_start(epoch)
            
            if epoch <= self.warmup_epochs:
                _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
            elif self.scheduler is not None and epoch != 1:
                self.scheduler.step()
            self.logger.info("epoch %d lr %e", epoch, self.optimizer.param_groups[0]["lr"])

            train_l1, train_l2, train_obj = \
                    self.train_epoch(self.train_queue, self.parallel_model, self.optimizer, self.device)
            self.logger.info("train_l1 %f ; train_l2 %f ; train_obj %f", train_l1, train_l2, train_obj)

            if epoch % self.eval_every == 0:
                valid_l1, valid_l2, valid_obj, valid_perfs = \
                        self.infer_epoch(self.valid_queue, self.parallel_model, self.device)
                self.logger.info("valid_l1 %f ; valid_l2 %f ; valid_obj %f ; valid performances: %s", valid_l1, valid_l2, valid_obj,\
                        "; ".join(["{}: {:.3f}".format(n, v) for n, v in valid_perfs.items()]))

            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch))
                self.save(path)
            self.on_epoch_end(epoch)

        if self.train_dir:
            self.save(os.path.join(self.train_dir, "final"))

    def on_epoch_start(self, epoch):
        pass
        
    def on_epoch_end(self, epoch):
        pass


class MARArtificialFinalTrainer(MARFinalTrainer):
    """
    Final trainer for artificial data.
    Different from "MARFinalTrainer", implant area is distinguished with a threshold and not involved in training or inference.

    Args:
        * threshold: [float] threshold to distinguish implant area
    """

    NAME = "mar_artificial_trainer"

    def __init__(self, threshold, **kwargs):
        super(MARArtificialFinalTrainer, self).__init__(**kwargs)
        self.threshold = threshold
    
    def infer_epoch(self, valid_queue, model, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        l1_distance = utils.AverageMeter()
        l2_distance = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        all_perfs = []
        model.eval()

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for step, (inputs, targets) in enumerate(valid_queue):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                logits = model(inputs)

                # Generate the mask and assemble the overall outputs
                mask = (targets < self.threshold)
                logits = mask * logits + (~ mask) * targets

                loss = self._obj_loss(inputs, logits, targets, model,
                                      add_evaluator_regularization = self.add_regularization)
                perfs = self._perf_func(inputs, logits, targets, model)
                all_perfs.append(perfs)
                n = inputs.size(0)

                l1_dis, l2_dis = cal_distance(logits, targets)
                l1_distance.update(l1_dis, n)
                l2_distance.update(l2_dis, n)
                
                objs.update(loss.item(), n)
                del loss
                
                if step % self.report_every == 0:
                    all_perfs_by_name = list(zip(*all_perfs))
                    obj_perfs = {
                        k: self.objective.aggregate_fn(k, False)(v)
                        for k, v in zip(self._perf_names, all_perfs_by_name)
                    }
                    self.logger.info("valid %03d %e %f %f %s", step, objs.avg, l1_distance.avg, l2_distance.avg, \
                            "; ".join(["{}: {:.3f}".format(perf_n, v) for perf_n, v in obj_perfs.items()]))

        all_perfs_by_name = list(zip(*all_perfs))
        obj_perfs = {
            k: self.objective.aggregate_fn(k, False)(v)
            for k, v in zip(self._perf_names, all_perfs_by_name)
        }
        return l1_distance.avg, l2_distance.avg, objs.avg, obj_perfs

    def train_epoch(self, train_queue, model, optimizer, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        l1_distance = utils.AverageMeter()
        l2_distance = utils.AverageMeter()
        model.train()

        for step, (inputs, targets) in enumerate(train_queue):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            
            logits = model(inputs)

            # Generate the mask and assemble the overall outputs
            mask = (targets < self.threshold)
            logits = mask * logits + (~ mask) * targets

            loss = self._obj_loss(inputs, logits, targets, model, 
                                  add_evaluator_regularization = self.add_regularization)

            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            
            optimizer.step()

            l1_dis, l2_dis = cal_distance(logits, targets)
            n = inputs.size(0)
            objs.update(loss.item(), n)
            l1_distance.update(l1_dis, n)
            l2_distance.update(l2_dis, n)
            del loss

            if step % self.report_every == 0:
                self.logger.info("train %03d %.3f; %.2f; %.2f",
                                 step, objs.avg, l1_distance.avg, l2_distance.avg)

        return l1_distance.avg, l2_distance.avg, objs.avg


class MARDisciminatorGuidingDistillationFinalTrainer(CNNFinalTrainer): 
    """
    Final trainer of discriminator guiding knowledge distillation

    Args:
        * train_discriminator_every: [int]
    """

    NAME = "mar_discriminator_guiding_distillation_trainer"
    
    def __init__(self, train_discriminator_every: int = 2, **kwargs):
        super(MARDisciminatorGuidingDistillationFinalTrainer, self).__init__(**kwargs)
        self.train_discriminator_every = train_discriminator_every

    def setup(self, load=None, load_state_dict=None,
              save_every=None, train_dir=None, report_every=50):
        super(MARDisciminatorGuidingDistillationFinalTrainer, self).setup(
                load, load_state_dict, save_every, train_dir, report_every)
        # initialize the discriminator and teacher
        self.objective._init_model(self.device, self.gpus)
        objective_path = os.path.join(load, "objective") if load else None
        if objective_path and os.path.exists(objective_path):
            self.objective.setup(load = objective_path)
        else:
            self.objective.setup()
        
    def save(self, path):
        super(MARDisciminatorGuidingDistillationFinalTrainer, self).save(path)
        self.objective.save(os.path.join(path, "objective"))

    def evaluate_split(self, split):
        if len(self.gpus) >= 2:
            self._forward_once_for_flops(self.model)
        
        assert split in {"train", "test"}
        queue = self.valid_queue if split == "test" else self.train_queue
        
        obj, perfs = self.infer_epoch(queue, self.parallel_model, self.device)
        self.logger.info("valid_obj %f ; valid performances: %s", obj, \
                "; ".join(["{}: {:.3f}".format(n, v) for n, v in perfs.items()]))
        
        return obj, perfs

    def train(self):
        # Save the model.log
        if self.train_dir is not None:
            with open(os.path.join(self.train_dir, "model.log"),"w") as f:
                f.write(str(self.model))
        
        # Start training
        for epoch in range(self.last_epoch + 1, self.epochs + 1):
            self.epoch = epoch

            # Update learning rates
            if epoch <= self.warmup_epochs:
                _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
            elif self.scheduler is not None and epoch != 1:
                self.scheduler.step()
            self.logger.info("epoch %d lr %e %e", epoch, \
                    self.optimizer.param_groups[0]["lr"], self.objective.update_lr(epoch))

            # Train an epoch
            obj, distillation_obj, discrimination_obj, discriminator_obj, discriminator_acc = \
                    self.train_epoch(self.train_queue, self.parallel_model, self.optimizer, self.device)
            self.logger.info("train: obj %.5f; distillation_obj %.5f; discrimination_obj %.5f; discriminator_obj %.5f; discriminator_acc %.5f", \
                    obj, distillation_obj, discrimination_obj, discriminator_obj, discriminator_acc) 

            # Infer for an epoch
            if epoch % self.eval_every == 0:
                obj, perfs = self.infer_epoch(self.valid_queue, self.parallel_model, self.device)
                self.logger.info("valid_obj %f ; valid performances: %s", obj, \
                        "; ".join(["{}: {:.5f}".format(n, v) for n, v in perfs.items()]))

            # Save the checkpoint
            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch))
                self.save(path)

        if self.train_dir:
            path = os.path.join(self.train_dir, "final")
            self.save(path)

    def infer_epoch(self, valid_queue, model, device):
        expect(self._is_setup, "trainer.setup should be called first")

        # Record statistics
        objs = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        all_perfs = []

        # Start evaluation
        model.eval()
        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for step, (inputs, targets) in enumerate(valid_queue):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                
                perfs = self._perf_func(inputs, outputs, targets, model)
                all_perfs.append(perfs)

                # Record statistics
                n = inputs.size(0)
                objs.update(perfs[0], n)

                if step % self.report_every == 0:
                    all_perfs_by_name = list(zip(*all_perfs))
                    obj_perfs = {
                            k: self.objective.aggregate_fn(k, False)(v) 
                            for k, v in zip(self._perf_names, all_perfs_by_name)
                    }
                    self.logger.info("valid %03d %e %s", step, objs.avg, "; ".join(
                        ["{}: {:.3f}".format(perf_n, v) for perf_n, v in obj_perfs.items()]))

        all_perfs_by_name = list(zip(*all_perfs))
        obj_perfs = {
                k: self.objective.aggregate_fn(k, False)(v) 
                for k, v in zip(self._perf_names, all_perfs_by_name)
        }
        return objs.avg, obj_perfs

    def train_epoch(self, train_queue, model, optimizer, device):
        expect(self._is_setup, "trainer.setup should be called first")

        # Record losses
        objs = utils.AverageMeter()
        distillation_objs = utils.AverageMeter()
        discrimination_objs = utils.AverageMeter()

        discriminator_objs = utils.AverageMeter()
        discriminator_acc = utils.AverageMeter() # Record the accuracy of the discriminator

        # Start training
        for step, (inputs, targets) in enumerate(train_queue):
            inputs = inputs.to(device)
            targets = targets.to(device)

            """
            At first step, we update the generator
            In this step, the discriminator is frozen
            """
            model.train()
            optimizer.zero_grad()

            # Update the generator
            loss, distillation_loss, discrimination_loss = self._obj_loss(inputs, None, 
                    targets, model, True, add_evaluator_regularization = self.add_regularization)
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            
            # Record statistics
            n = inputs.size(0)
            objs.update(loss.item(), n)
            distillation_objs.update(distillation_loss.item(), n)
            discrimination_objs.update(discrimination_loss.item(), n)
            del loss, distillation_loss, discrimination_loss

            """
            At second step, we update the discriminator with both fake and real data
            In this step, the generator is frozen
            """
            model.eval()
            
            # Update the discriminator
            with torch.no_grad():
                outputs = model(inputs).detach()
            loss, acc = self.objective.step(inputs, outputs, targets, update = self.epoch % self.train_discriminator_every == 0)
            
            # Record statistics
            discriminator_objs.update(loss, n) 
            discriminator_acc.update(acc, n)

            if step % self.report_every == 0:
                self.logger.info("train: %03d %.3f; %.3f; %.3f; %.3f; %.3f", step, objs.avg, 
                        distillation_objs.avg, discrimination_objs.avg, discriminator_objs.avg, discriminator_acc.avg)
                
        return objs.avg, distillation_objs.avg, discrimination_objs.avg, discriminator_objs.avg, discriminator_acc.avg


class MARDiscriminatorTrainer(CNNFinalTrainer):
    """
    Trainer used to train the discriminator statically.
    """

    NAME = "mar_discriminator_trainer"
    
    def train_epoch(self, train_queue, model, criterion, optimizer, device, epoch):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        model.train()

        for step, (inputs, target) in enumerate(train_queue):
            inputs = inputs.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = self._obj_loss(inputs, logits, target, model, add_evaluator_regularization = self.add_regularization)
            loss.backward()

            if self.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            prec1 = utils.accuracy(logits, target)[0]
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            del loss

            if step % self.report_every == 0:
                self.logger.info("train %03d %.3f; %.2f%%", step, objs.avg, top1.avg)

        return top1.avg, objs.avg

    def infer_epoch(self, valid_queue, model, criterion, device):
        expect(self._is_setup, "trainer.setup should be called first")
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        all_perfs = []
        model.eval()

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for step, (inputs, target) in enumerate(valid_queue):
                inputs = inputs.to(device)
                target = target.to(device)

                logits = model(inputs)
                loss = criterion(logits, target)
                perfs = self._perf_func(inputs, logits, target, model)
                all_perfs.append(perfs)
                prec1 = utils.accuracy(logits, target)[0]
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                del loss
                if step % self.report_every == 0:
                    all_perfs_by_name = list(zip(*all_perfs))
                    # support use objective aggregate fn, for stat method other than mean
                    # e.g., adversarial distance median; detection mAP (see det_trainer.py)
                    obj_perfs = {
                        k: self.objective.aggregate_fn(k, False)(v)
                        for k, v in zip(self._perf_names, all_perfs_by_name)
                    }
                    self.logger.info("valid %03d %e %f %s", step, objs.avg, top1.avg,
                                     "; ".join(["{}: {:.3f}".format(perf_n, v) for perf_n, v in obj_perfs.items()]))
        all_perfs_by_name = list(zip(*all_perfs))
        obj_perfs = {
            k: self.objective.aggregate_fn(k, False)(v)
            for k, v in zip(self._perf_names, all_perfs_by_name)
        }
        return top1.avg, objs.avg, obj_perfs
