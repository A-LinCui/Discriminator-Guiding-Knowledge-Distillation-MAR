import abc
import os
import six

import torch
from torch import Tensor
import torch.nn as nn
import torchvision
from torch.nn.modules.loss import _Loss 
from pytorch_msssim import SSIM, MS_SSIM

from aw_nas import utils
from aw_nas.utils.exception import expect
from aw_nas.objective.base import BaseObjective
from aw_nas.utils import RegistryMeta
from aw_nas.utils import DataParallel
from aw_nas.utils import DistributedDataParallel
from aw_nas.utils.torch_utils import accuracy

import sys
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
from mar_final_trainer import _warmup_update_lr, cal_distance


class VGGPerceptionLoss(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean"):
        super(VGGPerceptionLoss, self).__init__(size_average, reduce, reduction)
        self.model = torchvision.models.vgg16(pretrained = True).eval()

    def forward(self, input: Tensor, target: Tensor):
        # Data shape: [batch, dimension, depth, length, width] / [batch, dimension, length, width] 
        assert input.shape == target.shape, "Shapes of input and target don't match"
        self.model.to(input.device)
        if len(input.shape) == 4:  # 2D data
            output = self.model(input.repeat(1, 3, 1 ,1))
            real_output = self.model(target.repeat(1, 3, 1, 1))
            loss = sum([(o - real_o).norm(2) for (o, real_o) in zip(output, target)])
        elif len(input.shape) == 5:  # 3D data
            output = [self.model(_.permute(1, 0, 2, 3)) for _ in input.repeat(1, 3, 1, 1, 1)]
            real_output = [self.model(_.permute(1, 0, 2, 3)) for _ in target.repeat(1, 3, 1, 1, 1)]
            loss = sum([(o - real_o).norm(2) for (o, real_o) in zip(output, real_output)])
        return loss / len(input) if self.reduction == "mean" else loss


class SSIMLoss(_Loss):
    def __init__(self, data_range: float = 1.0, channel: int = 1, size_average = None, reduce = None, reduction: str = "mean"):
        super(SSIMLoss, self).__init__(size_average, reduce, reduction)
        self.inner_ssim = SSIM(data_range = data_range, size_average = True, channel = channel)

    def forward(self, input: Tensor, target: Tensor):
        # Data shape: [batch, dimension, depth, length, width] / [batch, dimension, length, width]
        assert input.shape == target.shape, "Shapes of input and target don't match"
        if len(input.shape) == 4:  # 2D data
            avg_loss = 1 - self.inner_ssim(input, target)
            return avg_loss if self.reduction == "mean" else avg_loss * len(input)
        elif len(input.shape) == 5:  # 3D data
            avg_loss = sum([1 - self.inner_ssim(img.permute(1, 0, 2, 3), label.permute(1, 0, 2, 3)) for (img, label) in zip(input, target)]) / len(input)
        return avg_loss if self.reduction == "mean" else avg_loss * len(input)


class MSSIMLoss(SSIMLoss):
    def __init__(self, data_range: float = 1.0, channel: int = 1, size_average = None, reduce = None, reduction: str = "mean"):
        super(MSSIMLoss, self).__init__(data_range, channel, size_average, reduce, reduction)
        self.inner_ssim = MS_SSIM(data_range = data_range, size_average = True, channel = channel)


class RRMSELoss(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean"):
        super(RRMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor):
        assert input.shape == target.shape, "Shapes of input and target don't match"
        loss = sum([(img - label).norm(2) / label.norm(2) for (img, label) in zip(input, target)])
        return loss / len(input) if self.reduction == "mean" else loss


class PSNRLoss(_Loss):
    def __init__(self, size_average = None, reduce = None, reduction: str = "mean", max_value: float = 1.0):
        super(PSNRLoss, self).__init__(size_average, reduce, reduction)
        self.max_value = max_value

    def forward(self, input: Tensor, target: Tensor):
        assert input.shape == target.shape, "Shapes of input and target don't match"
        loss = sum([self.psnr(img, label) for (img, label) in zip(input, target)])
        return loss / len(input) if self.reduction == "mean" else loss
        
    @staticmethod
    def mse(input: Tensor, target: Tensor):
        return (input - target).pow(2).mean()

    def psnr(self, input: Tensor, target: Tensor):
        return 10 * (2 * self.max_value - self.mse(input, target))


AVAILABLE_LOSS = {
        "RRMSELoss": lambda size_average = None, reduce = None, reduction = "mean": \
                RRMSELoss(size_average, reduce, reduction),
        "PSNRLoss": lambda size_average = None, reduce = None, reduction = "mean", max_value = 1.0: \
                PSNRLoss(size_average, reduce, reduction, max_value),
        "SSIMLoss": lambda data_range = 1.0, channel = 1, size_average = None, reduce = None, reduction = "mean": \
                SSIMLoss(data_range, channel, size_average, reduce, reduction),
        "MSSIMLoss": lambda data_range = 1.0, channel = 1, size_average = None, reduce = None, reduction = "mean": \
                MSSIMLoss(data_range, channel, size_average, reduce, reduction),
        "VGGPerceptionLoss": lambda size_average = None, reduce = None, reduction = "mean": \
                VGGPerceptionLoss(size_average, reduce, reduction)
        }


class AttentionLoss(_Loss):
    """
    Attention loss for metal artifact reduction.
    
    Metal artifacts are too spartial to be learned well in 3D data.
    Therefore, we let the model pay attention to metal artifacts by setting a larger loss 
    coefficient for those slices with metal artifacts.
    According to the empirical observation, the maximum pixel value of slices with metal 
    artifacts are much larger than clean ones.
    Thus, we set a threshold for slice maximum value to distinguish clean slices and metal 
    artifact slices.

    Args:
        * threshold: [float] threshold to distinguish attention area
        * attention_coeff: [float] loss coeff of the attention area
        * inner_criterion_type: type of the inner criterion
        * inner_criterion_kwargs: kwargs of the inner criterion
    """

    def __init__(self, threshold: float = 0.2, attention_coeff: float = 2.0,
                 inner_criterion_type: str = "MSELoss", inner_criterion_kwargs = None,
                 size_average = None, reduce = None, reduction: str = "mean"):
        super(AttentionLoss, self).__init__(size_average, reduce, reduction)
        self.threshold = threshold
        self.attention_coeff = attention_coeff

        # Initialize inner criterion
        inner_criterion_kwargs = inner_criterion_kwargs or {}
        if inner_criterion_type in AVAILABLE_LOSS.keys():
            self.criterion = AVAILABLE_LOSS[inner_criterion_type](**inner_criterion_kwargs)
        else:
            self.criterion = getattr(nn, inner_criterion_type)(**inner_criterion_kwargs)

    def forward(self, input: Tensor, output: Tensor, target: Tensor):
        assert len(input.shape) == 5, "AttentionLoss only supports 3D data"
        # Calculate the attention map
        attention = torch.sum(torch.sum(input > self.threshold, dim = -1), dim = -1)
        attention_map = (attention > 0)
        non_attention_map = (attention <= 0)
        # Calculate the loss
        loss = 0
        for i in range(len(input)):
            output_slice = output[i].permute(1, 0, 2, 3)
            target_slice = target[i].permute(1, 0, 2, 3)

            slice_attention = attention_map[i].permute(1, 0)
            attention_loss = self.criterion(
                    output_slice[slice_attention], target_slice[slice_attention])

            slice_non_attention = non_attention_map[i].permute(1, 0)
            non_attention_loss = self.criterion(
                    output_slice[slice_non_attention], target_slice[slice_non_attention])

            attention_slice_ratio = slice_attention.sum().item() / len(output_slice)
            loss += self.attention_coeff * attention_slice_ratio * attention_loss + \
                    (1 - attention_slice_ratio) * non_attention_loss

        return loss / len(input) if self.reduction == "mean" else loss


class BaseMARObjective(BaseObjective):
    def __init__(self, search_space = None, schedule_cfg = None):
        super(BaseMARObjective, self).__init__(search_space, schedule_cfg)
   
    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def get_reward(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        # Currently, we use l2 distance as reward
        return cal_distance(outputs, targets)[1]
   
    @staticmethod
    def _init_criterion(criterion_type: str, criterion_kwargs = None):
        criterion_kwargs = criterion_kwargs or {}
        if criterion_type in AVAILABLE_LOSS.keys():
            criterion = AVAILABLE_LOSS[criterion_type](**criterion_kwargs)
        else:
            criterion = getattr(nn, criterion_type)(**criterion_kwargs)
        return criterion


class MARObjective(BaseMARObjective):
    """
    Base objective for metal artifact reduction.

    Args:
        * criterion_type: [str] type of the used criterion
        * criterion_kwargs: kwargs of the criterion
    """

    NAME = "mar_objective"

    def __init__(self, search_space, criterion_type: str = "MSELoss", criterion_kwargs = None, schedule_cfg = None):
        super(MARObjective, self).__init__(search_space, schedule_cfg)
        self._criterion = self._init_criterion(criterion_type, criterion_kwargs)
    
    def perf_names(self):
        return ["l1", "l2"]

    def get_perfs(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        return cal_distance(outputs, targets)

    def get_loss(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net, 
                 add_controller_regularization: bool = True, add_evaluator_regularization: bool = True):
        return self._criterion(outputs, targets)


class MARAttentionObjective(MARObjective):
    """
    Objective with attention mechanism for metal artifact reduction.

    Args:
        * criterion_type: [str] type of the used inner criterion
        * criterion_kwargs: kwargs of the inner criterion
        * threshold: [float] threshold for attention area
        * attention_coeff: [float] coeff for loss of attention area
    """
    
    NAME = "mar_attention_objective"
    
    def __init__(self, search_space, threshold: float = 2.0, attention_coeff: float = 5.0,
                criterion_type: str = "MSELoss", criterion_kwargs = None, schedule_cfg = None):
        BaseObjective.__init__(self, search_space, schedule_cfg)
        self._criterion = AttentionLoss(threshold, attention_coeff, criterion_type, criterion_kwargs)

    def get_loss(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net, 
                 add_controller_regularization: bool = True, add_evaluator_regularization: bool = True):
        return self._criterion(inputs, outputs, targets)


class MARHybridObjective(BaseMARObjective):
    """
    Hybrid Objective for metal artifact reduction.

    Args:
        * criterion_list: [list] list of the used criterions
    """
    
    NAME = "mar_hybrid_objective"

    def __init__(self, search_space, criterion_list: list, schedule_cfg = None):
        super(MARHybridObjective, self).__init__(search_space, schedule_cfg)
        self._criterions = [self._init_criterion_from_cfg(criterion) for criterion in criterion_list]
        self._criterion_coeffs = [criterion["coeff"] for criterion in criterion_list]
        self._criterion_names = [criterion["criterion_type"] for criterion in criterion_list]

    def _init_criterion_from_cfg(self, criterion: dict):
        criterion_type = criterion["criterion_type"]
        criterion_kwargs = criterion["criterion_kwargs"] if "criterion_kwargs" in criterion else None
        return self._init_criterion(criterion_type, criterion_kwargs)

    def perf_names(self):
        return ["l1", "l2"] + self._criterion_names
    
    def get_perfs(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        return cal_distance(outputs, targets) + [loss.item() for loss in self._get_seperate_loss(inputs, outputs, targets, cand_net)]
    
    def get_loss(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net, 
            add_controller_regularization: bool = True, add_evaluator_regularization: bool = True):
        return sum(self._get_seperate_loss(inputs, outputs, targets, cand_net))

    def _get_seperate_loss(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        loss_list = []
        for (criterion, coeff) in zip(self._criterions, self._criterion_coeffs):
            loss = criterion(outputs, targets)
            loss_list.append(coeff * loss)
        return loss_list


class ParameterizedObjective(BaseObjective):
    """
    Parameterized objective for metal artifact reduction.

    Args:
        * model_type: [str] type of the inner model
        * model_cfg: configuration of the inner model
        * multiprocess: [bool]
        * save_as_state_dict: [bool]
    """
    
    def __init__(self, model_type: str, model_cfg = None, multiprocess: bool = False, 
                 save_as_state_dict: bool = True, search_space = None, schedule_cfg = None):
        super(ParameterizedObjective, self).__init__(search_space, schedule_cfg)
        self.device = None
        self.gpus = None
        self.multiprocess = multiprocess

        self.model_type = model_type
        self.model_cfg = model_cfg or {}

        self.model = None
        self.parallel_model = None

        self._is_setup = False
        self.save_as_state_dict = save_as_state_dict

    def _init_model(self, device, gpus):
        self.device = device
        self.gpus = gpus
        # Initialize the teacher
        model_cls = RegistryMeta.get_class("final_model", self.model_type)
        self.model_cfg["device"] = device
        self.model_cfg["search_space"] = None
        self.model = model_cls(**(self.model_cfg))

    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self.model).to(self.device)
            self.parallel_model = DistributedDataParallel(net, self.gpus, find_unused_parameters = True)
        elif len(self.gpus) >= 2:
            self.parallel_model = DataParallel(self.model, self.gpus).to(self.device)
        else:
            self.parallel_model = self.model

    def load(self, path):
        # load the model
        m_path = os.path.join(path, "model.pt") if os.path.isdir(path) else path
        if not os.path.exists(m_path):
            m_path = os.path.join(path, "model_state.pt")
            self._load_state_dict(m_path)
        else:
            self.model = torch.load(m_path, map_location = torch.device("cpu"))
        self.model.to(self.device)
        self._parallelize()
    
    def save(self, path):
        rank = (os.environ.get("LOCAL_RANK"))
        if rank is not None and rank != '0':
            return
        path = utils.makedir(path)
        if self.save_as_state_dict:
            torch.save(self.model.state_dict(), os.path.join(path, "model_state.pt"))
        else:
            # save the model directly instead of the state_dict,
            # so that it can be loaded and run directly, without specificy configuration
            torch.save(self.model, os.path.join(path, "model.pt"))
        self.logger.info("Saved {} checkpoint to {}".format(self.NAME, path))
    
    def _load_state_dict(self, path):
        checkpoint = torch.load(path, map_location = torch.device("cpu"))
        extra_keys = set(checkpoint.keys()).difference(set(self.model.state_dict().keys()))
        if extra_keys:
            self.logger.error("%d extra keys in checkpoint! "
                              "Make sure the genotype match", len(extra_keys))
        missing_keys = {key for key in set(self.model.state_dict().keys())\
                        .difference(checkpoint.keys()) \
                        if "auxiliary" not in key}
        if missing_keys:
            self.logger.error(("{} missing keys will not be loaded! Check your genotype, "
                               "This should be due to you're using the state dict dumped by"
                               " `awnas eval-arch --save-state-dict` in an old version, "
                               "and your genotype actually skip some "
                               "cells, which might means, many parameters of your "
                               "sub-network is not actually active, "
                               "and this genotype might not be so effective.")
                              .format(len(missing_keys)))
            self.logger.error(str(missing_keys))
        self.logger.info(self.model.load_state_dict(checkpoint, strict = False))

    def get_loss(self, inputs, outputs, targets, cand_net, 
                 add_controller_regularization = True, add_evaluator_regularization = True):
        return NotImplemented
    
    def setup(self, load = None, load_state_dict = None):
        expect(not (load is not None and load_state_dict is not None),
                "`load` and `load_state_dict` cannot be passed simultaneously.")
        if load is not None:
            self.load(load)
        else:
            assert self.model is not None
            if load_state_dict is not None:
                self._load_state_dict(load_state_dict)

            self.logger.info("{} param size = {} M".format(
                self.NAME, utils.count_parameters(self.model, count_binary=False)/1.e6))
            self._parallelize()
        self._is_setup = True


class DistillationObjective(ParameterizedObjective):
    """
    Distillation objective for metal artifact reduction.

    Args:
        * model_type: [str] type of the teacher model
        * model_cfg: configuration of the teacher model
        * model_path: [str] path of the teacher model
        * multiprocess: [bool]
        * save_as_state_dict: [bool]
    """

    NAME = "distillation_objective"

    def __init__(self, model_type: str, model_path: str, model_cfg = None, 
            multiprocess: bool = False, save_as_state_dict: bool = True, search_space = None, schedule_cfg = None):
        super(DistillationObjective, self).__init__(
                model_type, model_cfg, multiprocess, save_as_state_dict, search_space, schedule_cfg)
        self.model_path = model_path
        self._criterion = nn.L1Loss()

    def setup(self, load = None, load_state_dict = None):
        try:
            super(DistillationObjective, self).setup(load_state_dict = self.model_path)
        except:
            super(DistillationObjective, self).setup(load = self.model_path)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def get_reward(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        return NotImplemented

    def perf_names(self):
        return ["similarity"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        return NotImplemented

    def get_teacher_outputs(self, inputs):
        self.parallel_model.eval()
        with torch.no_grad():
            teacher_outputs = self.parallel_model(inputs).detach()
        return teacher_outputs

    def get_loss(self, inputs, outputs, targets, cand_net, teacher_outputs = None,
                 add_controller_regularization = True, add_evaluator_regularization = True):
        if teacher_outputs is None:
            teacher_outputs = self.get_teacher_outputs(inputs)
        if outputs is None:
            outputs = cand_net(inputs)
        return self._criterion(outputs, teacher_outputs) / len(inputs)


class DiscriminationObjective(ParameterizedObjective):
    """
    Discrimination objective for metal artifact reduction.

    Args:
        * model_type: [str] type of the discriminator
        * model_cfg: configuration of the discriminator
        * multiprocess: [bool]
        * save_as_state_dict: [bool]
        Other arguments are discriminator training settings
    """

    NAME = "discrimination_objective"

    def __init__(self, model_type: str, model_cfg = None, 
                 optimizer_type: str = "SGD", optimizer_kwargs = None,
                 multiprocess: bool = False, save_as_state_dict: bool = True,
                 learning_rate: float = 0.025, momentum: float = 0.9,
                 warmup_epochs: int = 0,
                 optimizer_scheduler = None,
                 weight_decay: float = 3e-4, no_bias_decay = False,
                 grad_clip: float = 5.0,
                 add_regularization: bool = True,
                 search_space = None, schedule_cfg = None):
        super(DiscriminationObjective, self).__init__(
                model_type, model_cfg, multiprocess, save_as_state_dict, search_space, schedule_cfg)

        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_scheduler_cfg = optimizer_scheduler

        # Training setting
        self._criterion = nn.CrossEntropyLoss().to(self.device)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.no_bias_decay = no_bias_decay
        self.add_regularization = add_regularization

    def _init_model(self, device, gpus):
        # Initialize discriminator
        super(DiscriminationObjective, self)._init_model(device, gpus)
        # Initialize optimizer and scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def get_reward(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        return self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def perf_names(self):
        return ["acc"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """ 
        Return the accuracy that the discriminator successfully picks out the fake inputs 
        """
        self.parallel_model.eval()
        logits = self.parallel_model(outputs)
        fake_targets = torch.zeros(len(inputs), dtype = torch.long).to(inputs.device)
        return [float(accuracy(logits, fake_targets)[0]) / 100]

    def step(self, inputs, outputs, targets, update = True):
        """
        Update the discriminator for one step
        The inputs is the original inputs with artifacts
        The outputs is the corresponding outputs of the generator
        The targets is the original targets without artifacts
        """
        if update:
            self.parallel_model.train()
            self.optimizer.zero_grad()

        batch_size = len(outputs)

        # Assemble the inputs and labels
        fake_targets = torch.zeros(batch_size, dtype = torch.long)
        real_targets = torch.ones(batch_size, dtype = torch.long)
        assemble_targets = torch.cat((fake_targets, real_targets)).to(inputs.device)
        assemble_inputs = torch.cat((outputs, targets)).to(inputs.device)

        #  Get the loss
        logits = self.parallel_model(assemble_inputs)
        loss = self._criterion(logits, assemble_targets)

        if update:
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        acc = utils.accuracy(logits, assemble_targets, topk = (1, 1))[0] / 100.

        return loss.item(), acc.item()

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization = True, add_evaluator_regularization = True):
        """
        Label of real data: 1
        Label of fake data: 0

        Return loss of the discriminator to predict the generator's outputs as real data
        and return success rate for the generator to fool the discriminator
        """
        self.parallel_model.eval()
        logits = self.parallel_model(outputs)
        real_labels = torch.ones(len(inputs), dtype = torch.long).to(inputs.device)
        loss = self._criterion(logits, real_labels)
        return loss

    def update_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
        elif self.scheduler is not None and epoch != 1:
            self.scheduler.step()
        return self.optimizer.param_groups[0]["lr"]

    def _init_optimizer(self):
        group_weight = []
        group_bias = []
        for name, param in self.model.named_parameters():
            if "bias" in name:
                group_bias.append(param)
            else:
                group_weight.append(param)
        assert len(list(self.model.parameters())) == len(group_weight) + len(group_bias)
        optim_cls = getattr(torch.optim, self.optimizer_type)
        if self.optimizer_type == "Adam":
            optim_kwargs = {
                "lr": self.learning_rate,
                "weight_decay": self.weight_decay
            }
        else:
            optim_kwargs = {
                "lr": self.learning_rate,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay
            }
        optim_kwargs.update(self.optimizer_kwargs or {})
        optimizer = optim_cls(
            [{"params": group_weight},
             {"params": group_bias,
              "weight_decay": 0 if self.no_bias_decay else self.weight_decay}],
            **optim_kwargs)

        return optimizer

    @staticmethod
    def _init_scheduler(optimizer, cfg):
        if cfg:
            cfg = {k:v for k, v in six.iteritems(cfg)}
            sch_cls = utils.get_scheduler_cls(cfg.pop("type"))
            return sch_cls(optimizer, **cfg)
        return None
    
    def save(self, path):
        super(DiscriminationObjective, self).save(path)
        torch.save({
            "optimizer":self.optimizer.state_dict()
        }, os.path.join(path, "optimizer.pt"))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(path, "scheduler.pt"))

    def load(self, path):
        # load the model
        super(DiscriminationObjective, self).load(path)
        log_strs = ["model from {}".format(path)]

        # init the optimzier/scheduler
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer, self.optimizer_scheduler_cfg)

        o_path = os.path.join(path, "optimizer.pt") if os.path.isdir(path) else None
        if o_path and os.path.exists(o_path):
            checkpoint = torch.load(o_path, map_location=torch.device("cpu"))
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            log_strs.append("optimizer from {}".format(o_path))

        if self.scheduler is not None:
            s_path = os.path.join(path, "scheduler.pt") if os.path.isdir(path) else None
            if s_path and os.path.exists(s_path):
                self.scheduler.load_state_dict(torch.load(s_path, map_location=torch.device("cpu")))
                log_strs.append("scheduler from {}".format(s_path))

        self.logger.info("Loaded objective checkpoint from %s: %s", path, ", ".join(log_strs))


class DiscriminatorGuidingDistillationObjective(BaseObjective):
    """
    Discriminator guiding knowledge distillation (DGKD) objective for metal artifact reduction.

    Args:
        * discrimination_cfg: configuration of the discrimination objective
        * distillation_cfg: configuration of the distillation objective
        * distillation_coeff: [float] coeff of the distillation loss
    """

    NAME = "discriminator_guiding_distillation_objective"

    def __init__(self, discrimination_cfg, distillation_cfg, distillation_coeff: float = 0.99995, search_space = None, schedule_cfg = None):
        super(DiscriminatorGuidingDistillationObjective, self).__init__(search_space, schedule_cfg)
        self.distillation_objective = DistillationObjective(search_space = search_space, **distillation_cfg)
        self.discrimination_objective = DiscriminationObjective(search_space = search_space, **discrimination_cfg)
        self.distillation_coeff = distillation_coeff
    
    def _init_model(self, device, gpus):
        self.distillation_objective._init_model(device, gpus)
        self.discrimination_objective._init_model(device, gpus)
    
    def update_lr(self, epoch):
        return self.discrimination_objective.update_lr(epoch)

    def step(self, inputs, outputs, targets, update = True):
        return self.discrimination_objective.step(inputs, outputs, targets, update)
    
    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def get_reward(self, inputs: Tensor, outputs: Tensor, targets: Tensor, cand_net):
        return NotImplemented

    def perf_names(self):
        return ["loss", "distillation_loss", "discrimination_loss", "acc"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        loss, distillation_loss, discrimination_loss = \
                self.get_loss(inputs, outputs, targets, cand_net, return_all = True)
        acc = self.discrimination_objective.get_perfs(inputs, outputs, targets, cand_net)
        return loss.item(), distillation_loss.item(), discrimination_loss.item(), acc

    def get_loss(self, inputs, outputs, targets, cand_net, return_all = False,
                 add_controller_regularization = True, add_evaluator_regularization = True):
        teacher_outputs = self.distillation_objective.get_teacher_outputs(inputs)
        if outputs is None:
            outputs = cand_net(inputs)
        distillation_loss = self.distillation_objective.get_loss(inputs, outputs, targets, cand_net, teacher_outputs)
        discrimination_loss = self.discrimination_objective.get_loss(inputs, outputs, targets, cand_net)
        loss = self.distillation_coeff * distillation_loss + (1 - self.distillation_coeff) * discrimination_loss

        if return_all:
            return loss, distillation_loss, discrimination_loss
        else:
            return loss

    def setup(self, load = None, load_state_dict = None):
        self.distillation_objective.setup(load, load_state_dict)
        self.discrimination_objective.setup(load, load_state_dict)

    def save(self, path):
        return self.discrimination_objective.save(path)

    def load(self, path):
        return self.discrimination_objective.load(path)
    
    def _load_state_dict(self, path):
        return self.discrimination_objective._load_state_dict(path)


class TwoDimensionBasedThreeDimensionObjective(DiscriminationObjective):
    """
    Args:
        * threshold: [float] threshold to split slices with metal artifacts
    """

    NAME = "two_dimension_based_three_dimension_discrimination_objective"
    
    def __init__(self, threshold, **kwargs):
        super(TwoDimensionBasedThreeDimensionObjective, self).__init__(**kwargs)
        self.threshold = threshold

    @staticmethod
    def extract_artifact_slices(inputs, outputs, targets, threshold):
        """
        Calculate the attention mask according to the given threshold
        And assemble the new batches of two-dimension data
        [batch size, channel, depth, length, width] -> [batch size, channel, length, width]
        """

        artifact_mask = torch.sum(torch.sum(inputs > threshold, dim = -1), dim = -1) > 0
        assert artifact_mask.sum(), \
                "No slices found with artifacts. Check the dataset or adjust the threshold."
        
        two_dimension_inputs = torch.clone(inputs[artifact_mask]).unsqueeze(dim = 1)
        two_dimension_outputs = torch.clone(outputs[artifact_mask]).unsqueeze(dim = 1)
        two_dimension_targets = torch.clone(targets[artifact_mask]).unsqueeze(dim = 1)
        return two_dimension_inputs, two_dimension_outputs, two_dimension_targets

    def get_perfs(self, inputs, outputs, targets, cand_net):
        return super(TwoDimensionBasedThreeDimensionObjective, self).get_perfs(
                *self.extract_artifact_slices(inputs, outputs, targets, self.threshold), cand_net)
        
    def step(self, inputs, outputs, targets, update):
        return super(TwoDimensionBasedThreeDimensionObjective, self).step(
                *self.extract_artifact_slices(inputs, outputs, targets, update, self.threshold))

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization = True, add_evaluator_regularization = True):
        return super(TwoDimensionBasedThreeDimensionObjective, self).get_loss(
                *self.extract_artifact_slices(inputs, outputs, targets, self.threshold), 
                cand_net, add_controller_regularization, add_evaluator_regularization)


class TwoDimensionBasedDiscriminatorGuidingDistillationObjective(DiscriminatorGuidingDistillationObjective):
    NAME = "two_dimension_based_discriminator_guiding_distillation_objective"

    def __init__(self, discrimination_cfg, distillation_cfg, distillation_coeff: float = 0.5,
                 search_space = None, schedule_cfg = None):
        BaseObjective.__init__(self, search_space, schedule_cfg)
        self.distillation_objective = DistillationObjective(search_space = search_space, **distillation_cfg)
        self.discrimination_objective = TwoDimensionBasedThreeDimensionObjective(search_space = search_space, **discrimination_cfg)
        self.distillation_coeff = distillation_coeff
 

if __name__ == "__main__":
    cfg = {
            "objective_type": "mar_objective",
            "objective_cfg": {
                "criterion_type": "MSELoss",
                "criterion_kwargs": {"reduction": "sum"}
                }
            }
    type_ = cfg["objective" + "_type"]
    cfg = cfg.get("objective" + "_cfg", None)
    cls = RegistryMeta.get_class("objective", type_)
    objective = cls(search_space=None, **cfg)
