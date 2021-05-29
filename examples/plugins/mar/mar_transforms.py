import abc
import random
import copy

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as F
from skimage import transform as skitransform

# Basic operations
# --------------------------------------------------------------------------------

def crop_3d(img, shallow : int, top: int, left: int, depth: int, height: int, width: int):
    # Crop the 3D image
    return img[..., shallow:shallow + depth, top:top + height, left:left + width]


def resize_3d(img, resize, mode):
    # Resize the 3D image
    if isinstance(img, torch.Tensor):
        return torch.from_numpy(skitransform.resize(img.numpy(), resize, mode=mode))
    else:
        return skitransform.resize(img, resize, mode=mode)

# End of basic operations --------------------------------------------------------
# --------------------------------------------------------------------------------

# 3D transformation
# --------------------------------------------------------------------------------

# Single transformation ----------

class OnlyToTensor(torch.nn.Module):
    def __init__(self):
        super(OnlyToTensor, self).__init__()
        self.basetotensor = transforms.ToTensor()

    def forward(self, img):
        return self.basetotensor(img) * 255.
        

class CenterCropEdge(torch.nn.Module):
    """ 
    Crop edges of the input 
    """

    def __init__(self, crop_size):
        super(CenterCropEdge, self).__init__()
        self.crop_size = crop_size

    def forward(self, img):
        return img[..., self.crop_size : - self.crop_size]


class RandomCropEdge(torch.nn.Module):
    """ 
    Randomly crop edges of the input 
    """

    def __init__(self, crop_size):
        super(RandomCropEdge, self).__init__()
        self.crop_size = crop_size

    def forward(self, img):
        top, down = self.get_params(self.crop_size)
        return img[..., down : - top]

    @staticmethod
    def get_params(crop_size):
        top = random.randint(1, 2 * crop_size)
        down = 2 * crop_size - top
        return top, down


class RandomCrop3D(torch.nn.Module):
    """ 
    Randomly crop the input 3D image to target size 
    """

    def __init__(self, size):
        super(RandomCrop3D, self).__init__()
        self.size = np.array(size)

    def forward(self, img):
        params = self.get_params(img, self.size)
        return crop_3d(img, *params)

    @staticmethod
    def get_params(img, size):
        crop_out = img.shape[-len(size):] - size
        starts = [random.randint(0, _crop) for _crop in crop_out]
        ends = (starts + size).tolist()
        return starts + ends


class RandomResizedCrop3D(torch.nn.Module):
    """ 
    Randomly crop the input 3D image and resize to the target size 
    """

    def __init__(self, size, ratio: float, mode: str = "reflect"):
        super(RandomResizedCrop3D, self).__init__()
        self.size = size
        self.ratio = ratio
        self.mode = mode

    @staticmethod
    def get_params(img, size, ratio):
        crop_ratio = ratio * np.random.rand(len(size))
        after_crop_size = np.array(img.shape[-len(size):] * (1 - crop_ratio), dtype=np.int32)
        crop_out = img.shape[-len(size):] - after_crop_size
        starts = [random.randint(0, _crop) for _crop in crop_out]
        ends = (starts + after_crop_size).tolist()
        return starts + ends

    def forward(self, img):
        params = self.get_params(img, self.size, self.ratio)
        img = crop_3d(img, *params)
        img = resize_3d(img, self.size, self.mode)
        return img


class Resize3D(torch.nn.Module):
    """ 
    Resize the input 3D image to the target size 
    """
    
    def __init__(self, size, mode: str = "reflect"):
        super(Resize3D, self).__init__()
        self.size = size
        self.mode = mode

    def forward(self, img):
        return resize_3d(img, self.size, mode = self.mode)


class TensorUnsqueeze(torch.nn.Module):
    def __init__(self, dim: int = 0):
        super(TensorUnsqueeze, self).__init__()
        self.dim = dim

    def forward(self, data):
        return data.unsqueeze(self.dim)


class PaddingToSize(torch.nn.Module):
    """ 
    Pad the input to target size 
    """

    def __init__(self, target_size, mode: str = "minimum"):
        super(PaddingToSize, self).__init__()
        self.target_size = target_size
        self.mode = mode

    def forward(self, data):
        pad_size = np.clip(self.target_size - np.array(data.shape), 0, np.inf)
        data = np.pad(data, tuple([(0, int(_size)) for _size in pad_size]), mode=self.mode)
        return data


class AdaptiveScale(torch.nn.Module):
    """
    Rescale the input to an extreme without changing the original shape ratio or exceeding the target size 
    """

    def __init__(self, target_size):
        super(AdaptiveScale, self).__init__()
        self.target_size = target_size

    def forward(self, data):
        scale_coeffs = self.target_size / np.array(data.shape)
        data = skitransform.rescale(data, min(scale_coeffs))
        return data


# Paired transformation ----------

class PairedCompose(transforms.Compose):
    def __call__(self, img, label):
        for t in self.transforms:
            if isinstance(t, BasePairedTransform):
                img, label = t(img, label)
            else:
                img, label = t(img), t(label)

        return img, label


class BasePairedTransform(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
    
    @staticmethod
    def check_data(img, label):
        assert img.shape == label.shape, "Paired data have different shapes"

    @abc.abstractmethod
    def forward(self, img, label):
        """ 
        Transform inputs under the same randomness 
        """


class PairedRandomResizedCrop3D(BasePairedTransform, RandomResizedCrop3D):
    def __init__(self, size, ratio: float, mode: str = "reflect"):
        BasePairedTransform.__init__(self)
        RandomResizedCrop3D.__init__(self, size, ratio, mode)

    def forward(self, img, label):
        self.check_data(img, label)
        params = self.get_params(img, self.size, self.ratio)
        img, label = crop_3d(img, *params), crop_3d(label, *params)
        img, label = resize_3d(img, self.size, self.mode), resize_3d(label, self.size, self.mode)
        return img, label


class PairedRandomVerticalFlip(BasePairedTransform, transforms.RandomVerticalFlip):
    def __init__(self, p):
        BasePairedTransform.__init__(self)
        transforms.RandomVerticalFlip.__init__(self, p)

    def forward(self, img, label):
        self.check_data(img, label)
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairedRandomHorizontalFlip(BasePairedTransform, transforms.RandomHorizontalFlip):
    def __init__(self, p):
        BasePairedTransform.__init__(self)
        transforms.RandomHorizontalFlip.__init__(self, p)

    def forward(self, img, label):
        self.check_data(img, label)
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairedRandomIdentity(BasePairedTransform):
    def __init__(self, p):
        BasePairedTransform.__init__(self)
        self.p = p

    def forward(self, img, label):
        self.check_data(img, label)
        if random.random() < self.p:
            img = copy.deepcopy(label)
        return img, label


class PairedRandomCropEdge(BasePairedTransform, RandomCropEdge):
    def __init__(self, crop_size):
        BasePairedTransform.__init__(self)
        RandomCropEdge.__init__(self, crop_size)

    def forward(self, img, label):
        self.check_data(img, label)
        top, down = self.get_params(self.crop_size)
        return img[..., down : - top], label[..., down : - top]


class PairedRandomRotation(BasePairedTransform, transforms.RandomRotation):
    """ 
    Only support Tensor inputs at present 
    """
    
    def __init__(self, degrees, expand = False, center = None, resample = Image.BILINEAR):
        BasePairedTransform.__init__(self)
        transforms.RandomRotation.__init__(self, degrees, expand = expand, center = center, resample = resample)

    def forward(self, img, label):
        self.check_data(img, label)
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill),\
                F.rotate(label, angle, self.resample, self.expand, self.center, self.fill)

# Triple transformation ----------

class BaseTripleTransform(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
    
    @staticmethod
    def check_data(img, label, reference):
        assert img.shape == label.shape == reference.shape, "Triple data have different shapes"

    @abc.abstractmethod
    def forward(self, img, label, reference):
        """ Transform inputs under the same  randomness """

class TripleCompose(transforms.Compose):
    def __call__(self, img, label, reference):
        for t in self.transforms:
            if isinstance(t, BaseTripleTransform):
                img, label, reference = t(img, label, reference)
            else:
                img, label, reference = t(img), t(label), t(reference)

        return img, label, reference


class TripleRandomResizedCrop3D(BaseTripleTransform, RandomResizedCrop3D):
    def __init__(self, size, ratio: float, mode: str = "reflect"):
        BaseTripleTransform.__init__(self)
        RandomResizedCrop3D.__init__(self, size, ratio, mode)

    def forward(self, img, label, reference):
        self.check_data(img, label, reference)
        params = self.get_params(img, self.size, self.ratio)
        img = resize_3d(crop_3d(img, *params), self.size, self.mode)
        label = resize_3d(crop_3d(label, *params), self.size, self.mode)
        reference = resize_3d(crop_3d(reference, *params), self.size, self.mode)
        return img, label, reference


class TripleRandomVerticalFlip(BaseTripleTransform, transforms.RandomVerticalFlip):
    def __init__(self, p):
        BaseTripleTransform.__init__(self)
        transforms.RandomVerticalFlip.__init__(self, p)

    def forward(self, img, label, reference):
        self.check_data(img, label, reference)
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(label), F.vflip(reference)
        return img, label, reference


class TripleRandomHorizontalFlip(BaseTripleTransform, transforms.RandomHorizontalFlip):
    def __init__(self, p):
        BaseTripleTransform.__init__(self)
        transforms.RandomHorizontalFlip.__init__(self, p)

    def forward(self, img, label, reference):
        self.check_data(img, label, reference)
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(label), F.hflip(reference)
        return img, label, reference


class TripleRandomIdentity(BaseTripleTransform):
    def __init__(self, p):
        BaseTripleTransform.__init__(self)
        self.p = p

    def forward(self, img, label, reference):
        self.check_data(img, label, reference)
        if random.random() < self.p:
            img = label
        return img, label, reference


class TripleRandomCropEdge(BaseTripleTransform, RandomCropEdge):
    def __init__(self, crop_size):
        BaseTripleTransform.__init__(self)
        RandomCropEdge.__init__(self, crop_size)

    def forward(self, img, label, reference):
        self.check_data(img, label, reference)
        top, down = self.get_params(self.crop_size)
        return img[..., down : - top], label[..., down : - top], reference[..., down : - top]


class TripleRandomRotation(BaseTripleTransform, transforms.RandomRotation):
    """ 
    Only support Tensor inputs at present 
    """

    def __init__(self, degrees, expand = False, center = None, resample = Image.BILINEAR):
        BaseTripleTransform.__init__(self)
        transforms.RandomRotation.__init__(self, degrees, expand = expand, center = center, resample = resample)

    def forward(self, img, label, reference):
        self.check_data(img, label, reference)
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill),\
                F.rotate(label, angle, self.resample, self.expand, self.center, self.fill),\
                F.rotate(reference, angle, self.resample, self.expand, self.center, self.fill)

# End of 3D transformation -------------------------------------------------------
# --------------------------------------------------------------------------------
