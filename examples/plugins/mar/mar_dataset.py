import os
import yaml
import abc
import random
import h5py

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from aw_nas.dataset.base import BaseDataset
from aw_nas import Component, utils
from aw_nas.utils import RegistryMeta

import sys
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH)
from mar_transforms import *


# Data loader
# --------------------------------------------------------------------------------

def mat_loader(path):
    return scio.loadmat(path)


def mat_loader_h5py(path):
    data = h5py.File(path, "r")["images"]
    return data["Input"], data["Label"]
    
# End of data loader -------------------------------------------------------------
# --------------------------------------------------------------------------------


# 2D dataset
# --------------------------------------------------------------------------------

class TwoDimensionArtificialDataset(Dataset):
    """
    This dataset is used to load the "MAR-2D-PRE" dataset simulated by 
    "Metal artifact reduction for practical dental computed tomography by improving interpolation-based reconstruction with deep learning"
    [https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13644]
    """

    def __init__(self, root, transform = None, target_transform = None, loader = mat_loader_h5py):
        super(TwoDimensionArtificialDataset, self).__init__()
        self.root = root
        self.fileList = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        # Load the data
        image, label = self.loader(os.path.join(self.root, self.fileList[index]))
        image = 20 * np.array(image).astype(np.float32).T[:, :, np.newaxis]
        label = 20 * np.array(label).astype(np.float32).T[:, :, np.newaxis]
        # Tranform the data
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.fileList)


class TwoDimensionClinicalDataset(Dataset):
    """
    This dataset is used to load the clinical dataset of
    "Metal artifact reduction for practical dental computed tomography by improving interpolation-based reconstruction with deep learning"
    [https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.13644]
    """

    def __init__(self, root, transform = None, target_transform = None, loader = mat_loader):
        super(TwoDimensionClinicalDataset, self).__init__()
        self.root = root
        self.fileList = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        # Load the data
        data = self.loader(os.path.join(self.root, self.fileList[index]))
        image, label = data["images"][0][0]
        image = 20 * image[:, :, np.newaxis].astype(np.float32)
        label = 20 * label[:, :, np.newaxis].astype(np.float32)
        # Tranform the data
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.fileList)


class TwoDimensionThreeDimensionArtificialDataset(Dataset):
    """ 
    This dataset is used to load the MAR-2D-SIM dataset generated from MAR-3D-SIM with our proposed "Threshold-Discriminant-Method" 
    """
    
    def __init__(self, root, transform = None, target_transform = None, loader = mat_loader):
        super(TwoDimensionThreeDimensionArtificialDataset, self).__init__()
        self.root = root
        self.fileList = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        # Load the data
        data = self.loader(os.path.join(self.root, self.fileList[index]))
        image = 20 * data["ImgMA"].astype(np.float32)[:, :, np.newaxis]
        label = 20 * data["ImgFreeRecon"].astype(np.float32)[:, :, np.newaxis]
        # Tranform the data
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

    def __len__(self):
        return len(self.fileList)


class TwoDimensionThreeDimensionClinicalDataset(TwoDimensionThreeDimensionArtificialDataset):
    """ 
    This dataset is used to load the MAR-2D-CLINIC dataset generated from MAR-3D-CLINIC with our proposed "Threshold-Discriminant-Method" 
    """
    
    def __getitem__(self, index):
        # Load the data
        data = self.loader(os.path.join(self.root, self.fileList[index]))
        image = 20 * data["ImgMA"].astype(np.float32)[:, :, np.newaxis]
        label = 20 * data["ImgFree"].astype(np.float32)[:, :, np.newaxis] 
        # Tranform the data
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label


class TwoDimensionMAR(BaseDataset):
    """
    Base dataloader of two-dimensional data.
    Args:
        * target_size: [list] size to resize
        * flip_p: [float] probability to flip the data horizontally
        * identity_p: [float] probability to assign the label as input
        * degrees: [float] maximum degree to randomly rotate
    """

    def __init__(self, Dataset, target_size: list = [320, 320], flip_p: float = 0.5, identity_p: float = 0., degrees: int = 0):
        super(TwoDimensionMAR, self).__init__()
        train_transform = PairedCompose([
            transforms.ToTensor(),
            transforms.Resize(target_size),
            PairedRandomRotation(degrees = degrees),
            PairedRandomHorizontalFlip(p = flip_p),
            PairedRandomIdentity(p = identity_p)
            ])
        
        test_transform = PairedCompose([
            transforms.ToTensor(),
            transforms.Resize(target_size),
            ])
        
        self.datasets = {}
        self.datasets["train"] = Dataset(root = os.path.join(self.data_dir, "train"), transform = train_transform)
        self.datasets["test"] = Dataset(root = os.path.join(self.data_dir, "test"), transform = test_transform)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"


class TwoDimensionArtificialMAR(TwoDimensionMAR):
    """ 
    Dataloader of the "MAR-2D-PRE" dataset 
    """
    
    NAME = "two_dimension_artificial_mar"
       
    def __init__(self, **kwargs):
        super(TwoDimensionArtificialMAR, self).__init__(TwoDimensionArtificialDataset, **kwargs)
     

class TwoDimensionClinicalMAR(TwoDimensionMAR):
    """ 
    Dataloader of the "TwoDimensionClinicalDataset" 
    """
    
    NAME = "two_dimension_clinical_mar"

    def __init__(self, **kwargs):
        super(TwoDimensionClinicalMAR, self).__init__(TwoDimensionClinicalDataset, **kwargs)


class TwoDimensionThreeDimensionArtificialMAR(TwoDimensionMAR):
    """ 
    Dataloader of the "MAR-2D-SIM" dataset 
    """
    
    NAME = "two_dimension_three_dimension_artificial_mar"

    def __init__(self, **kwargs):
        super(TwoDimensionThreeDimensionArtificialMAR, self).__init__(TwoDimensionThreeDimensionArtificialDataset, **kwargs)
 

class TwoDimensionThreeDimensionClinicalMAR(TwoDimensionMAR):
    """ 
    Dataloader of the "MAR-2D-CLINIC" dataset 
    """
    
    NAME = "two_dimension_three_dimension_clinical_mar"

    def __init__(self, **kwargs):
        super(TwoDimensionThreeDimensionClinicalMAR, self).__init__(TwoDimensionThreeDimensionClinicalDataset, **kwargs)

# End of 2D dataset -----------------------------------------------------------
# -----------------------------------------------------------------------------


# 3D dataset
# -----------------------------------------------------------------------------

class ThreeDimensionDataset(Dataset):
    """
    Three-dimensional dataset for "MAR-3D-SIM" and "MAR-3D-CLINIC"
    
    Args:
        * clip: [bool] whether clip the data to [0, 1]
    """
    
    def __init__(self, root: str, transform = None, target_transform = None, loader = mat_loader, clip = False):
        super(ThreeDimensionDataset, self).__init__()
        self.root = root
        self.fileList = os.listdir(root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.clip = clip

    def __len__(self):
        return len(self.fileList)

    def __getitem__(self, index):
        # Load the data
        data = self.loader(os.path.join(self.root, self.fileList[index]))
        image = data["ImgMA"].astype(np.float32)
        label = data["ImgFree"].astype(np.float32)
        # Clip the data
        image = np.clip(20. * image, 0, 1 if self.clip else np.inf)
        label = np.clip(20. * label, 0, 1 if self.clip else np.inf)
        # Transform the data
        if self.transform is not None:
            image, label = self.transform(img = image, label = label)
        return image, label


class ThreeDimensionMAR(BaseDataset):
    """
    Base dataloader of three-dimensional data.
    Args:
        * target_size: [list] size to resize
        * edge_crop: [int] half number of useless slices to crop
        * train_random_crop: [bool] whether randomly crop the useless slices
        * flip_p: [float] probability to flip the data horizontally
        * identity_p: [float] probability to assign the label as input
        * degrees: [float] maximum degree to randomly rotate
        * clip: [bool] whether clip the data to [0, 1]
    """

    def __init__(self, target_size: list = [320, 320, 64], train_random_crop: bool = False, clip: bool = False, 
            edge_crop: int = 10, flip_p: float = 0.5, identity_p: float = 0., degrees: int = 0):
        super(ThreeDimensionMAR, self).__init__()
        train_transform = PairedCompose([
            PairedRandomCropEdge(crop_size = edge_crop) if train_random_crop \
                    else CenterCropEdge(crop_size = edge_crop),
            Resize3D(target_size),
            transforms.ToTensor(),
            PairedRandomRotation(degrees = degrees),
            PairedRandomHorizontalFlip(p = flip_p),
            PairedRandomIdentity(p = identity_p),
            TensorUnsqueeze(dim = 0)
            ])
        
        test_transform = PairedCompose([
            CenterCropEdge(crop_size = edge_crop),
            Resize3D(target_size),
            transforms.ToTensor(),
            TensorUnsqueeze(dim = 0)
            ])
        
        self.datasets = {}
        self.datasets["train"] = ThreeDimensionDataset(root = os.path.join(self.data_dir, "train"), transform = train_transform, clip = clip)
        self.datasets["test"] = ThreeDimensionDataset(root = os.path.join(self.data_dir, "test"), transform = test_transform, clip = clip)
    
    def splits(self):
        return self.datasets
    
    @classmethod
    def data_type(cls):
        return "image"


class ThreeDimensionClinicalMAR(ThreeDimensionMAR):
    """ 
    Dataloader of "MAR-3D-CLINIC" 
    """

    NAME = "three_dimension_clinical_mar"
       

class ThreeDimensionArtificialMAR(ThreeDimensionMAR):
    """ 
    Dataloader of "MAR-3D-SIM" 
    """

    NAME = "three_dimension_artificial_mar"
    
# End of 3D dataset -----------------------------------------------------------
# -----------------------------------------------------------------------------


# 3D dataset to test the discriminator statically
# -----------------------------------------------------------------------------

class ThreeDimensionDiscriminateDataset(ThreeDimensionDataset):
    """
    Three-dimension dataset to train the discriminator statically
    The corresponding label of the metal-artifact image is 0
    And the corresponding label of the metal-free image is 1
    """

    def __getitem__(self, index):
        data = self.loader(os.path.join(self.root, self.fileList[index // 2]))
        # Generate the label
        label = index % 2
        # Load the input
        image = data["ImgFree" if label else "ImgMA"].astype(np.float32)
        image = np.clip(20. * image, 0, 1 if self.clip else np.inf)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.fileList) * 2


class ThreeDimensionDiscriminateMAR(ThreeDimensionMAR):
    """
    Base dataloader of three-dimensional data for training the discriminator statically
    Args:
        * target_size: [list] size to resize
        * edge_crop: [int] half number of useless slices to crop
        * train_random_crop: [bool] whether randomly crop the useless slices
        * flip_p: [float] probability to flip the data horizontally
        * identity_p: [float] probability to assign the label as input
        * degrees: [float] maximum degree to randomly rotate
        * clip: [bool] whether clip the data to [0, 1]
    """

    def __init__(self, target_size: list = [320, 320, 64], train_random_crop: bool = False, 
            clip: bool = False, edge_crop: int = 10, flip_p: float = 0.5, degrees: int = 0):
        super(ThreeDimensionDiscriminateMAR, self).__init__()
        train_transform = transforms.Compose([
            RandomCropEdge(crop_size = edge_crop) if train_random_crop \
                    else CenterCropEdge(crop_size = edge_crop),
            Resize3D(target_size),
            transforms.ToTensor(),
            transforms.RandomRotation(degrees = degrees),
            transforms.RandomHorizontalFlip(p = flip_p),
            TensorUnsqueeze(dim = 0)
            ])
        
        test_transform = transforms.Compose([
            CenterCropEdge(crop_size = edge_crop),
            Resize3D(target_size),
            transforms.ToTensor(),
            TensorUnsqueeze(dim = 0)
            ])

        self.datasets = {}
        self.datasets["train"] = ThreeDimensionDiscriminateDataset(root = os.path.join(self.data_dir, "train"), transform = train_transform, clip = clip)
        self.datasets["test"] = ThreeDimensionDiscriminateDataset(root = os.path.join(self.data_dir, "test"), transform = test_transform, clip = clip)


class ThreeDimensionDiscriminateArtificialMAR(ThreeDimensionDiscriminateMAR):
    """ 
    Test the disciminator on "MAR-3D-SIM" dataset 
    """

    NAME = "three_dimension_discriminate_artificial_mar"
   

class ThreeDimensionDiscriminateClinicalMAR(ThreeDimensionDiscriminateMAR):
    """ 
    Test the disciminator on "MAR-3D-CLINIC" dataset 
    """
    
    NAME = "three_dimension_discriminate_clinical_mar"

# End of 3D dataset to test the discriminator statically ----------------------
# -----------------------------------------------------------------------------


def test_two_dimension_artificial_mar():
    cfg = {
            "dataset_type": "two_dimension_artificial_mar",
            "dataset_cfg": {
                "target_size": [320, 320],
                "flip_p": 0.5,
                "identity_p": 0.}
            }
    type_ = cfg["dataset_type"]
    cfg = cfg.get("dataset_cfg", None)

    cls = RegistryMeta.get_class("dataset", type_)
    dataset = cls(**cfg)
    data_queue = torch.utils.data.DataLoader(dataset.splits()["test"], 
                                             batch_size = 1, num_workers = 2)
    l1, l2 = 0, 0
    for images, labels in data_queue:
        l1_distance = torch.sum(torch.abs(images - labels))
        l2_distance = (images - labels).norm(2)
        l1 += l1_distance.item()
        l2 += l2_distance.item()
    l1, l2 = l1 / len(data_queue), l2 / len(data_queue)
    print(l1, l2)


def test_two_dimension_three_dimension_artificial_mar():
    cfg = {
            "dataset_type": "two_dimension_three_dimension_artificial_mar",
            "dataset_cfg": {
                "target_size": [320, 320],
                "flip_p": 0.5,
                "degrees": 0.,
                "identity_p": 0.}
            }
    type_ = cfg["dataset_type"]
    cfg = cfg.get("dataset_cfg", None)

    cls = RegistryMeta.get_class("dataset", type_)
    dataset = cls(**cfg)
    data_queue = torch.utils.data.DataLoader(dataset.splits()["train"], 
                                             batch_size = 1, num_workers = 2)
    print(len(data_queue))


def test_two_dimension_three_dimension_clinical_mar():
    cfg = {
            "dataset_type": "two_dimension_three_dimension_clinical_mar",
            "dataset_cfg": {
                "target_size": [320, 320],
                "flip_p": 0.5,
                "degrees": 0.,
                "identity_p": 0.}
            }
    type_ = cfg["dataset_type"]
    cfg = cfg.get("dataset_cfg", None)

    cls = RegistryMeta.get_class("dataset", type_)
    dataset = cls(**cfg)
    data_queue = torch.utils.data.DataLoader(dataset.splits()["train"], 
                                             batch_size = 100, num_workers = 2)
    cnt = 0
    for images, labels in data_queue:
        cnt += abs(images - labels).sum().item()
    print(cnt / len(data_queue))


def test_two_dimension_clinical_mar():
    cfg = {
            "dataset_type": "two_dimension_clinical_mar",
            "dataset_cfg": {
                "target_size": [320, 320],
                "flip_p": 0.5,
                "identity_p": 0.}
            }
    type_ = cfg["dataset_type"]
    cfg = cfg.get("dataset_cfg", None)

    cls = RegistryMeta.get_class("dataset", type_)
    dataset = cls(**cfg)
    data_queue = torch.utils.data.DataLoader(dataset.splits()["train"], 
                                             batch_size = 1, num_workers = 2)
    for images, labels in data_queue:
        print(images.shape, labels.shape)


def test_three_dimension_artificial_mar():
    cfg = {
            "dataset_type": "three_dimension_artificial_mar",
            "dataset_cfg": {
                "train_random_crop": False,
                "target_size": [320, 320, 64],
                "flip_p": 0.0,
                "identity_p": 0.,
                "clip": False,
                "degrees": 15
                }
            }
    
    type_ = cfg["dataset_type"]
    cfg = cfg.get("dataset_cfg", None)

    cls = RegistryMeta.get_class("dataset", type_)
    dataset = cls(**cfg)
    data_queue = torch.utils.data.DataLoader(dataset.splits()["train"], 
                                             batch_size = 1, num_workers = 2)
    for images, labels in data_queue:
        print(abs(images - labels).sum().item())


def test_three_dimension_clinical_mar():
    cfg = {
            "dataset_type": "three_dimension_clinical_mar",
            "dataset_cfg": {
                "train_random_crop": False,
                "target_size": [320, 320, 64],
                "flip_p": 0.0,
                "identity_p": 0.,
                "clip": False,
                "degrees": 15
                }
            }
    
    type_ = cfg["dataset_type"]
    cfg = cfg.get("dataset_cfg", None)

    cls = RegistryMeta.get_class("dataset", type_)
    dataset = cls(**cfg)
    data_queue = torch.utils.data.DataLoader(dataset.splits()["test"], 
                                             batch_size = 1, num_workers = 2)
    for images, labels in data_queue:
        slice_max = [_slice.max().item() for _slice in images[0][0]]
        print(slice_max)


if __name__ == "__main__":
    test_two_dimension_three_dimension_artificial_mar()
