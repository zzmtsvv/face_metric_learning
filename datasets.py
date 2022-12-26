import os
import albumentations as A
from torch.utils.data import Dataset
import cv2
import random
import torch
import numpy as np
from torch.nn import functional as F
from config_ import config


class ContrastiveFaceDataset(Dataset):
    def __init__(self, img_paths=None, img_dir=config.dataset_directory, train=True, transforms=None) -> None:
        super().__init__()

        self.images = img_paths
        self.is_train = train
        self.img_dir = img_dir
        self.transforms = transforms
        if transforms is None:
            self.transforms = A.Compose([
                A.Resize(224, 224),
                A.Normalize()
            ])
        self.augmentations = self.make_augmentations() if train else None

    def make_augmentations(self) -> A.Compose:
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.3
            )
        ])
    
    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def check_similarity(path1, path2):
        # костыль чтобы чекнуть одинаковые ли имена
        if path1[:-9] == path2[:-9]:
            return 1
        return 0
    
    def get_full_image_path(self, relative_path):
        return os.path.join(self.img_dir, relative_path[:-9], relative_path)
    
    def __getitem__(self, index):
        init_path = self.images[index]
        second_path = random.choice(self.images[:index] + self.images[index + 1:])
        label = self.check_similarity(init_path, second_path)

        if config.loss == "cosine_embedding_loss":
            label = 2 * label - 1 # перевести 0/1 в -1/1
        
        image1 = self.transforms(image=cv2.imread(self.get_full_image_path(init_path)))["image"]
        image2 = self.transforms(image=cv2.imread(self.get_full_image_path(second_path)))["image"]

        if self.augmentations is not None:
            image1 = self.augmentations(image=image1)["image"]
            image2 = self.augmentations(image=image2)["image"]

        return image1.transpose(2, 0, 1) / 255.0, image2.transpose(2, 0, 1) / 255.0, label


class SoftContrastiveFaceDataset(ContrastiveFaceDataset):
    def __init__(self, img_paths=None, img_dir=config.dataset_directory, train=True, transforms=None) -> None:
        super().__init__(img_paths, img_dir, train, transforms)
    
    def make_augmentations(self) -> A.Compose:
        return super().make_augmentations()
    
    def __len__(self):
        return super().__len__()
    
    @staticmethod
    def calc_similarity(img1: np.ndarray, img2: np.ndarray):
        x1 = torch.from_numpy(img1).flatten(start_dim=1).float()
        x2 = torch.from_numpy(img2).flatten(start_dim=1).float()

        # лежит в промежутке от 0 до 1
        phi = ((F.cosine_similarity(x1, x2) + 1.0) / 2.0).float()

        return phi
    
    def get_full_image_path(self, relative_path):
        return super().get_full_image_path(relative_path)
    
    def __getitem__(self, index):
        init_path = self.images[index]
        second_path = random.choice(self.images[:index] + self.images[index + 1:])
        
        image1 = self.transforms(image=cv2.imread(self.get_full_image_path(init_path)))["image"]
        image2 = self.transforms(image=cv2.imread(self.get_full_image_path(second_path)))["image"]

        soft_target = self.calc_similarity(image1, image2)

        if self.augmentations is not None:
            image1 = self.augmentations(image=image1)["image"]
            image2 = self.augmentations(image=image2)["image"]

        return image1.transpose(2, 0, 1), image2.transpose(2, 0, 1), soft_target
