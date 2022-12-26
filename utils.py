import os
import random
import numpy as np
import torch
from config_ import config


def upload_paths():
    images = []
    for _, _, files in os.walk(config.dataset_directory):
        for file in files:
            if file.endswith('.jpg'):
                images.append(file)
    random.shuffle(images)
    return images


def seed_everything(seed: int = config.random_seed) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
