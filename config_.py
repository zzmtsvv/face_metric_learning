from dataclasses import dataclass
import torch


@dataclass 
class config:
    loss = "arcface_loss"  # три опции: 'arcface_loss' or 'modified_bce' or 'cosine_embedding_loss'
    dataset_directory = "lfw"  # нужно поменять при необходимости
    random_seed = 42
    best_weights_path = f"{loss}_best_model.pth"
    train_directory = None
    test_directory = None
    image_height = 224
    image_width = 224
    batch_size = 16
    num_epochs = 10
    lr = 3e-4
    max_grad_norm = 1.0
    weight_decay = 1e-3
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    validation_data_ratio = 0.2

