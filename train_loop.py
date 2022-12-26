import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import upload_paths
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from config_ import config
from models import Baseline
from datasets import ContrastiveFaceDataset
from models import ArcFaceLoss, ContrastiveSigmoidWrapper
from tqdm.auto import tqdm
from torch.amp import autocast


class ROC_AUC(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        with torch.inference_mode():
            pred = torch.sigmoid(x).round().detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        return roc_auc_score(y, pred)


class F1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, y):
        with torch.inference_mode():
            pred = torch.sigmoid(x).round().detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        return f1_score(y, pred)


class Trainer:
    losses = {
        "arcface_loss": ArcFaceLoss,
        "modified_bce": ContrastiveSigmoidWrapper,
        "cosine_embedding_loss": nn.CosineEmbeddingLoss
    }

    def __init__(self) -> None:

        self.f1_score = F1()
        self.best_f1 = 0

        self.model = Baseline()
        self.model.to(config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                            lr=config.lr, weight_decay=config.weight_decay)
        self.criterion = self.losses[config.loss]()

        dataset = upload_paths()
        train_data, val_data = train_test_split(dataset, test_size=config.validation_data_ratio)
        train_dataset = ContrastiveFaceDataset(img_paths=train_data, train=True)
        val_dataset = ContrastiveFaceDataset(img_paths=val_data, train=False)

        self.train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    def measure_metrics(self, outputs, labels):
        return self.f1_score(outputs, labels)

    def fit(self):
        print(f"training starts on {config.device_str}")
        self.model.to(config.device)

        for epoch in range(1, config.num_epochs + 1):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

    def train_epoch(self, epoch):
        self.model.train()

        pbar = tqdm(
            enumerate(self.train_dataloader), 
            total = len(self.train_dataloader),
            desc = f"Epoch(train) {epoch} ")
        running_loss = 0

        for idx, (img1, img2, label) in pbar:
            img1 = img1.to(config.device)
            img2 = img2.to(config.device)
            label = label.to(config.device)

            self.optimizer.zero_grad()
            
            with autocast(device_type=config.device_str):
                embedding1 = self.model(img1)
                embedding2 = self.model(img2)
                loss = self.criterion(embedding1, embedding2, label.float())

            running_loss += loss.item()

            pbar.set_postfix(
                dict(loss = round(running_loss / (idx + 1), 5)))

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
            self.optimizer.step()


    def validate_epoch(self, epoch):
        self.model.eval()

        pbar = tqdm(
            enumerate(self.val_dataloader), 
            total = len(self.val_dataloader),
            desc = f"Epoch(validation) {epoch} ")
        
        running_loss = 0
        running_f1 = 0

        for idx, (img1, img2, label) in pbar:
            img1 = img1.to(config.device)
            img2 = img2.to(config.device)
            label = label.to(config.device)

            with torch.no_grad():
                emb1 = self.model(img1)
                emb2 = self.model(img2)
                pred = (emb1 - emb2).pow(2).sum(1)
            
            running_loss += self.criterion(emb1, emb2, label.float()).item()
            
            f1 = self.measure_metrics(pred, label)
            running_f1 += f1

            pbar.set_postfix(
                dict(
                    f1 = round(running_f1 / (idx + 1), 5),
                    loss = round(running_loss/(idx + 1), 5)
                )
            )

            if running_f1 / (idx + 1) > self.best_f1:
                self.best_f1 = running_f1 / (idx + 1)
                torch.save(self.model.state_dict(), config.best_weights_path)
                print(f"saved model weights at: {config.best_weights_path}")