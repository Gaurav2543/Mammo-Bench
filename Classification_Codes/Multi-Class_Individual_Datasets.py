import os
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, classification_report, confusion_matrix

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

# Modify the cache directory
CACHE_DIR = 'cache'
os.environ['TORCH_HOME'] = CACHE_DIR

path = 'models'
if not os.path.exists(path):
    os.makedirs(path)

class Mammogram_Data(Dataset):
    def __init__(self, dataset, image_path_col, target_size=(224, 224)):
        self.dataset = dataset
        self.image_path_col = image_path_col
        self.target_size = target_size
        self.label_map = self._create_label_map()
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats for RGB
        ])
        self.length = len(dataset)

    def _create_label_map(self):
        return {'Normal': 0, 'Benign': 1, 'Malignant': 2}
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):  
        dataset_idx = idx
        row = self.dataset.iloc[dataset_idx]
        image_path = row[self.image_path_col]
        label = row['classification']
        label_idx = self.label_map[label]
        image = self._load_image(image_path)
        image = self.base_transform(image)
        return image, label_idx

    def _load_image(self, image_path):
        image_path = '/data/Mammo-Bench/' + image_path
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        # if the image path contains dmid, then do not normalize the image
        if 'dmid' in image_path:
            image = image.astype(np.float32)
        else:
            image = image.astype(np.float32) / 255.0
        # Stack the grayscale image to create 3 channels
        image = np.stack([image, image, image], axis=-1)
        return image

def get_data_loader(data, batch_size=32, shuffle=True, pin_memory=True):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

class MammogramClassifier(pl.LightningModule):
    def __init__(self, num_classes=3, model_name='resnet101', learning_rate=5e-5, weight_decay=5e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.val_outputs = []
        self.test_outputs = []
        self.best_loss = float('inf')
        self.class_names = ['Normal', 'Benign', 'Malignant']  

    def create_model(self):
        if self.hparams.model_name == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(2048, self.hparams.num_classes)
        return model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.val_outputs.append({'val_loss': loss, 'preds': preds, 'targets': y})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.val_outputs]).cpu()
        targets = torch.cat([x['targets'] for x in self.val_outputs]).cpu()
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        self.log('val_f1', f1, sync_dist=True)
        self.val_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        self.test_outputs.append({'test_loss': loss, 'preds': preds, 'targets': y})
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu()
        targets = torch.cat([x['targets'] for x in self.test_outputs]).cpu()
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average='weighted', zero_division=0)
        precision = precision_score(targets, preds, average='weighted', zero_division=0)
        recall = recall_score(targets, preds, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(targets, preds)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
        self.log('test_f1', f1, sync_dist=True)
        self.log('test_precision', precision, sync_dist=True)
        self.log('test_recall', recall, sync_dist=True)
        self.log('test_mcc', mcc, sync_dist=True)
        self.print_classification_report(targets, preds)
        self.generate_confusion_matrix(targets, preds)
        
        self.test_outputs.clear()

    def print_classification_report(self, targets, preds):
        report = classification_report(targets, preds, zero_division=0)
        print("\nClassification Report:")
        print(report)

    def generate_confusion_matrix(self, targets, preds):
        cm = confusion_matrix(targets, preds)
        print("\nConfusion Matrix:")
        print(cm)

class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(
            disable=True,  # disable the progress bar for validation
        )
        return bar

def main():
    # Print GPU information
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set float32 matrix multiplication precision to 'medium' or 'high'
    torch.set_float32_matmul_precision('medium')
    datasets = ['inbreast', 'mini-ddsm', 'cdd-cesm', 'dmid', 'vindr-mammo']

    for data in datasets:
            # Print GPU information
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Set float32 matrix multiplication precision to 'medium' or 'high'
        torch.set_float32_matmul_precision('medium')

        # Load and prepare your data
        print("Loading and preparing data...")
        dataset = pd.read_csv(f'CSV_Files/mammo-bench.csv', low_memory=False)

        # exclude the Suspicious Malignant because of its uncertainty and small sample size
        dataset = dataset[dataset['classification'] != 'Suspicious Malignant']

        # exclude the BIRADS 0.0 because they are not normal, they require additional tests to determine if they are normal or abnormal
        dataset = dataset[dataset['BIRADS'] != 0.0]
        dataset = dataset[dataset['source_dataset'] == data]

        train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.75, random_state=42)

        train_dataset = Mammogram_Data(train_data, 'preprocessed_image_path')
        val_dataset = Mammogram_Data(val_data, 'preprocessed_image_path')
        test_dataset = Mammogram_Data(test_data, 'preprocessed_image_path')

        # print the class distribution
        print("Train dataset class distribution:")
        print(train_data['classification'].value_counts())
        print("\nValidation dataset class distribution:")
        print(val_data['classification'].value_counts())
        print("\nTest dataset class distribution:")
        print(test_data['classification'].value_counts())

        # Reduce batch size
        batch_size = 156
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)

        # Initialize the model
        model = MammogramClassifier(learning_rate=5e-6, weight_decay=1e-2, model_name='resnet101', num_classes=3)

        # Set up checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=path,
            filename='Mammo-Bench_Multi_Class-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )

        # Set up progress bar
        progress_bar = LitProgressBar()

        # Set the strategy based on the number of GPUs
        if num_gpus > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
        else:
            strategy = 'auto'  # Use 'auto' for single GPU or CPU

        trainer = pl.Trainer(
            max_epochs=50,
            accelerator='gpu',
            devices='auto',
            strategy=strategy,
            callbacks=[checkpoint_callback, progress_bar],
            log_every_n_steps=10,
            accumulate_grad_batches=2,  # Gradient accumulation
            precision='16-mixed'  # Use half precision
        )

        print("Starting model training...")
        trainer.fit(model, train_loader, val_loader)
        print("Model training completed")

        print("Loading best model...")
        best_model_path = checkpoint_callback.best_model_path
        model = MammogramClassifier.load_from_checkpoint(best_model_path)
        print("Testing model...")
        trainer.test(model, test_loader)
        print("Model testing completed")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
