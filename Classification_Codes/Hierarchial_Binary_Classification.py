
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, classification_report, confusion_matrix

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set paths
CACHE_DIR = 'cache'
os.environ['TORCH_HOME'] = CACHE_DIR

BATCH_SIZE1 = 128  
BATCH_SIZE2 = 16  

# Dataset Class
class MammogramData(Dataset):
    def __init__(self, dataset, image_path_col, task='stage1', target_size=(224, 224)):
        if task == 'stage1':
            self.dataset = dataset.copy()
            self.dataset['classification'] = self.dataset['classification'].apply(lambda x: 0 if x == 'Normal' else 1)
        elif task == 'stage2':
            self.dataset = dataset[dataset['classification'].isin(['Benign', 'Malignant'])].copy()
            self.dataset['classification'] = self.dataset['classification'].apply(lambda x: 0 if x == 'Benign' else 1)
        
        self.image_path_col = image_path_col
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.target_size),
            transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats for RGB
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        image_path = '/data/Mammo-Bench/' + row[self.image_path_col]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        if 'dmid' not in image_path:
            image = image.astype(np.float32) / 255.0
        image = np.stack([image, image, image], axis=-1)
        image = self.transform(image)
        return image, row['classification']

class MammogramClassifier(pl.LightningModule):    
    def __init__(self, num_classes=2, learning_rate=1e-6, weight_decay=1e-2):
        super(MammogramClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = models.resnet101(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(2048, 1)
        self.criterion = nn.BCEWithLogitsLoss()
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        # Convert labels to float and ensure correct shape
        labels = labels.float()
        loss = self.criterion(outputs, labels)
        # For accuracy calculation, we need to apply sigmoid and round
        preds = torch.round(torch.sigmoid(outputs))
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_accuracy', acc, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        # Convert labels to float and ensure correct shape
        labels = labels.float()
        loss = self.criterion(outputs, labels)
        # For accuracy calculation, we need to apply sigmoid and round
        preds = torch.round(torch.sigmoid(outputs))
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', acc, prog_bar=True, sync_dist=True)
        
        # Store predictions and targets for epoch end calculations
        self.val_outputs.append({'preds': preds.detach(), 'targets': labels.detach()})
        return loss

    def on_validation_epoch_end(self):
        # Check if there are any validation outputs
        if not self.val_outputs:
            return
            
        # Aggregate predictions and targets from validation outputs
        preds = torch.cat([x['preds'] for x in self.val_outputs])
        targets = torch.cat([x['targets'] for x in self.val_outputs])
        
        # Calculate validation accuracy
        val_acc = (preds == targets).float().mean()
        
        # Print validation accuracy if not in first epoch
        if self.current_epoch > 0:
            print(f"\nValidation Accuracy: {val_acc:.4f}")

        # Clear outputs after aggregation to prepare for the next epoch
        self.val_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        # For predictions with BCEWithLogitsLoss, we need sigmoid and round
        preds = torch.round(torch.sigmoid(outputs))
        
        # Store outputs for aggregation at the end of testing
        self.test_outputs.append({'preds': preds, 'targets': labels})

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs])
        targets = torch.cat([x['targets'] for x in self.test_outputs])
        
        # Calculate metrics
        accuracy = accuracy_score(targets.cpu(), preds.cpu())
        f1 = f1_score(targets.cpu(), preds.cpu(), average='weighted')
        precision = precision_score(targets.cpu(), preds.cpu(), average='weighted')
        recall = recall_score(targets.cpu(), preds.cpu(), average='weighted')
        mcc = matthews_corrcoef(targets.cpu(), preds.cpu())
        report = classification_report(targets.cpu(), preds.cpu())
        cm = confusion_matrix(targets.cpu(), preds.cpu())
        
        print(f"\n{'Metric':<15}{'Score':<10}")
        print(f"{'-'*25}")
        print(f"{'Accuracy':<15}{accuracy:.4f}")
        print(f"{'F1 Score':<15}{f1:.4f}")
        print(f"{'Precision':<15}{precision:.4f}")
        print(f"{'Recall':<15}{recall:.4f}")
        print(f"{'MCC':<15}{mcc:.4f}\n")
        
        print("Classification Report:")
        print(report)
        
        print("\nConfusion Matrix:")
        print(cm)

        # Clear test outputs after aggregation
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def classify(self, images):
        outputs = self(images)
        return torch.argmax(outputs, dim=1)

def main():
    # Print GPU information
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Set float32 matrix multiplication precision to 'medium' or 'high'
    torch.set_float32_matmul_precision('medium')

    # Load and prepare your data
    print("Loading and preparing data...")
    dataset = pd.read_csv('CSV_Files/mammo-bench_nbm_classification.csv', low_memory=False)

    # exclude the Suspicious Malignant because of its uncertainty and small sample size
    dataset = dataset[dataset['classification'] != 'Suspicious Malignant']

    # exclude the BIRADS 0.0 because they are not normal, they require additional tests to determine if they are normal or abnormal
    dataset = dataset[dataset['BIRADS'] != 0.0]

    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.75, random_state=42)

    print("Dataset class distribution:")
    print("\nTrain dataset:")
    print(train_data['classification'].value_counts())
    print("\nValidation dataset:")
    print(val_data['classification'].value_counts())
    print("\nTest dataset:")
    print(test_data['classification'].value_counts())

    train_loader_stage1 = DataLoader(MammogramData(train_data, 'preprocessed_image_path', task='stage1'), batch_size=BATCH_SIZE1, shuffle=True, num_workers=8, pin_memory=True)
    val_loader_stage1 = DataLoader(MammogramData(val_data, 'preprocessed_image_path', task='stage1'), batch_size=BATCH_SIZE1, num_workers=8, pin_memory=True)
    # test_data_stage_1 is the test_data except the benign and malignant are replaced with abnormal
    test_data_stage1 = test_data.copy()
    test_data_stage1['classification'] = test_data_stage1['classification'].apply(lambda x: 'Abnormal' if x in ['Benign', 'Malignant'] else x)
    test_loader_stage1 = DataLoader(MammogramData(test_data_stage1, 'preprocessed_image_path', task='stage1'), batch_size=BATCH_SIZE1, num_workers=8, pin_memory=True)

    # Stage 1 Model Training
    model_stage1 = MammogramClassifier(num_classes=2, learning_rate=1e-4, weight_decay=1e-3)
    checkpoint_callback_stage1 = ModelCheckpoint(dirpath=os.path.dirname('models/stage1'), filename="binary_stage1-{epoch:02d}-{val_loss:.4f}", save_top_k=3, monitor="val_loss", mode="min")

    trainer_stage1 = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else 'auto',
        callbacks=[checkpoint_callback_stage1],
        precision="16-mixed",
        log_every_n_steps=10
    )

    print("Starting Stage 1 training...")
    trainer_stage1.fit(model_stage1, train_loader_stage1, val_loader_stage1)
    print("Stage 1 training completed.")

    # Stage 1 Testing
    print("Starting Stage 1 testing...")
    trainer_stage1 = Trainer(devices=1)
    trainer_stage1.test(model_stage1, test_loader_stage1)
    print("Stage 1 testing completed.")

    # Load best model for Stage 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_stage1 = MammogramClassifier.load_from_checkpoint(checkpoint_callback_stage1.best_model_path)
    model_stage1 = model_stage1.to(device)

    # Prepare Data for Stage 2
    model_stage1.eval()
    test_preds = []

    # Create a mapping for the true labels
    label_map = {'Normal': 0, 'Benign': 1, 'Malignant': 1}
    true_labels = test_data['classification'].map(label_map).values

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(MammogramData(test_data, 'preprocessed_image_path', task='stage1'), batch_size=BATCH_SIZE2)):
            images = images.cuda()
            # Apply sigmoid to get probabilities and threshold at 0.5
            outputs = model_stage1(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
            test_preds.extend(preds)

    # Convert predictions to numpy array
    test_preds = np.array(test_preds)

    # Calculate and print metrics
    print("\nStage 1 Evaluation Metrics:")
    print("\nClassification Report:")
    print(classification_report(true_labels, test_preds, target_names=['Normal', 'Abnormal']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_labels, test_preds)
    print(cm)

    # Calculate other metrics
    accuracy = accuracy_score(true_labels, test_preds)
    precision = precision_score(true_labels, test_preds, average='weighted')
    recall = recall_score(true_labels, test_preds, average='weighted')
    mcc = matthews_corrcoef(true_labels, test_preds)
    f1 = f1_score(true_labels, test_preds, average='weighted')

    print(f"\nDetailed Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    abnormal_train_data = train_data.copy()
    train_preds = []

    for images, labels in tqdm(DataLoader(MammogramData(train_data, 'preprocessed_image_path', task='stage1'), batch_size=BATCH_SIZE2)):
        images = images.cuda()
        outputs = model_stage1(images)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int)
        train_preds.extend(preds)

    abnormal_train_data = abnormal_train_data.iloc[np.where(np.array(train_preds) == 1)[0]]

    # # save the abnormal_train_data to a csv file
    abnormal_train_data.to_csv('stage2_abnormal_train_data.csv', index=False)
    abnormal_train_data = pd.read_csv('stage2_abnormal_train_data.csv', low_memory=False)

    print("Stage 2 training data class distribution:")
    print(abnormal_train_data['classification'].value_counts())

    # Stage 2 Training
    train_loader_stage2 = DataLoader(MammogramData(abnormal_train_data, 'preprocessed_image_path', task='stage2'), batch_size=BATCH_SIZE1, shuffle=True, num_workers=8, pin_memory=True)
    val_data_stage2, test_data_stage2 = train_test_split(test_data, test_size=0.75, random_state=42)

    val_loader_stage2 = DataLoader(MammogramData(val_data_stage2, 'preprocessed_image_path', task='stage2'), batch_size=BATCH_SIZE1, num_workers=8, pin_memory=True)
    test_loader_stage2 = DataLoader(MammogramData(test_data_stage2, 'preprocessed_image_path', task='stage2'), batch_size=BATCH_SIZE1, num_workers=8, pin_memory=True)

    # Stage 2 Model Training
    model_stage2 = MammogramClassifier(num_classes=2, learning_rate=1e-5, weight_decay=1e-2)
    checkpoint_callback_stage2 = ModelCheckpoint(dirpath=os.path.dirname('models/stage2'), filename="binary_stage2-{epoch:02d}-{val_loss:.4f}", save_top_k=3, monitor="val_loss", mode="min")

    trainer_stage2 = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=num_gpus,
        strategy=DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else 'auto',
        callbacks=[checkpoint_callback_stage2],
        precision="16-mixed",
        log_every_n_steps=10,
        accumulate_grad_batches=4
    )

    print("Starting Stage 2 training...")
    trainer_stage2.fit(model_stage2, train_loader_stage2, val_loader_stage2)
    print("Stage 2 training completed.")

    # Load best model for Stage 2 and perform testing
    model_stage2 = MammogramClassifier.load_from_checkpoint(checkpoint_callback_stage2.best_model_path)
    model_stage2 = model_stage2.to(device)

    print("Starting Stage 2 testing...")
    trainer_stage2 = Trainer(devices=1)
    trainer_stage2.test(model_stage2, test_loader_stage2)
    print("Stage 2 testing completed.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
