# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models.video import r3d_18
import warnings
import joblib
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

class BCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction=self.reduction)
        return bce_loss(inputs, targets)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ResNet3D(nn.Module):
    def __init__(self, num_classes=2, num_features=5):
        super(ResNet3D, self).__init__()
        self.resnet3d = r3d_18(pretrained=True)
        self.resnet3d.stem[0] = nn.Conv3d(
            in_channels=1, out_channels=64,
            kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_image = nn.Sequential(
            nn.Linear(self.resnet3d.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.fc_features = nn.Sequential(
            nn.Linear(num_features, 128),
        )
        self.fc_final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, features):
        x = self.resnet3d.stem(x)
        x = self.resnet3d.layer1(x)
        x = self.resnet3d.layer2(x)
        x = self.resnet3d.layer3(x)
        x = self.resnet3d.layer4(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        image_features = self.fc_image(x)
        radiomics_features = self.fc_features(features.float())
        combined_features = image_features + radiomics_features
        output = self.fc_final(combined_features)
        return output

class PET3DDataset(Dataset):
    def __init__(self, image_files, labels, feature_data, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.feature_data = feature_data
        self.transform = transform
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        while True:
            try:
                image_path = self.image_files[idx]
                label = self.labels[idx]
                features = self.feature_data[idx]
                image_nifti = nib.load(image_path)
                image = image_nifti.get_fdata()
                if self.transform:
                    image = self.transform(image)
                return image, features, label
            except:
                idx = (idx + 1) % len(self.image_files)

def get_data_loaders(train_label_csv, train_image_dir, train_feature_xlsx, batch_size=16, target_size=(64, 48, 64)):
    df_labels = pd.read_csv(train_label_csv, encoding="utf-8-sig")
    labels = df_labels["label"].astype(int).tolist()

    image_filenames = sorted(os.listdir(train_image_dir), key=lambda x: int(x.split('_')[0]))

    feature = pd.read_excel(train_feature_xlsx, header=0)
    sec = ['3866', '3776', '4039', '510', '1627']
    sec = [int(x) for x in sec]
    feature_data = feature.iloc[:, sec].values

    assert len(feature_data) == len(image_filenames)
    assert len(labels) == len(image_filenames)

    train_img_files, val_img_files, train_labels, val_labels, train_features, val_features = train_test_split(
        image_filenames, labels, feature_data, test_size=0.1, random_state=23, stratify=labels
    )

    train_img_files = [os.path.join(train_image_dir, f) for f in train_img_files]
    val_img_files = [os.path.join(train_image_dir, f) for f in val_img_files]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    class Resize3D:
        def __init__(self, target_size):
            self.target_size = target_size
        def __call__(self, img):
            import torch.nn.functional as F
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(img_tensor, size=self.target_size, mode="trilinear", align_corners=False)
            return resized.squeeze()

    transform = transforms.Compose([
        Resize3D(target_size=target_size),
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0)),
        transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)),
    ])

    train_dataset = PET3DDataset(train_img_files, train_labels, train_features, transform=transform)
    val_dataset = PET3DDataset(val_img_files, val_labels, val_features, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, save_path="best_model.pth"):
    best_val_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for images, features, labels in train_loader:
            images, features, labels = images.to(device), features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for images, features, labels in val_loader:
                images, features, labels = images.to(device), features.to(device), labels.to(device)
                outputs = model(images, features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"Epoch {epoch + 1}/{num_epochs},Val AUC: {val_auc:.2f}")

        if val_auc > best_val_auc:

            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with, Val AUC: {val_auc:.4f}")
            print(f"Predictions: {all_predictions}")
            print(f"True Labels: {all_labels}")
            print(f"Predicted Probabilities for Class 1: {all_probs}")
            cm = confusion_matrix(all_labels, all_predictions)
            print("Confusion Matrix:")
            print(cm)
    print(f"Training complete. Best Val AUC: {best_val_auc:.4f}")

def test(model, test_image_dir, test_feature_data, test_label_csv, batch_size=16, target_size=(64,64,64), device='cuda'):
    model.eval()

    class Resize3D:
        def __init__(self, target_size):
            self.target_size = target_size
        def __call__(self, img):
            import torch.nn.functional as F
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(img_tensor, size=self.target_size, mode="trilinear", align_corners=False)
            return resized.squeeze()

    transform = transforms.Compose([
        Resize3D(target_size=target_size),
        transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0)),
        transforms.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-8)),
    ])

    class TestDataset(Dataset):
        def __init__(self, image_files, feature_data, transform=None):
            self.image_files = image_files
            self.feature_data = feature_data
            self.transform = transform
        def __len__(self):
            return len(self.image_files)
        def __getitem__(self, idx):
            image_path = self.image_files[idx]
            features = self.feature_data[idx]
            image_nifti = nib.load(image_path)
            image = image_nifti.get_fdata()
            if self.transform:
                image = self.transform(image)
            return image, features

    test_image_files = sorted(os.listdir(test_image_dir), key=lambda x: int(x.split('_')[0]))
    test_image_files = [os.path.join(test_image_dir, f) for f in test_image_files]

    test_features = pd.read_excel(test_feature_data, header=0)
    sec = ['3866', '3776', '4039', '510', '1627']
    sec = [int(x) for x in sec]
    selected_features = test_features.iloc[:, sec].values

    scaler = joblib.load(r"./scaler_train.pkl")
    selected_features = scaler.transform(selected_features)

    test_dataset = TestDataset(test_image_files, selected_features, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    labels = pd.read_csv(test_label_csv, encoding="utf-8-sig")["label"].astype(int).tolist()

    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for images, features in test_loader:
            images, features = images.to(device), features.to(device)
            outputs = model(images, features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(labels, all_predictions)
    precision = precision_score(labels, all_predictions)
    recall = recall_score(labels, all_predictions)
    f1 = f1_score(labels, all_predictions)
    auc = roc_auc_score(labels, all_probs)

    print(all_predictions)
    print(all_probs)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    prediction_results = pd.DataFrame({
        'Image': test_image_files,
        'Predicted Class': all_predictions,
        'Predicted Probability': all_probs
    })
    prediction_results.to_csv(r"C:\Users\zength\Desktop\fusion_test_predictions.csv", index=False)

train_label_csv = r"C:\Users\zength\Desktop\labels_train.csv"
train_image_dir = r"C:\Users\zength\Desktop\resample_img_train"
train_feature_xlsx = r"C:\Users\zength\Desktop\rad_train_features.xlsx"

test1_label_csv = r"C:\Users\zength\Desktop\labels_test1.csv"
test1_image_dir = r"C:\Users\zength\Desktop\resample_img_test1"
test1_feature_xlsx = r"C:\Users\zength\Desktop\rad_test1_features.xlsx"

batch_size = 16
num_epochs = 500
learning_rate = 1e-4
target_size = (64,64,64)

train_loader, val_loader = get_data_loaders(
    train_label_csv, train_image_dir, train_feature_xlsx, batch_size=batch_size, target_size=target_size
)
model = ResNet3D(num_classes=2).to('cuda')
criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
device = torch.device("cuda")

# train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=num_epochs, save_path=r"C:\Users\zength\Desktop\best_fusion_model.pth")

checkpoint = r'C:\Users\zength\Desktop\best_fusion_model.pth'
model.load_state_dict(torch.load(checkpoint))
test(model, test1_image_dir, test1_feature_xlsx, test1_label_csv, batch_size=16, target_size=(64,64,64), device='cuda')