import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
from torchvision import transforms
from torchvision.models.video import r3d_18
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.init as init
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.attention = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, x):
        batch_size, channels, D, H, W = x.size()
        x_flat = x.view(batch_size, channels, -1).permute(2, 0, 1)
        attn_output, _ = self.attention(x_flat, x_flat, x_flat)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, channels, D, H, W)
        return attn_output

class ResNet3D(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet3D, self).__init__()
        self.resnet3d = r3d_18(pretrained=True)
        self.resnet3d.stem[0] = nn.Conv3d(
            in_channels=1, out_channels=64,
            kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        self.resnet3d.fc = nn.Sequential(
            nn.Linear(self.resnet3d.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.resnet3d.stem(x)
        x = self.resnet3d.layer1(x)
        x = self.resnet3d.layer2(x)
        x = self.resnet3d.layer3(x)
        x = self.resnet3d.layer4(x)
        x = self.resnet3d.avgpool(x)
        x = x.flatten(1)
        x = self.resnet3d.fc(x)
        return x

class PET3DDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            try:
                image_path = self.image_files[idx]
                label = int(self.labels[idx])
                image_nifti = nib.load(image_path)
                image = image_nifti.get_fdata()
                if self.transform:
                    image = self.transform(image)
                return image, label
            except:
                idx = (idx + 1) % len(self.image_files)

def get_data_loaders(train_csv_path, img_train_dir, target_size, batch_size=8):
    df_train = pd.read_csv(train_csv_path, encoding="utf-8-sig")
    labels_all = df_train["label"].astype(int).tolist()

    image_files_all = sorted(os.listdir(img_train_dir), key=lambda x: int(x.split('_')[0]))
    assert len(labels_all) == len(image_files_all), f"{len(labels_all)} != {len(image_files_all)}"

    train_files, val_files, train_labels, val_labels = train_test_split(
        image_files_all, labels_all,
        test_size=0.1,
        random_state=23,
        stratify=labels_all
    )

    train_img_files = [os.path.join(img_train_dir, f) for f in train_files]
    val_img_files = [os.path.join(img_train_dir, f) for f in val_files]

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

    train_dataset = PET3DDataset(train_img_files, train_labels, transform=transform)
    val_dataset = PET3DDataset(val_img_files, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=300, save_path="best_model.pth"):
    model.to(device)
    best_val_auc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        all_predictions, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)


def test(model, test_image_dir, test_label_csv, target_size, batch_size=8, device='cuda'):
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
        def __init__(self, image_files, transform=None):
            self.image_files = image_files
            self.transform = transform
        def __len__(self):
            return len(self.image_files)
        def __getitem__(self, idx):
            image_path = self.image_files[idx]
            image_nifti = nib.load(image_path)
            image = image_nifti.get_fdata()
            if self.transform:
                image = self.transform(image)
            return image

    test_image_files = sorted(os.listdir(test_image_dir), key=lambda x: int(x.split('_')[0]))
    test_image_files = [os.path.join(test_image_dir, f) for f in test_image_files]
    test_dataset = TestDataset(test_image_files, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    df_lab = pd.read_csv(test_label_csv, encoding="utf-8-sig")
    labels = df_lab["label"].astype(int).tolist()

    all_predictions, all_probs = [], []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
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
    print(labels)
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
    prediction_results.to_csv('test_predictions.csv', index=False)

train_csv_path = r"./labels_train.csv"
test1_csv_path = r"./labels_test1.csv"
test2_csv_path = r"./labels_test2.csv"

image_train_dir = r"./resample_img_train"
image_test1_dir = r"./resample_img_test1"
image_test2_dir = r"./resample_img_test2"

batch_size = 16
num_epochs = 200
learning_rate = 1e-4
target_size = (64, 64, 64)

train_loader, val_loader = get_data_loaders(
    train_csv_path=train_csv_path,
    img_train_dir=image_train_dir,
    target_size=target_size,
    batch_size=batch_size,
)

model = ResNet3D(num_classes=2).to('cuda')
criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
device = torch.device("cuda")

train_model(model, train_loader, val_loader, criterion, optimizer, device, scheduler, num_epochs=num_epochs, save_path=r"C:\Users\zength\Desktop\best_model.pth")
