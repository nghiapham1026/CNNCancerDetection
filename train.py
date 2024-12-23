import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Local imports
from dataset import HistopathologyDataset
from transforms import train_transform, val_transform
from models import SimpleCNN, create_resnet18_model

# Training Config
DATA_DIR = "./data"
TRAIN_IMG_DIR = "train"
LABELS_CSV = "train_labels.csv"
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_and_compare_architectures():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv(os.path.join(DATA_DIR, LABELS_CSV))

    # Optional: Subsample for quick runs
    # df = df.sample(20000, random_state=42).reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_dataset = HistopathologyDataset(train_df, os.path.join(DATA_DIR, TRAIN_IMG_DIR), transform=train_transform)
    val_dataset = HistopathologyDataset(val_df, os.path.join(DATA_DIR, TRAIN_IMG_DIR), transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Two architectures to compare
    architectures = {
        "scratch": SimpleCNN(num_classes=2),
        "resnet18": create_resnet18_model(num_classes=2)
    }

    history = {
        "scratch": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []},
        "resnet18": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    }

    criterion = nn.CrossEntropyLoss()

    for arch_name, model in architectures.items():
        print(f"\n=== Training {arch_name} model ===")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history[arch_name]["train_loss"].append(train_loss)
            history[arch_name]["val_loss"].append(val_loss)
            history[arch_name]["train_acc"].append(train_acc)
            history[arch_name]["val_acc"].append(val_acc)

            duration = time.time() - start_time
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                  f"Time: {duration:.1f} sec")

    return history

def main():
    # 1. Train the models
    results = train_and_compare_architectures()

    # 2. Plot results
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    
    # Subplot: Loss
    plt.subplot(1, 2, 1)
    for arch_name in results:
        plt.plot(epochs_range, results[arch_name]["train_loss"], label=f"{arch_name} Train Loss")
        plt.plot(epochs_range, results[arch_name]["val_loss"], label=f"{arch_name} Val Loss", linestyle="--")
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Subplot: Accuracy
    plt.subplot(1, 2, 2)
    for arch_name in results:
        plt.plot(epochs_range, results[arch_name]["train_acc"], label=f"{arch_name} Train Acc")
        plt.plot(epochs_range, results[arch_name]["val_acc"], label=f"{arch_name} Val Acc", linestyle="--")
    plt.title("Training vs. Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
