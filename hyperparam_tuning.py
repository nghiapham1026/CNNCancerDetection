import os
import time
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# Local imports
from dataset import HistopathologyDataset
from transforms import train_transform, val_transform
from models import create_resnet18_model, create_resnet34_model
from train import train_one_epoch, evaluate

# Config
DATA_DIR = "./data"
TRAIN_IMG_DIR = "train"
LABELS_CSV = "train_labels.csv"
BATCH_SIZE = 32

def experiment_with_params(model_fn, train_dataset, val_dataset, learning_rate=1e-3, weight_decay=0.0, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_fn().to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = 0.0
    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        duration = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.4f} | Time: {duration:.1f} sec")

    return best_val_acc

def hyperparam_tuning_experiment():
    df = pd.read_csv(os.path.join(DATA_DIR, LABELS_CSV))
    df = df.sample(10000, random_state=42).reset_index(drop=True)  # Subsample for speed

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_dataset = HistopathologyDataset(train_df, os.path.join(DATA_DIR, TRAIN_IMG_DIR), transform=train_transform)
    val_dataset   = HistopathologyDataset(val_df,   os.path.join(DATA_DIR, TRAIN_IMG_DIR), transform=val_transform)

    learning_rates = [1e-3, 1e-4]
    weight_decays  = [0.0, 1e-5]
    architectures  = {
        "ResNet18": create_resnet18_model,
        "ResNet34": create_resnet34_model
    }

    results = []

    for arch_name, arch_fn in architectures.items():
        for lr, wd in itertools.product(learning_rates, weight_decays):
            print(f"\n=== Experiment: {arch_name}, LR={lr}, WD={wd} ===")
            best_val_acc = experiment_with_params(
                model_fn=arch_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                learning_rate=lr,
                weight_decay=wd,
                epochs=10
            )
            results.append({
                "Architecture": arch_name,
                "Learning Rate": lr,
                "Weight Decay": wd,
                "Best Val Acc": best_val_acc
            })
    
    results_df = pd.DataFrame(results)
    return results_df

def main():
    tuning_results = hyperparam_tuning_experiment()
    print("\n=== Hyperparameter Tuning Results ===")
    print(tuning_results)

    # Optional bar chart for final accuracies
    plt.figure(figsize=(8, 4))
    x_positions = range(len(tuning_results))
    plt.bar(x_positions, tuning_results["Best Val Acc"], color='skyblue')
    plt.xticks(x_positions, tuning_results.apply(
        lambda row: f"{row['Architecture']}\nLR={row['Learning Rate']}\nWD={row['Weight Decay']}", axis=1),
        rotation=45, ha="right")
    plt.ylim([0, 1])
    plt.ylabel("Best Validation Accuracy")
    plt.title("Hyperparameter Tuning Comparison")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
