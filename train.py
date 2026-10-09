import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import random
import argparse

# Set fixed seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

from models.steganalysis import SteganalysisCNN
from models.stego import hide_data

class StegoDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.num_base_images = len(image_paths)

    def __len__(self):
        return self.num_base_images * 2

    def __getitem__(self, idx):
        base_idx = idx // 2
        is_stego = (idx % 2 == 1)
        
        img_path = self.image_paths[base_idx]
        img = Image.open(img_path).convert('RGB')
        
        if is_stego:
            # Embed a random payload
            payload = "SECRET_" + str(np.random.randint(1000, 9999))
            img_arr = np.array(img)
            try:
                stego_arr = hide_data(img_arr, payload, method='lsb')
                img = Image.fromarray(stego_arr)
            except ValueError:
                # If image too small for payload, skip embedding
                pass
            label = 1
        else:
            label = 0
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = total_loss / len(loader)
    
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, prec, rec, f1, cm, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description="Train Steganalysis Model")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to directory containing pure cover images")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    args = parser.parse_args()

    set_seed(42)

    # 1 & 2. Dataset verification
    if not os.path.exists(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory {args.dataset_dir} not found. Please provide a directory of cover images.")
        
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    base_images = [os.path.join(args.dataset_dir, f) for f in os.listdir(args.dataset_dir) if f.lower().endswith(valid_exts)]
    
    if len(base_images) < 10:
        raise ValueError("Insufficient dataset size. Please provide at least 10 cover images.")

    print(f"Found {len(base_images)} cover images. Stego images will be generated dynamically.")

    # 3. Prevent Data Leakage & Class Imbalance
    # Split base images first, then dataset class duplicates them exactly 50/50 Cover/Stego
    train_paths, test_paths = train_test_split(base_images, test_size=0.2, random_state=42)
    train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=42)
    
    print(f"Splits (base images) - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # 4. Preprocessing identical to inference
    preprocessing_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = StegoDataset(train_paths, transform=preprocessing_transform)
    val_dataset = StegoDataset(val_paths, transform=preprocessing_transform)
    test_dataset = StegoDataset(test_paths, transform=preprocessing_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = SteganalysisCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_f1 = 0.0
    best_model_state = None
    
    # 5 & 6. Reproducible training and Tracking metrics
    print("\nStarting Training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        val_loss, prec, rec, f1, cm, _, _ = evaluate_model(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {f1:.4f}")
        
        # 9. Save the best validated model
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: Model did not improve F1 above 0. Using final epoch weights.")

    # 7. Evaluate performance on an independent test set
    print("\nEvaluating on Test Set...")
    test_loss, prec, rec, f1, cm, preds, labels = evaluate_model(model, test_loader, device, criterion)
    print(f"Test Precision: {prec:.4f} | Test Recall: {rec:.4f} | Test F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    # 8. Compare against Baseline (Statistical PoV)
    print("\nEvaluating Baseline (Statistical PoV Ratio) on Test Set...")
    baseline_preds = []
    for idx in range(len(test_dataset)):
        base_idx = idx // 2
        is_stego = (idx % 2 == 1)
        img_path = test_paths[base_idx]
        img = Image.open(img_path).convert('RGB')
        
        if is_stego:
            img_arr = np.array(img)
            payload = "SECRET_" + str(np.random.randint(1000, 9999))
            try:
                stego_arr = hide_data(img_arr, payload, method='lsb')
                img = Image.fromarray(stego_arr)
            except ValueError:
                pass
                
        flat = np.array(img).flatten()
        counts, _ = np.histogram(flat, bins=256, range=(0, 256))
        diff1 = np.sum(np.abs(counts[0::2] - counts[1::2]))
        diff2 = np.sum(np.abs(counts[1:-1:2] - counts[2::2]))
        ratio = float(diff1 / (diff2 + 1e-5))
        pred = 1 if ratio < 0.90 else 0
        baseline_preds.append(pred)
        
    base_prec = precision_score(labels, baseline_preds, zero_division=0)
    base_rec = recall_score(labels, baseline_preds, zero_division=0)
    base_f1 = f1_score(labels, baseline_preds, zero_division=0)
    print(f"Baseline Precision: {base_prec:.4f} | Baseline Recall: {base_rec:.4f} | Baseline F1: {base_f1:.4f}")

    # 10. Safeguard against deploying a collapsed model
    if f1 < 0.55 or cm.min() == 0:
        print("\n[!] SAFEGUARD TRIGGERED: Model exhibits poor performance or class collapse.")
        print("[!] The model predicts mostly one class or fails to detect steganography reliably.")
        print("[!] The trained model will NOT be saved. Please provide a larger dataset or train longer.")
        exit(1)

    print("\nModel passed safeguards. Saving model...")
    os.makedirs('models', exist_ok=True)
    torch.save(best_model_state, 'models/stego_model.pth')
    
    class_mapping = {"0": "Cover", "1": "Stego"}
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
        
    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "test_f1_score": f1,
        "test_precision": prec,
        "test_recall": rec
    }
    with open('models/training_config.json', 'w') as f:
        json.dump(training_config, f)
        
    print("Successfully saved model, class mapping, and training config.")

if __name__ == "__main__":
    main()
