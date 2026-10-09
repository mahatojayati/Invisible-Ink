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
import numpy as np

# Import the model architecture and embedding function
from models.steganalysis import SteganalysisCNN
from models.stego import hide_data

class StegoDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
        # We will create cover and stego pairs dynamically or they can be pre-generated.
        # To save disk space, we load the base image. If index is even, we return Cover.
        # If index is odd, we embed data and return Stego.
        # This doubles our dataset size automatically.
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
            # Use LSB
            stego_arr = hide_data(img_arr, payload, method='lsb')
            img = Image.fromarray(stego_arr)
            label = 1
        else:
            label = 0
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

def generate_synthetic_images(num_images=100, output_dir='dataset_images'):
    """Generate some random fractal/noise images for training if no dataset is provided."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i in range(num_images):
        path = os.path.join(output_dir, f"img_{i}.png")
        if not os.path.exists(path):
            # Create a simple synthetic image (Perlin noise like)
            arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
            # Smooth it to look slightly more like a natural image
            import cv2
            arr = cv2.GaussianBlur(arr, (5, 5), 0)
            Image.fromarray(arr).save(path)
        paths.append(path)
    return paths

def train_model():
    print("Generating/loading base images...")
    base_images = generate_synthetic_images(num_images=200) # Small dataset for demonstration
    
    # 1. SPLITTING STRATEGY (Avoid Data Leakage)
    # We must split the BASE images into train/val/test BEFORE creating stego pairs.
    # If we split after creating pairs, the model might memorize the cover image in Train
    # and unfairly detect the stego version in Test.
    train_paths, test_paths = train_test_split(base_images, test_size=0.2, random_state=42)
    train_paths, val_paths = train_test_split(train_paths, test_size=0.2, random_state=42)
    
    print(f"Dataset split (base images): {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

    # 2. PREPROCESSING (Must match inference EXACTLY)
    # We use CenterCrop instead of Resize to avoid destroying LSB artifacts via interpolation.
    preprocessing_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = StegoDataset(train_paths, transform=preprocessing_transform)
    val_dataset = StegoDataset(val_paths, transform=preprocessing_transform)
    test_dataset = StegoDataset(test_paths, transform=preprocessing_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = SteganalysisCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 3
    print("Starting training...")
    for epoch in range(num_epochs):
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
            
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {100 * val_correct / val_total:.2f}%")
        
    print("Evaluating on Test Set...")
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * test_correct / test_total:.2f}%")
    
    # Save the trained model and class mapping
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/stego_model.pth')
    
    class_mapping = {
        "0": "Cover",
        "1": "Stego"
    }
    with open('models/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
        
    print("Model saved to models/stego_model.pth")
    print("Class mapping saved to models/class_mapping.json")

if __name__ == "__main__":
    train_model()
