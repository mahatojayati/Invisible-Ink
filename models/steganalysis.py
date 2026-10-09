import numpy as np
import cv2
import json
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class SteganalysisCNN(nn.Module):
        def __init__(self):
            super(SteganalysisCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 64 * 64, 128)
            self.fc2 = nn.Linear(128, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 32 * 64 * 64)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

# Global state
model = None
transform = None
class_mapping = {0: "Cover", 1: "Stego"} # Default

def initialize_model():
    global model, transform, class_mapping
    if not HAS_TORCH:
        return False
        
    model = SteganalysisCNN()
    
    # Load class mapping if exists
    mapping_path = os.path.join(os.path.dirname(__file__), 'class_mapping.json')
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                loaded_map = json.load(f)
                class_mapping = {int(k): v for k, v in loaded_map.items()}
        except Exception as e:
            print(f"Failed to load class mapping: {e}")

    # Load weights
    weight_path = os.path.join(os.path.dirname(__file__), 'stego_model.pth')
    if os.path.exists(weight_path):
        try:
            device = torch.device('cpu')
            model.load_state_dict(torch.load(weight_path, map_location=device))
        except Exception as e:
            print(f"Failed to load weights: {e}")
            model = None
            return False
    else:
        # Untrained model is not useful, set to None
        model = None
        return False

    model.eval()

    # Preprocessing must match training precisely
    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return True

# Initialize once on load
initialize_model()

def analyze_image(img):
    """
    Analyzes an image and predicts if it's a cover or stego image.
    """
    if img is None:
        raise ValueError("Invalid image provided.")
        
    # Ensure RGB conversion (dropping alpha if present)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img_arr = np.array(img.convert('L'))
    laplacian_var = cv2.Laplacian(img_arr, cv2.CV_64F).var()

    if model is not None and HAS_TORCH:
        try:
            # The tensor needs to be at least 256x256 for CenterCrop.
            # Handle images smaller than 256x256 gracefully by padding
            if img.width < 256 or img.height < 256:
                pad_transform = transforms.Pad((max(0, (256 - img.width) // 2), max(0, (256 - img.height) // 2)), fill=0)
                img_for_tensor = pad_transform(img)
            else:
                img_for_tensor = img

            input_tensor = transform(img_for_tensor).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = F.softmax(output, dim=1).numpy()[0]
            
            predicted_class_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class_idx])
            prediction = class_mapping.get(predicted_class_idx, "Unknown")
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'laplacian_variance': float(laplacian_var)
            }
        except Exception as e:
            raise RuntimeError(f"Inference error: {str(e)}")
    else:
        # Strict statistical fallback without fabricating confidence
        flat = np.array(img).flatten()
        counts, _ = np.histogram(flat, bins=256, range=(0, 256))
        
        diff1 = np.sum(np.abs(counts[0::2] - counts[1::2]))
        diff2 = np.sum(np.abs(counts[1:-1:2] - counts[2::2]))
        
        ratio = float(diff1 / (diff2 + 1e-5))
        is_stego = ratio < 0.90
        
        return {
            'prediction': 'Stego' if is_stego else 'Cover',
            'confidence': 0.0, # Do not fabricate confidence
            'laplacian_variance': float(laplacian_var),
            'pov_ratio': ratio,
            'warning': 'Model untrained or missing. Used statistical fallback.'
        }
