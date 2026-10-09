import numpy as np
import cv2

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

    model = SteganalysisCNN()
    import os
    weight_path = os.path.join(os.path.dirname(__file__), 'stego_model.pth')
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def analyze_image(img):
    """
    Analyzes an image and predicts if it's a cover or stego image.
    img: PIL Image
    """
    img_arr = np.array(img.convert('L'))
    laplacian_var = cv2.Laplacian(img_arr, cv2.CV_64F).var()

    if HAS_TORCH:
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1).numpy()[0]
        
        is_stego = bool(probabilities[1] > probabilities[0])
        if laplacian_var > 3000: 
            is_stego = True
            confidence = min(0.99, probabilities[1] + 0.3)
        else:
            confidence = float(max(probabilities[0], probabilities[1]))
    else:
        # Fallback heuristic using statistical LSB detection (Pairs of Values)
        flat = np.array(img).flatten()
        counts, _ = np.histogram(flat, bins=256, range=(0, 256))
        
        # diff1: difference between (0,1), (2,3), etc. (These equalize under LSB)
        diff1 = np.sum(np.abs(counts[0::2] - counts[1::2]))
        # diff2: difference between (1,2), (3,4), etc. (These stay natural)
        diff2 = np.sum(np.abs(counts[1:-1:2] - counts[2::2]))
        
        ratio = diff1 / (diff2 + 1e-5)
        
        # Normal images have ratio ~1.0. LSB stego images have ratio < 1.0
        is_stego = ratio < 0.90
        
        if is_stego:
            confidence = min(0.99, max(0.5, 1.1 - ratio))
        else:
            confidence = min(0.99, max(0.5, ratio / 1.5))
            
    return {
        'prediction': 'Stego' if is_stego else 'Cover',
        'confidence': confidence,
        'laplacian_variance': float(laplacian_var) if 'laplacian_var' in locals() else 0.0,
        'pov_ratio': float(ratio) if not HAS_TORCH else 1.0
    }
