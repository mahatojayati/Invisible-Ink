# InvisibleInk

**InvisibleInk: Machine Learning based Steganalysis and secure data hiding**

This project provides an integrated system that securely embeds secret data within cover images using standard and adaptive Least Significant Bit (LSB) embedding techniques, and applies a Machine Learning-based steganalysis model (PyTorch CNN) to detect whether a given image is a plain cover image or a stego image containing hidden data.

## Features
- **Secure Hide (Steganography)**: Hide text messages inside PNG images using Standard LSB or Adaptive Embedding (edge-based).
- **Extract Data**: Retrieve hidden messages from stego images.
- **ML Steganalysis**: Analyze an image using a PyTorch Convolutional Neural Network to detect steganographic payloads and noise residuals.

## Installation & Setup

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python app.py
   ```
4. Open the web interface:
   Visit `http://localhost:5001` in your browser.

## Technologies Used
- **Backend**: Python, Flask, PyTorch, OpenCV, NumPy, Pillow
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism aesthetics), JavaScript
- **Algorithms**: LSB Substitution, Adaptive Embedding (Canny Edge detection), CNN Binary Classifier
