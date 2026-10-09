import os
import io
import uuid
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
from PIL import Image

from models.stego import hide_data, extract_data
from models.steganalysis import analyze_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

import models.steganalysis as sa

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'torch_available': sa.HAS_TORCH,
        'model_loaded': sa.model is not None,
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/embed', methods=['POST'])
def embed():
    if 'image' not in request.files or 'message' not in request.form:
        return jsonify({'error': 'Image or message missing'}), 400
    
    file = request.files['image']
    message = request.form['message']
    method = request.form.get('method', 'lsb') # lsb or adaptive
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream).convert('RGB')
        img_arr = np.array(img)
        
        # Hide data
        stego_arr = hide_data(img_arr, message, method=method)
        stego_img = Image.fromarray(stego_arr)
        
        # Save to memory and return
        img_byte_arr = io.BytesIO()
        stego_img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        filename = f"stego_{uuid.uuid4().hex[:8]}.png"
        return send_file(img_byte_arr, mimetype='image/png', as_attachment=True, download_name=filename)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return jsonify({'error': 'Image missing'}), 400
        
    file = request.files['image']
    method = request.form.get('method', 'lsb')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        img = Image.open(file.stream).convert('RGB')
        img_arr = np.array(img)
        
        message = extract_data(img_arr, method=method)
        return jsonify({'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'Image missing'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        img = Image.open(file.stream).convert('RGB')
        
        # ML Steganalysis
        result = analyze_image(img)
        return jsonify(result)
    except RuntimeError as re:
        # Explicitly handle model unavailability so frontend doesn't show "Cover"
        return jsonify({
            'prediction': 'Unavailable',
            'confidence': 0.0,
            'laplacian_variance': 0.0,
            'error': str(re)
        }), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
