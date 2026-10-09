import os
import unittest
import numpy as np
import cv2
import io
import tempfile
import json
from PIL import Image

from app import app
from models.stego import hide_data, extract_data
from models.steganalysis import analyze_image, initialize_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

class TestSteganalysisApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test client
        app.testing = True
        cls.client = app.test_client()
        
        # Temporary directory for test images
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Generate some cover images
        cls.cover_paths = []
        cls.stego_paths = []
        
        for i in range(10):
            # Generate random smooth image
            arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
            arr = cv2.GaussianBlur(arr, (5, 5), 0)
            
            cover_path = os.path.join(cls.temp_dir.name, f"cover_{i}.png")
            Image.fromarray(arr).save(cover_path)
            cls.cover_paths.append(cover_path)
            
            # Generate corresponding stego image
            stego_arr = hide_data(arr, "TEST_PAYLOAD", method='lsb')
            stego_path = os.path.join(cls.temp_dir.name, f"stego_{i}.png")
            Image.fromarray(stego_arr).save(stego_path)
            cls.stego_paths.append(stego_path)

        # Force initialize model mapping
        initialize_model()

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_01_embedding_and_extraction(self):
        """Verify that embedding and extraction work reliably."""
        img = Image.open(self.cover_paths[0])
        img_arr = np.array(img)
        
        # Test LSB
        payload = "TESTING_LSB_123"
        stego_arr = hide_data(img_arr, payload, method='lsb')
        extracted = extract_data(stego_arr, method='lsb')
        self.assertEqual(payload, extracted, "LSB extraction failed.")

    def test_02_invalid_and_corrupt_files(self):
        """Test API behavior with invalid files, corrupted images, and unsupported formats."""
        # 1. Missing image
        resp = self.client.post('/api/analyze')
        self.assertEqual(resp.status_code, 400)
        self.assertIn('Image missing', resp.get_json()['error'])
        
        # 2. Corrupt image file
        data = {'image': (io.BytesIO(b"this is not an image"), 'corrupt.png')}
        resp = self.client.post('/api/analyze', data=data, content_type='multipart/form-data')
        self.assertEqual(resp.status_code, 500)

    def test_03_api_response_format(self):
        """Test that the analyze API returns the expected format and valid confidence scores."""
        with open(self.cover_paths[0], 'rb') as f:
            data = {'image': (f, 'cover.png')}
            resp = self.client.post('/api/analyze', data=data, content_type='multipart/form-data')
            
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()
        self.assertIn('prediction', json_data)
        self.assertIn('confidence', json_data)
        self.assertIn('laplacian_variance', json_data)
        
        self.assertIn(json_data['prediction'], ['Cover', 'Stego'])
        self.assertTrue(0.0 <= json_data['confidence'] <= 1.0)

    def test_04_class_mapping_logic(self):
        """Verify the class-index-to-label mapping is loaded and accurate."""
        import models.steganalysis as sa
        self.assertTrue(isinstance(sa.class_mapping, dict))
        self.assertIn(0, sa.class_mapping)
        self.assertIn(1, sa.class_mapping)
        self.assertEqual(sa.class_mapping[0], 'Cover')
        self.assertEqual(sa.class_mapping[1], 'Stego')

    def test_05_preprocessing_consistency(self):
        """Ensure preprocessing applies a CenterCrop rather than Resize to preserve LSB artifacts."""
        import models.steganalysis as sa
        if sa.HAS_TORCH and sa.transform is not None:
            has_crop = any(isinstance(t, sa.transforms.CenterCrop) for t in sa.transform.transforms)
            has_resize = any(isinstance(t, sa.transforms.Resize) for t in sa.transform.transforms)
            self.assertTrue(has_crop, "CenterCrop must be present for preprocessing consistency.")
            self.assertFalse(has_resize, "Resize must NOT be present as it destroys LSB artifacts.")

    def test_06_evaluation_suite_and_collapse_regression(self):
        """
        Report metrics on an independent test set.
        Add a regression test to detect if the model predicts Cover for EVERY test image.
        """
        all_labels = []
        all_preds = []
        
        # Collect predictions for Cover images (Label 0)
        for path in self.cover_paths:
            with open(path, 'rb') as f:
                data = {'image': (f, 'cover.png')}
                resp = self.client.post('/api/analyze', data=data, content_type='multipart/form-data')
                pred = resp.get_json()['prediction']
                all_preds.append(0 if pred == 'Cover' else 1)
                all_labels.append(0)
                
        # Collect predictions for Stego images (Label 1)
        for path in self.stego_paths:
            with open(path, 'rb') as f:
                data = {'image': (f, 'stego.png')}
                resp = self.client.post('/api/analyze', data=data, content_type='multipart/form-data')
                pred = resp.get_json()['prediction']
                all_preds.append(0 if pred == 'Cover' else 1)
                all_labels.append(1)
                
        cm = confusion_matrix(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print("\n--- Model Evaluation Suite ---")
        print(f"Total Samples: {len(all_labels)}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"False Positive Rate (FPR): {fpr:.4f}")
        print(f"False Negative Rate (FNR): {fnr:.4f}")
        print("------------------------------\n")
        
        # Regression Test: Check for model collapse (Predicts Cover for everything)
        if sum(all_preds) == 0:
            self.fail("Regression Test Failed: Model exhibits class collapse and predicts 'Cover' for every single test image. (Even on known stego images).")

if __name__ == '__main__':
    unittest.main()
