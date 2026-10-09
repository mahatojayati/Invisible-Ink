import numpy as np
import cv2

DELIMITER = "||END||"

def message_to_bin(msg):
    return ''.join([format(ord(c), '08b') for c in msg])

def bin_to_message(bin_str):
    msg = ""
    for i in range(0, len(bin_str), 8):
        byte = bin_str[i:i+8]
        if len(byte) == 8:
            msg += chr(int(byte, 2))
    return msg

def hide_data(img_arr, message, method='lsb'):
    """
    Hides a message into an image array.
    """
    img = img_arr.copy()
    message += DELIMITER
    binary_msg = message_to_bin(message)
    msg_len = len(binary_msg)
    
    if method == 'lsb':
        flat_img = img.flatten()
        if msg_len > len(flat_img):
            raise ValueError("Message too large for this image.")
            
        for i in range(msg_len):
            flat_img[i] = (flat_img[i] & 254) | int(binary_msg[i])
            
        return flat_img.reshape(img.shape)
        
    elif method == 'adaptive':
        # Simple adaptive: avoid smooth areas by calculating edge magnitude
        # We embed only in pixels where edge magnitude > threshold
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        flat_img = img.flatten()
        flat_edges = np.repeat(edges.flatten(), 3)
        
        # Indices of highly textured areas
        textured_idx = np.where(flat_edges > 0)[0]
        
        if msg_len > len(textured_idx):
            # Fallback to standard LSB if not enough textured pixels
            print("Not enough textured pixels, falling back to LSB")
            return hide_data(img_arr, message, method='lsb')
            
        for i in range(msg_len):
            idx = textured_idx[i]
            flat_img[idx] = (flat_img[idx] & 254) | int(binary_msg[i])
            
        return flat_img.reshape(img.shape)
    else:
        raise ValueError("Unknown embedding method")

def extract_data(img_arr, method='lsb'):
    """
    Extracts message from image array.
    """
    if method == 'lsb':
        flat_img = img_arr.flatten()
        binary_msg = ""
        
        for i in range(len(flat_img)):
            binary_msg += str(flat_img[i] & 1)
            # Check for delimiter every 8 bits
            if len(binary_msg) % 8 == 0:
                if len(binary_msg) >= len(DELIMITER)*8:
                    current_msg = bin_to_message(binary_msg[-len(DELIMITER)*8:])
                    if current_msg.endswith(DELIMITER):
                        return bin_to_message(binary_msg[:-len(DELIMITER)*8])
        
        return "No hidden message found."

    elif method == 'adaptive':
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        flat_img = img_arr.flatten()
        flat_edges = np.repeat(edges.flatten(), 3)
        
        textured_idx = np.where(flat_edges > 0)[0]
        binary_msg = ""
        
        for i in range(len(textured_idx)):
            idx = textured_idx[i]
            binary_msg += str(flat_img[idx] & 1)
            
            if len(binary_msg) % 8 == 0:
                if len(binary_msg) >= len(DELIMITER)*8:
                    current_msg = bin_to_message(binary_msg[-len(DELIMITER)*8:])
                    if current_msg.endswith(DELIMITER):
                        return bin_to_message(binary_msg[:-len(DELIMITER)*8])
                        
        return "No hidden message found."
