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

def _get_crop_bounds(img_shape, target_size=(256, 256)):
    """
    Returns (top, left, height, width) for the central region that matches
    CenterCrop(target_size). If the image is smaller than target_size, bounds are (0, 0, h, w).
    """
    h, w = img_shape[:2]
    crop_h, crop_w = target_size
    top = max(0, (h - crop_h) // 2) if h >= crop_h else 0
    left = max(0, (w - crop_w) // 2) if w >= crop_w else 0
    height = min(h, crop_h)
    width = min(w, crop_w)
    return top, left, height, width

def _extract_from_flat(flat_arr):
    """Helper to extract a message ending with DELIMITER from a flat array."""
    binary_msg = ""
    delim_len_bits = len(DELIMITER) * 8
    for i in range(len(flat_arr)):
        binary_msg += str(flat_arr[i] & 1)
        if len(binary_msg) % 8 == 0 and len(binary_msg) >= delim_len_bits:
            tail = bin_to_message(binary_msg[-delim_len_bits:])
            if tail == DELIMITER:
                return bin_to_message(binary_msg[:-delim_len_bits])
    return None

def hide_data(img_arr, message, method='lsb'):
    """
    Hides a message into an image array.
    Embeds in the central crop region so subsequent CenterCrop operations preserve the payload.
    """
    img = img_arr.copy()
    full_message = message + DELIMITER
    binary_msg = message_to_bin(full_message)
    msg_len = len(binary_msg)
    
    top, left, h, w = _get_crop_bounds(img.shape)
    sub = img[top:top+h, left:left+w]
    channels = 3 if len(sub.shape) == 3 else 1
    
    if method == 'lsb':
        flat_sub = sub.flatten()
        if msg_len > len(flat_sub):
            # Fallback to entire image if message exceeds the central window
            flat_img = img.flatten()
            if msg_len > len(flat_img):
                raise ValueError("Message too large for this image.")
            for i in range(msg_len):
                flat_img[i] = (flat_img[i] & 254) | int(binary_msg[i])
            return flat_img.reshape(img.shape)
            
        for i in range(msg_len):
            flat_sub[i] = (flat_sub[i] & 254) | int(binary_msg[i])
            
        img[top:top+h, left:left+w] = flat_sub.reshape(sub.shape)
        return img
        
    elif method == 'adaptive':
        # Mask out bit 0 to ensure identical edge detection between embedding and extraction
        clean_sub = sub & 0xFE
        gray = cv2.cvtColor(clean_sub, cv2.COLOR_RGB2GRAY) if channels == 3 else clean_sub
        edges = cv2.Canny(gray, 50, 150)
        
        flat_sub = sub.flatten()
        flat_edges = np.repeat(edges.flatten(), channels)
        
        # Indices of textured areas
        textured_idx = np.where(flat_edges > 0)[0]
        
        if msg_len > len(textured_idx):
            # Fallback to standard LSB if not enough textured pixels
            print("Not enough textured pixels, falling back to LSB")
            return hide_data(img_arr, message, method='lsb')
            
        for i in range(msg_len):
            idx = textured_idx[i]
            flat_sub[idx] = (flat_sub[idx] & 254) | int(binary_msg[i])
            
        img[top:top+h, left:left+w] = flat_sub.reshape(sub.shape)
        return img
    else:
        raise ValueError("Unknown embedding method")

def extract_data(img_arr, method='lsb'):
    """
    Extracts message from image array.
    """
    top, left, h, w = _get_crop_bounds(img_arr.shape)
    sub = img_arr[top:top+h, left:left+w]
    channels = 3 if len(sub.shape) == 3 else 1
    
    if method == 'lsb':
        # Check central crop window first
        msg = _extract_from_flat(sub.flatten())
        if msg is not None:
            return msg
        # Fallback to checking full image
        msg = _extract_from_flat(img_arr.flatten())
        if msg is not None:
            return msg
        return "No hidden message found."

    elif method == 'adaptive':
        # Mask bit 0 so extraction exactly mirrors embedding edge detection
        clean_sub = sub & 0xFE
        gray = cv2.cvtColor(clean_sub, cv2.COLOR_RGB2GRAY) if channels == 3 else clean_sub
        edges = cv2.Canny(gray, 50, 150)
        
        flat_sub = sub.flatten()
        flat_edges = np.repeat(edges.flatten(), channels)
        
        textured_idx = np.where(flat_edges > 0)[0]
        if len(textured_idx) > 0:
            extracted_bits_arr = flat_sub[textured_idx]
            msg = _extract_from_flat(extracted_bits_arr)
            if msg is not None:
                return msg
                
        # If not found in edges, fallback to standard LSB in case embedding fell back
        return extract_data(img_arr, method='lsb')
    else:
        raise ValueError("Unknown embedding method")
