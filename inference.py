import os
import io
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from model import SkribblCNN

CLASSES = ['apple', 'car', 'dog', 'tree', 'clock']

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "models", "skribbl_cnn_best.pth")

device = torch.device('cpu')
model = SkribblCNN(num_classes=len(CLASSES)).to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Model loaded successfully from: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model.eval()

def preprocess_canvas_image(image_bytes: bytes) -> np.ndarray:
    """
    Calibrated Preprocessing:
    1. Handles RGBA / Transparent canvas backgrounds.
    2. Direct bilinear downsampling to 28x28 (preserves natural aspect ratios).
    3. Contrast boost / thresholding to prevent faint strokes from vanishing.
    """
    # 1. Open raw image
    raw_img = Image.open(io.BytesIO(image_bytes))
    
    # 2. Fix Transparency (Composite over pure white)
    white_bg = Image.new("RGBA", raw_img.size, (255, 255, 255, 255))
    if raw_img.mode in ('RGBA', 'LA') or (raw_img.mode == 'P' and 'transparency' in raw_img.info):
        raw_img = Image.alpha_composite(white_bg, raw_img.convert('RGBA'))
    
    # Convert to Grayscale
    gray_img = raw_img.convert('L')
    
    # 3. Direct Resize 280x280 -> 28x28 (Matches what we had in Colab)
    resized_img = gray_img.resize((28, 28), Image.Resampling.BILINEAR)
    
    # 4. Invert (Black strokes on white canvas -> White strokes on black background)
    img_array = 255.0 - np.array(resized_img, dtype=np.float32)
    
    # 5. CONTRAST BOOST & THRESHOLDING:
    # Amplify faint/thin strokes so they match QuickDraw density
    img_array = img_array / 255.0
    # Boost any pixel with a stroke signal (> 0.15) to be crisper
    img_array = np.where(img_array > 0.15, img_array * 1.5, img_array)
    img_array = np.clip(img_array, 0.0, 1.0)
    
    return img_array.astype(np.float32)

def predict_drawing_bytes(image_bytes: bytes) -> dict:
    try:
        img_array = preprocess_canvas_image(image_bytes)
        
        # Format Tensor: [1, 1, 28, 28]
        tensor_img = torch.tensor(img_array).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(tensor_img)
            probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
            
        top_idx = int(np.argmax(probabilities))
        
        return {
            "success": True,
            "prediction": CLASSES[top_idx],
            "confidence": float(probabilities[top_idx]),
            "probabilities": {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        }
    except Exception as e:
        return {"success": False, "error": str(e)}