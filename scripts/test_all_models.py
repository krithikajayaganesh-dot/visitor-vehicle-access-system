import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import pytesseract

# Try importing easyocr
try:
    import easyocr
    easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
except ImportError:
    easyocr_reader = None
    print("EasyOCR not installed. Run 'pip install easyocr'")

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDIAN_LPR_DIR = os.path.join(BASE_DIR, "temp", "Indian_LPR")
INPUT_DIR = os.path.join(BASE_DIR, "temp", "Indian-Number-Plate-Recognition-master", "inputs")

# Add Indian_LPR to path for imports
if INDIAN_LPR_DIR not in sys.path:
    sys.path.append(INDIAN_LPR_DIR)

# Indian_LPR Imports
from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
from src.object_detection.utils.utils import preprocess_image as od_preprocess
from src.semantic_segmentation.models.hrnet import hrnet
from src.semantic_segmentation.utils.util import (
    plate_locate, get_score_and_class_from_prediction,
    upsample_coordinates, convert_coordinates_to_bbox,
    get_warped_plates, preprocess_image as ss_preprocess
)
from src.License_Plate_Recognition.model.LPRNet import build_lprnet

CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-"
]

def Greedy_Decode_inference(Net, images):
    if torch.cuda.is_available():
        images = images.cuda()
    prebs = Net(images).cpu().detach().numpy()
    preb_labels = []
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = [np.argmax(preb[:, j], axis=0) for j in range(preb.shape[1])]
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
    return ["".join(CHARS[j] for j in label) for label in preb_labels]


def load_fcos():
    model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
    weights_path = os.path.join(INDIAN_LPR_DIR, "weights", "best_od.pth")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    if torch.cuda.is_available(): model = model.cuda()
    return model

def load_hrnet():
    model = hrnet().eval()
    weights_path = os.path.join(INDIAN_LPR_DIR, "weights", "best_semantic.pth")
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))["state_dict"]
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    if torch.cuda.is_available(): model = model.cuda()
    return model

def load_lprnet():
    model = build_lprnet(lpr_max_len=16, class_num=37).eval()
    weights_path = os.path.join(INDIAN_LPR_DIR, "weights", "best_lprnet.pth")
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    if torch.cuda.is_available(): model = model.cuda()
    return model

# ─── OCR Wrappers ───
def ocr_tesseract(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return pytesseract.image_to_string(thresh, config=config).strip()

def ocr_easyocr(crop):
    if not easyocr_reader: return "N/A"
    results = easyocr_reader.readtext(crop, detail=0, paragraph=True)
    return " ".join(results).replace("-", "").strip().upper()

def ocr_lprnet(lprnet, crop):
    im_lpr = cv2.resize(crop, (94, 24)).astype("float32")
    im_lpr -= 127.5
    im_lpr *= 0.0078125
    im_lpr = torch.from_numpy(np.transpose(im_lpr, (2, 0, 1))).unsqueeze(0)
    if torch.cuda.is_available(): im_lpr = im_lpr.cuda()
    return Greedy_Decode_inference(lprnet, im_lpr)[0]

# ─── Detection Methods ───
def detect_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(gray, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 1.5 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                # Add slight padding
                x1 = max(0, x - 5)
                y1 = max(0, y - 5)
                x2 = min(img.shape[1], x + w + 5)
                y2 = min(img.shape[0], y + h + 5)
                return img[y1:y2, x1:x2]
    return None

def detect_fcos(img, od_model):
    img_tensor = od_preprocess(img)
    if torch.cuda.is_available(): img_tensor = img_tensor.cuda()
    with torch.no_grad():
        out = od_model(img_tensor)
        boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in out[2][0].cpu().numpy().tolist()]
    
    if not boxes: return None
    b = boxes[0] # Use highest confidence
    h, w = img.shape[:2]
    # Add 5% padding to prevent clipping characters (like the '13' in MH12JC2813)
    pad_w = int((b[2] - b[0]) * 0.05)
    pad_h = int((b[3] - b[1]) * 0.05)
    
    x1 = max(0, b[0] - pad_w)
    y1 = max(0, b[1] - pad_h)
    x2 = min(w, b[2] + pad_w)
    y2 = min(h, b[3] + pad_h)
    
    return img[y1:y2, x1:x2]

def detect_hrnet(img, semantic_model):
    img_tensor = ss_preprocess(img)
    if torch.cuda.is_available(): img_tensor = img_tensor.cuda()
    with torch.no_grad():
        out = semantic_model(img_tensor, (img_tensor.shape[2], img_tensor.shape[3]))
        prediction_softmax = torch.nn.Softmax(dim=1)(out["output"])
        out_argmax = torch.argmax(out["output"], dim=1).detach().cpu().squeeze(dim=0).numpy().astype(np.uint8)
        coordinates, _ = plate_locate(out_argmax)
        scores = get_score_and_class_from_prediction(out_argmax, prediction_softmax, coordinates)
        pred_boxes = convert_coordinates_to_bbox(coordinates)
        
        valid_coords = [c for score, c in zip(scores, coordinates) if score[0] > 0.5]
        if not valid_coords: return None
        
        coordinates, _ = upsample_coordinates(valid_coords, out_argmax.shape, img.shape)
        plate_images = get_warped_plates(img, coordinates)
        return plate_images[0] if plate_images else None

def main():
    print("=" * 110)
    print("  ULTIMATE ANPR BENCHMARK — ALL MODELS")
    print("=" * 110)
    
    print("Loading models...")
    od_model = load_fcos()
    semantic_model = load_hrnet()
    lprnet = load_lprnet()
    print("Models loaded successfully!\n")
    
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png'))])
    
    # Store results for the final table
    results = []
    
    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        
        print(f"Processing: {img_name}")
        
        # Crops
        crop_contour = detect_contours(img)
        crop_fcos = detect_fcos(img, od_model)
        crop_hrnet = detect_hrnet(img, semantic_model)
        
        res = {"Image": img_name}
        
        # 1. Contour + Tesseract
        res["Contour+Tess"] = ocr_tesseract(crop_contour) if crop_contour is not None else "FAILED"
        
        # 2. Contour + EasyOCR
        res["Contour+Easy"] = ocr_easyocr(crop_contour) if crop_contour is not None else "FAILED"
        
        # 3. FCOS + LPRNet
        res["FCOS+LPRNet"] = ocr_lprnet(lprnet, crop_fcos) if crop_fcos is not None else "FAILED"
        
        # 4. HRNet + LPRNet
        res["HRNet+LPRNet"] = ocr_lprnet(lprnet, crop_hrnet) if crop_hrnet is not None else "FAILED"
        
        # 5. FCOS + Tesseract (Hybrid)
        res["FCOS+Tess"] = ocr_tesseract(crop_fcos) if crop_fcos is not None else "FAILED"
        
        # 6. FCOS + EasyOCR (Hybrid)
        res["FCOS+Easy"] = ocr_easyocr(crop_fcos) if crop_fcos is not None else "FAILED"
        
        # Clean EasyOCR brackets output (like (DL2CAY3180] -> DL2CAY3180)
        res["FCOS+Easy"] = ''.join(c for c in res["FCOS+Easy"] if c.isalnum() or c.isspace())
        res["Contour+Easy"] = ''.join(c for c in res["Contour+Easy"] if c.isalnum() or c.isspace())
        
        results.append(res)
        
    # Print Table
    print("\n" + "=" * 125)
    print(f"{'Image':<10} | {'Contour+Tess':<15} | {'Contour+EasyOCR':<18} | {'FCOS+LPRNet':<15} | {'HRNet+LPRNet':<15} | {'FCOS+Tess (Hybrid)':<20} | {'FCOS+EasyOCR (Hybrid)':<20}")
    print("-" * 125)
    for r in results:
        print(f"{r['Image']:<10} | {r['Contour+Tess']:<15} | {r['Contour+Easy']:<18} | {r['FCOS+LPRNet']:<15} | {r['HRNet+LPRNet']:<15} | {r['FCOS+Tess']:<20} | {r['FCOS+Easy']:<20}")
    print("=" * 125)
    print("\nBenchmark complete!")

if __name__ == "__main__":
    main()
