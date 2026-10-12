import os
import sys
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# Setup model weights directory
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
INDIAN_LPR_DIR = REPO_DIR / "temp" / "Indian_LPR"

if str(INDIAN_LPR_DIR) not in sys.path:
    sys.path.append(str(INDIAN_LPR_DIR))

# Char set for LPRNet
CHARS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "-"
]


def normalize_plate_text(text: str) -> str:
    """Normalize license plate text by stripping non-alphanumeric chars and uppercasing."""
    if not text:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    return cleaned


class ANPRPipeline:
    """Unified Automatic Number Plate Recognition (ANPR) Pipeline.
    Integrates FCOS object detection, Fast-Plate-OCR (ONNX Runtime), and LPRNet.
    """

    def __init__(self):
        self._fcos_model = None
        self._lprnet_model = None
        self._fast_plate_ocr = None
        self._torch_available = False
        self._fcos_initialized = False
        self._fast_plate_initialized = False

        self._init_models()

    def _init_models(self):
        # 1. Initialize Fast-Plate-OCR (ONNX Runtime)
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            self._fast_plate_ocr = LicensePlateRecognizer(hub_ocr_model="cct-xs-v2-global-model")
            self._fast_plate_initialized = True
        except Exception as e:
            print(f"[ANPR] Fast-Plate-OCR initialization notice: {e}")

        # 2. Initialize FCOS & LPRNet (PyTorch)
        try:
            import torch
            self._torch_available = True
            od_weights = INDIAN_LPR_DIR / "weights" / "best_od.pth"
            lpr_weights = INDIAN_LPR_DIR / "weights" / "best_lprnet.pth"

            if od_weights.exists():
                from src.object_detection.model.fcos import FCOSDetector
                from src.object_detection.model.config import DefaultConfig
                fcos = FCOSDetector(mode="inference", config=DefaultConfig).eval()
                fcos.load_state_dict(torch.load(str(od_weights), map_location="cpu"))
                self._fcos_model = fcos
                self._fcos_initialized = True

            if lpr_weights.exists():
                from src.License_Plate_Recognition.model.LPRNet import build_lprnet
                lpr = build_lprnet(lpr_max_len=16, class_num=37).eval()
                lpr.load_state_dict(torch.load(str(lpr_weights), map_location="cpu"))
                self._lprnet_model = lpr
        except Exception as e:
            print(f"[ANPR] PyTorch models initialization notice: {e}")

    def detect_plate_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect license plate bounding box using FCOS, with contour fallback."""
        h, w = img_bgr.shape[:2]

        # Method 1: FCOS Deep Learning Detector
        if self._fcos_initialized and self._torch_available:
            try:
                import torch
                from src.object_detection.utils.utils import preprocess_image as od_preprocess
                img_tensor = od_preprocess(img_bgr)
                with torch.no_grad():
                    out = self._fcos_model(img_tensor)
                    boxes = out[2][0].cpu().numpy().tolist()
                if boxes and len(boxes) > 0:
                    b = boxes[0]
                    pad_w = int((b[2] - b[0]) * 0.05)
                    pad_h = int((b[3] - b[1]) * 0.05)
                    x1 = max(0, int(b[0]) - pad_w)
                    y1 = max(0, int(b[1]) - pad_h)
                    x2 = min(w, int(b[2]) + pad_w)
                    y2 = min(h, int(b[3]) + pad_h)
                    if x2 > x1 and y2 > y1:
                        return (x1, y1, x2, y2)
            except Exception as e:
                print(f"[ANPR] FCOS detection failed, falling back: {e}")

        # Method 2: Classical Computer Vision / Contour Detection Fallback
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(gray, 30, 200)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

            for contour in contours:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
                if len(approx) == 4:
                    x, y, bw, bh = cv2.boundingRect(approx)
                    aspect_ratio = bw / float(bh) if bh > 0 else 0
                    if 1.5 <= aspect_ratio <= 6.0 and bw > 60 and bh > 15:
                        x1 = max(0, x - 5)
                        y1 = max(0, y - 5)
                        x2 = min(w, x + bw + 5)
                        y2 = min(h, y + bh + 5)
                        return (x1, y1, x2, y2)
        except Exception as e:
            print(f"[ANPR] Contour detection failed: {e}")

        # Fallback: Assume central crop if entire image is plate close-up
        return (0, 0, w, h)

    def decode_lprnet(self, crop_bgr: np.ndarray) -> str:
        """Run LPRNet CTC decoding on cropped plate."""
        if not self._lprnet_model or not self._torch_available:
            return ""
        try:
            import torch
            im_lpr = cv2.resize(crop_bgr, (94, 24)).astype("float32")
            im_lpr -= 127.5
            im_lpr *= 0.0078125
            im_lpr = torch.from_numpy(np.transpose(im_lpr, (2, 0, 1))).unsqueeze(0)
            prebs = self._lprnet_model(im_lpr).cpu().detach().numpy()
            preb = prebs[0, :, :]
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
            return "".join(CHARS[j] for j in no_repeat_blank_label)
        except Exception as e:
            print(f"[ANPR] LPRNet decode error: {e}")
            return ""

    def recognize_plate(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """End-to-end license plate recognition on an input vehicle image."""
        bbox = self.detect_plate_bbox(img_bgr)
        if bbox is None:
            return {
                "success": False,
                "plate_number": "",
                "bbox": None,
                "engine": "none",
                "message": "No license plate detected in image."
            }

        x1, y1, x2, y2 = bbox
        crop = img_bgr[y1:y2, x1:x2]

        plate_predictions = []

        # 1. Fast-Plate-OCR (Primary: ONNX Runtime)
        if self._fast_plate_ocr:
            try:
                preds = self._fast_plate_ocr.run(crop)
                if preds and len(preds) > 0:
                    raw_text = preds[0].plate
                    cleaned = normalize_plate_text(raw_text)
                    if cleaned:
                        plate_predictions.append(("fast_plate_ocr", cleaned, 0.95))
            except Exception as e:
                print(f"[ANPR] Fast-Plate-OCR run error: {e}")

        # 2. LPRNet (Consensus / Fallback: PyTorch)
        if self._lprnet_model:
            lpr_text = self.decode_lprnet(crop)
            cleaned_lpr = normalize_plate_text(lpr_text)
            if cleaned_lpr:
                plate_predictions.append(("lprnet", cleaned_lpr, 0.90))

        if not plate_predictions:
            return {
                "success": False,
                "plate_number": "",
                "bbox": [x1, y1, x2, y2],
                "engine": "none",
                "message": "Plate detected but text recognition failed."
            }

        # Select best prediction
        best_engine, best_plate, confidence = plate_predictions[0]

        return {
            "success": True,
            "plate_number": best_plate,
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence,
            "engine": best_engine,
            "consensus": [p[1] for p in plate_predictions],
            "message": f"Plate recognized: {best_plate}"
        }


# Singleton pipeline instance
anpr_pipeline = ANPRPipeline()
