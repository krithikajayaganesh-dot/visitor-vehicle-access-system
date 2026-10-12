import base64
import cv2
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Optional

from database import get_db
from models import VisitorPass, AccessLog, PassStatus, LogAction, HardwareAPIKey
from auth import get_hardware_or_guard
from anpr_service import anpr_pipeline, normalize_plate_text

router = APIRouter(prefix="/api/anpr", tags=["ANPR"])


@router.get("/status")
def get_anpr_status():
    """Diagnostic endpoint returning status of ANPR engines."""
    return {
        "status": "online",
        "fcos_detector": anpr_pipeline._fcos_initialized,
        "fast_plate_ocr_onnx": anpr_pipeline._fast_plate_initialized,
        "lprnet_ocr": anpr_pipeline._lprnet_model is not None,
        "torch_available": anpr_pipeline._torch_available,
        "engines": ["FCOS", "Fast-Plate-OCR (ONNX)", "LPRNet"],
    }


@router.post("/scan")
async def scan_vehicle_plate(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    action: str = Form("entry"),
    db: Session = Depends(get_db),
    user_or_hw: dict = Depends(get_hardware_or_guard),
):
    """Scan vehicle license plate via camera capture or snapshot upload.
    Automatically checks against registered visitor passes and verifies access.
    """
    img_bgr = None

    if file:
        file_bytes = await file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        decoded = base64.b64decode(image_base64)
        np_arr = np.frombuffer(decoded, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image data provided.")

    # Run ML recognition
    recog_result = anpr_pipeline.recognize_plate(img_bgr)
    if not recog_result["success"] or not recog_result["plate_number"]:
        return {
            "verified": False,
            "gate_open": False,
            "plate_number": "",
            "message": "❌ No license plate recognized. Please adjust camera or perform manual verification.",
            "anpr_debug": recog_result,
        }

    detected_plate = recog_result["plate_number"]
    normalized_detected = normalize_plate_text(detected_plate)

    # Search database for visitor pass matching vehicle_number
    # Fetch all active / valid passes and compare normalized vehicle numbers
    passes = db.query(VisitorPass).all()
    matched_pass = None

    for p in passes:
        if p.vehicle_number:
            norm_p = normalize_plate_text(p.vehicle_number)
            # Check exact match or substring match (e.g., state code prefix matching)
            if norm_p and (norm_p == normalized_detected or norm_p in normalized_detected or normalized_detected in norm_p):
                matched_pass = p
                break

    # Determine actor
    scanned_by_id = user_or_hw["user"].id if user_or_hw["type"] == "user" else 1

    if not matched_pass:
        return {
            "verified": False,
            "gate_open": False,
            "plate_number": detected_plate,
            "normalized_plate": normalized_detected,
            "message": f"⚠️ Plate detected ({detected_plate}), but no registered visitor pass found. Use Walk-in Entry.",
            "visitor_pass": None,
            "anpr_debug": recog_result,
        }

    # Pass found — evaluate state rules
    now = datetime.utcnow()
    if now > matched_pass.expires_at:
        matched_pass.status = PassStatus.expired
        db.commit()
        return {
            "verified": False,
            "gate_open": False,
            "plate_number": detected_plate,
            "message": f"❌ Pass Expired for vehicle {matched_pass.vehicle_number} ({matched_pass.visitor_name})",
            "visitor_pass": {
                "id": matched_pass.id,
                "visitor_name": matched_pass.visitor_name,
                "vehicle_number": matched_pass.vehicle_number,
                "status": matched_pass.status,
            },
            "anpr_debug": recog_result,
        }

    # Anti-passback & entry/exit state validation
    if action == "entry":
        if matched_pass.status in [PassStatus.in_campus, PassStatus.used]:
            return {
                "verified": False,
                "gate_open": False,
                "plate_number": detected_plate,
                "message": f"❌ Entry Denied: Vehicle {matched_pass.vehicle_number} is ALREADY inside campus (anti-passback rule).",
                "visitor_pass": {
                    "id": matched_pass.id,
                    "visitor_name": matched_pass.visitor_name,
                    "status": matched_pass.status,
                },
                "anpr_debug": recog_result,
            }
        matched_pass.status = PassStatus.in_campus

    elif action == "exit":
        if matched_pass.status in [PassStatus.not_inside, PassStatus.approved, PassStatus.pending]:
            return {
                "verified": False,
                "gate_open": False,
                "plate_number": detected_plate,
                "message": f"❌ Exit Denied: Vehicle {matched_pass.vehicle_number} has not entered campus yet.",
                "visitor_pass": {
                    "id": matched_pass.id,
                    "visitor_name": matched_pass.visitor_name,
                    "status": matched_pass.status,
                },
                "anpr_debug": recog_result,
            }
        matched_pass.status = PassStatus.exited

    # Log access
    access_log = AccessLog(
        visitor_pass_id=matched_pass.id,
        scanned_by=scanned_by_id,
        action=action,
    )
    db.add(access_log)
    db.commit()
    db.refresh(matched_pass)

    return {
        "verified": True,
        "gate_open": True,
        "action": action,
        "plate_number": detected_plate,
        "message": f"✅ Vehicle {action.upper()} authorized for {matched_pass.visitor_name} ({matched_pass.vehicle_number})",
        "visitor_pass": {
            "id": matched_pass.id,
            "visitor_name": matched_pass.visitor_name,
            "vehicle_number": matched_pass.vehicle_number,
            "purpose": matched_pass.purpose,
            "status": matched_pass.status,
        },
        "anpr_debug": recog_result,
    }
