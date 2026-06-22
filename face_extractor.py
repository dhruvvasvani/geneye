import cv2
import numpy as np
from PIL import Image

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

_dummy = np.zeros((64, 64), dtype=np.uint8)
FACE_CASCADE.detectMultiScale(_dummy)
EYE_CASCADE.detectMultiScale(_dummy)


def extract_eye_region(image_np: np.ndarray, target_size: int = 384) -> Image.Image:
    h, w = image_np.shape[:2]

    FACE_DET_DIM = 320
    scale_f = min(1.0, FACE_DET_DIM / max(h, w))
    face_img = cv2.resize(image_np, (int(w * scale_f), int(h * scale_f))) if scale_f < 1.0 else image_np

    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray_face, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30)
    )

    offset_x = offset_y = 0
    scale_e = scale_f

    if len(faces) == 0:
        face_roi_gray    = gray_face
        fallback_roi_bgr = image_np[: h // 2, :]
    else:
        fx, fy, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        ox, oy = int(fx / scale_f), int(fy / scale_f)
        ow, oh = int(fw / scale_f), int(fh / scale_f)

        fallback_roi_bgr = image_np[oy : oy + oh // 2, ox : ox + ow]
        face_roi_orig    = image_np[oy : oy + oh, ox : ox + ow]
        rh, rw = face_roi_orig.shape[:2]

        EYE_DET_DIM = 256
        scale_e = min(1.0, EYE_DET_DIM / max(rh, rw))
        face_roi_small = (
            cv2.resize(face_roi_orig, (int(rw * scale_e), int(rh * scale_e)))
            if scale_e < 1.0
            else face_roi_orig
        )
        face_roi_gray = cv2.cvtColor(face_roi_small, cv2.COLOR_BGR2GRAY)
        offset_x, offset_y = ox, oy

    eyes = EYE_CASCADE.detectMultiScale(
        face_roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15)
    )

    if len(eyes) == 0:
        eye_crop = fallback_roi_bgr
    else:
        ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[0]
        ex_o = offset_x + int(ex / scale_e)
        ey_o = offset_y + int(ey / scale_e)
        ew_o = int(ew / scale_e)
        eh_o = int(eh / scale_e)

        pad = int(ew_o * 0.35)
        x1 = max(0, ex_o - pad);     y1 = max(0, ey_o - pad)
        x2 = min(w, ex_o + ew_o + pad); y2 = min(h, ey_o + eh_o + pad)
        eye_crop = image_np[y1:y2, x1:x2]

    if eye_crop.size == 0:
        eye_crop = image_np

    eye_resized = cv2.resize(eye_crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(cv2.cvtColor(eye_resized, cv2.COLOR_BGR2RGB))
