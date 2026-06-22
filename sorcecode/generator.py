import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 384

ANOMALY_TYPES = {
    "corneal_opacity":    None,
    "cataract":           None,
    "glaucoma":           None,
    "retinal_detachment": None,
}


def _detect_iris_pupil(gray: np.ndarray):
    h, w = gray.shape
    cx_def, cy_def = w // 2, h // 2

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    iris_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=h // 2,
        param1=60, param2=28,
        minRadius=h // 5,
        maxRadius=h // 2,
    )

    if iris_circles is not None:
        c = np.round(iris_circles[0][0]).astype(int)
        icx, icy, ir = int(c[0]), int(c[1]), int(c[2])
    else:
        icx, icy, ir = cx_def, cy_def, int(min(h, w) * 0.42)

    x1 = max(0, icx - ir); y1 = max(0, icy - ir)
    x2 = min(w, icx + ir); y2 = min(h, icy + ir)
    roi = blurred[y1:y2, x1:x2]

    pupil_circles = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=1,
        minDist=roi.shape[0],
        param1=50, param2=18,
        minRadius=max(4, ir // 6),
        maxRadius=ir // 2,
    ) if roi.size > 0 else None

    pr = int(ir * 0.30) if pupil_circles is None else int(np.round(pupil_circles[0][0][2]))
    pr = max(4, min(pr, ir - 4))

    return icx, icy, ir, pr


def _soft_circle_mask(h: int, w: int, cx: int, cy: int, r: int,
                      softness: float = 0.12) -> np.ndarray:
    ys, xs = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).astype(np.float32)
    feather = max(1, softness * r)
    return np.clip((r - dist) / feather, 0.0, 1.0)


def _annulus_mask(h, w, cx, cy, r_inner, r_outer, softness=0.10):
    inner = _soft_circle_mask(h, w, cx, cy, r_inner, softness)
    outer = _soft_circle_mask(h, w, cx, cy, r_outer, softness)
    return np.clip(outer - inner, 0, 1)


def _perlin_like(h: int, w: int, scale: float = 0.12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    out = np.zeros((h, w), np.float32)
    weight_sum = 0.0
    for octave in range(4):
        k = max(1, int(max(h, w) * scale * (2 ** octave))) | 1
        layer = cv2.GaussianBlur(noise, (k, k), 0)
        amp = 1.0 / (2 ** octave)
        out += layer * amp
        weight_sum += amp
    out /= weight_sum
    mn, mx = out.min(), out.max()
    if mx > mn:
        out = (out - mn) / (mx - mn)
    return out


def _apply_corneal_opacity(img: np.ndarray, sev: float,
                           icx: int, icy: int, ir: int, pr: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    cornea_r = int(ir * 0.90)
    limbal_r = int(ir * 0.72)

    cornea_mask  = _soft_circle_mask(h, w, icx, icy, cornea_r,  softness=0.08)
    central_mask = _soft_circle_mask(h, w, icx, icy, limbal_r,  softness=0.15)
    pupil_mask   = _soft_circle_mask(h, w, icx, icy, pr,        softness=0.20)

    cornea_mask  = np.clip(cornea_mask  - pupil_mask * 0.7, 0, 1)
    central_mask = np.clip(central_mask - pupil_mask * 0.7, 0, 1)

    noise = _perlin_like(h, w, scale=0.08, seed=7)
    striae = np.sin(np.sqrt((np.mgrid[0:h, 0:w][1] - icx) ** 2 +
                            (np.mgrid[0:h, 0:w][0] - icy) ** 2) * 0.18) * 0.5 + 0.5
    texture = noise * 0.6 + striae.astype(np.float32) * 0.4

    opacity_color = np.array([230, 228, 224], np.float32)
    tinge         = np.array([200, 210, 215], np.float32)

    periph_mask = np.clip(cornea_mask - central_mask, 0, 1)
    periph_alpha = np.clip(sev * 0.45, 0, 0.45)
    central_alpha = np.clip(sev * 0.88, 0, 0.88)

    combined_alpha = central_mask * central_alpha + periph_mask * periph_alpha
    combined_alpha *= (0.75 + texture * 0.25)

    for c in range(3):
        target = opacity_color[c] * 0.7 + tinge[c] * 0.3
        out[:, :, c] = out[:, :, c] * (1 - combined_alpha) + target * combined_alpha

    scatter = np.random.normal(0, 6 * sev, (h, w)).astype(np.float32)
    for c in range(3):
        out[:, :, c] += scatter * cornea_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_cataract(img: np.ndarray, sev: float,
                    icx: int, icy: int, ir: int, pr: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    nuclear_r   = int(pr * 1.4)
    cortical_r  = int(pr * 2.2)

    nuclear_mask  = _soft_circle_mask(h, w, icx, icy, nuclear_r,  0.12)
    cortical_mask = _annulus_mask(h, w, icx, icy, nuclear_r, cortical_r, 0.10)
    psc_mask      = _soft_circle_mask(h, w, icx, icy, int(pr * 0.55), 0.25)

    noise = _perlin_like(h, w, scale=0.06, seed=12)

    angle_map = np.arctan2(np.mgrid[0:h, 0:w][0] - icy,
                           np.mgrid[0:h, 0:w][1] - icx)
    spokes = (np.sin(angle_map * 8) * 0.5 + 0.5).astype(np.float32)
    cortical_texture = noise * 0.5 + spokes * 0.5

    nuc_yellow = np.array([160, 200, 240], np.float32)
    nuc_brown  = np.array([ 90, 130, 190], np.float32)
    nuc_color  = nuc_yellow * (1 - sev * 0.7) + nuc_brown * (sev * 0.7)
    nuc_alpha  = np.clip(sev * 0.85, 0, 0.85)

    cort_color = np.array([195, 200, 205], np.float32)
    cort_alpha = np.clip(sev * 0.65 * cortical_texture, 0, 0.65)

    psc_alpha = np.clip(sev * 0.75, 0, 0.75)

    for c in range(3):
        out[:, :, c] = (out[:, :, c] * (1 - nuclear_mask * nuc_alpha)
                        + nuc_color[c] * nuclear_mask * nuc_alpha)
        out[:, :, c] = (out[:, :, c] * (1 - cortical_mask * cort_alpha)
                        + cort_color[c] * cortical_mask * cort_alpha)
        out[:, :, c] = (out[:, :, c] * (1 - psc_mask * psc_alpha)
                        + 245 * psc_mask * psc_alpha)

    cry = np.random.normal(0, 10 * sev, (h, w)).astype(np.float32)
    lens_mask = _soft_circle_mask(h, w, icx, icy, cortical_r, 0.10)
    for c in range(3):
        out[:, :, c] += cry * lens_mask
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_glaucoma(img: np.ndarray, sev: float,
                    icx: int, icy: int, ir: int, pr: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    cornea_mask  = _soft_circle_mask(h, w, icx, icy, int(ir * 0.95), 0.08)
    periph_mask  = _annulus_mask(h, w, icx, icy, int(ir * 0.55), int(ir * 0.95), 0.12)
    pupil_mask   = _soft_circle_mask(h, w, icx, icy, pr, 0.15)

    oedema_color = np.array([175, 190, 175], np.float32)
    oedema_alpha = np.clip(sev * 0.55, 0, 0.55)
    oedema_mask  = np.clip(cornea_mask - pupil_mask * 0.5, 0, 1)

    ys = np.mgrid[0:h, 0:w][0].astype(np.float32)
    striae = (np.sin((ys - icy) * 0.35) * 0.5 + 0.5) * 0.25
    oedema_alpha_tex = oedema_alpha * (1 + striae * sev)

    for c in range(3):
        out[:, :, c] = (out[:, :, c] * (1 - oedema_mask * oedema_alpha_tex)
                        + oedema_color[c] * oedema_mask * oedema_alpha_tex)

    dark_alpha = np.clip(sev * 0.70, 0, 0.70)
    for c in range(3):
        out[:, :, c] *= (1 - periph_mask * dark_alpha)

    iris_mask = _soft_circle_mask(h, w, icx, icy, ir, 0.10)
    tint_shift = np.array([8, 14, -6], np.float32)
    for c in range(3):
        out[:, :, c] += tint_shift[c] * sev * iris_mask

    blur_k = max(1, int(sev * 9) | 1)
    blurred = cv2.GaussianBlur(out.astype(np.uint8), (blur_k, blur_k), 0).astype(np.float32)
    out = out * (1 - sev * 0.30) + blurred * (sev * 0.30)

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_retinal_detachment(img: np.ndarray, sev: float,
                               icx: int, icy: int, ir: int, pr: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)

    scleral_mask  = _annulus_mask(h, w, icx, icy, int(ir * 0.92), int(ir * 1.55), 0.10)
    limbal_mask   = _annulus_mask(h, w, icx, icy, int(ir * 0.85), int(ir * 1.10), 0.08)
    iris_mask     = _soft_circle_mask(h, w, icx, icy, ir, 0.10)

    noise = _perlin_like(h, w, scale=0.15, seed=3)

    injection_color = np.array([30, 30, 200], np.float32)
    injection_alpha = np.clip(sev * 0.75 * (0.6 + noise * 0.4), 0, 0.75)
    for c in range(3):
        out[:, :, c] = (out[:, :, c] * (1 - scleral_mask * injection_alpha)
                        + injection_color[c] * scleral_mask * injection_alpha)

    heme_color = np.array([20, 20, 160], np.float32)
    heme_noise = _perlin_like(h, w, scale=0.20, seed=17)
    heme_thresh = np.clip(1.0 - sev * 0.9, 0.2, 0.85)
    heme_patch  = (heme_noise > heme_thresh).astype(np.float32) * scleral_mask
    heme_alpha  = np.clip(sev * 0.80, 0, 0.80) * heme_patch
    for c in range(3):
        out[:, :, c] = (out[:, :, c] * (1 - heme_alpha)
                        + heme_color[c] * heme_alpha)

    limbal_color = np.array([40, 60, 220], np.float32)
    limbal_alpha = np.clip(sev * 0.65, 0, 0.65)
    for c in range(3):
        out[:, :, c] = (out[:, :, c] * (1 - limbal_mask * limbal_alpha)
                        + limbal_color[c] * limbal_mask * limbal_alpha)

    red_tint = np.array([-10, -10, 30], np.float32)
    for c in range(3):
        out[:, :, c] += red_tint[c] * sev * iris_mask

    fold_strength = sev * 5.0
    ys_f, xs_f = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs_f - icx; dy = ys_f - icy
    dist_r = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    angle_r = np.arctan2(dy, dx)
    ripple = np.sin(dist_r * 0.06 + angle_r * 4) * fold_strength
    map_x = np.clip(xs_f + ripple * dx / dist_r, 0, w - 1).astype(np.float32)
    map_y = np.clip(ys_f + ripple * dy / dist_r, 0, h - 1).astype(np.float32)
    out = cv2.remap(out.astype(np.uint8), map_x, map_y,
                    cv2.INTER_LINEAR).astype(np.float32)

    bk = max(1, int(sev * 5) | 1)
    blurred = cv2.GaussianBlur(out.astype(np.uint8), (bk, bk), 0).astype(np.float32)
    chemosis_zone = np.clip(scleral_mask + limbal_mask, 0, 1)[:, :, np.newaxis]
    out = out * (1 - chemosis_zone * sev * 0.25) + blurred * (chemosis_zone * sev * 0.25)

    return np.clip(out, 0, 255).astype(np.uint8)


_ANOMALY_FN = {
    "corneal_opacity":    _apply_corneal_opacity,
    "cataract":           _apply_cataract,
    "glaucoma":           _apply_glaucoma,
    "retinal_detachment": _apply_retinal_detachment,
}

ANOMALY_TYPES = _ANOMALY_FN


def init_pipeline():
    pass


def generate_synthetic_biometric(
    anomaly_type: str = "none",
    severity: float = 0.0,
    size: int = IMAGE_SIZE,
    base_image: Image.Image = None,
) -> Image.Image:
    if base_image is None:
        return Image.new("RGB", (size, size), color=(30, 30, 30))

    base_image = base_image.convert("RGB").resize((size, size), Image.LANCZOS)

    if anomaly_type == "none" or severity <= 0:
        return base_image

    fn = _ANOMALY_FN.get(anomaly_type)
    if fn is None:
        return base_image

    bgr  = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    icx, icy, ir, pr = _detect_iris_pupil(gray)

    result_bgr = fn(bgr, float(severity), icx, icy, ir, pr)
    return Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
