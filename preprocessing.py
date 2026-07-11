"""Adaptive image preprocessing for Aadhaar/PAN card images.

Handles the defects that actually occur on ID-card inputs (jpg/jpeg/png,
and PDF pages already converted to images):

  Correctable  : global brightness (gamma), low contrast / shadows (CLAHE),
                 noise / compression artifacts (denoise), mild blur
                 (unsharp mask), skew (rotation correction)
  Detect-only  : severe blur, glare patches, possible occlusion
                 -> returned as flags; text hidden by glare/occlusion cannot
                    be restored, and must never be "invented" for ID data.

Design: diagnose first, then apply ONLY the fixes for detected defects.
Clean images pass through untouched (over-processing hurts OCR).

Fix order matters: deskew -> denoise -> lighting -> sharpen (last, so it
does not amplify noise).
"""

import cv2
import numpy as np

# ---------------- Thresholds (tune against your sample set) ----------------

BLUR_REJECT = 12
BLUR_MILD = 100
DARK_MEAN = 70
BRIGHT_MEAN = 225
LOW_CONTRAST_SPREAD = 60
NOISE_SIGMA = 8.0
SKEW_MIN_DEG = 1.5
GLARE_FRAC = 0.02


# ---------------- Diagnosis ----------------

def diagnose(img):
    """Return dict of detected issues and raw quality scores."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    issues = []
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_REJECT:
        issues.append("severe_blur")
    elif blur_score < BLUR_MILD:
        issues.append("blur")
    mean = gray.mean()
    p_lo, p_hi = np.percentile(gray, [0.5, 99.5])
    spread = float(p_hi - p_lo)
    if mean < DARK_MEAN:
        issues.append("dark")
    elif mean > BRIGHT_MEAN and spread < LOW_CONTRAST_SPREAD:
        issues.append("bright")
    if spread < LOW_CONTRAST_SPREAD:
        issues.append("low_contrast")
    if _estimate_noise(gray) > NOISE_SIGMA:
        issues.append("noise")
    angle = _estimate_skew(gray)
    if abs(angle) > SKEW_MIN_DEG:
        issues.append("skew")
    if _glare_fraction(gray) > GLARE_FRAC:
        issues.append("glare")
    return {"issues": issues, "blur_score": blur_score,
            "brightness": mean, "skew_angle": angle}


def _estimate_noise(gray):
    """Fast noise estimate: median absolute deviation of the Laplacian."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.median(np.abs(lap - np.median(lap)))) * 1.4826


def _estimate_skew(gray):
    """Estimate dominant text-line angle via minAreaRect on thresholded pixels.
    Returns angle in degrees (positive = counter-clockwise tilt).
    Only trustworthy for small tilts (< ~20 deg), which is the common case
    for crooked scans/photos.
    """
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return float(angle)


def _glare_fraction(gray):
    """Fraction of the image that is blown-out white glare blobs."""
    _, bright = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    return float(np.count_nonzero(bright)) / bright.size


# ---------------- Targeted fixes ----------------

def _fix_skew(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _fix_noise(img):
    """Mild non-local-means denoise (WhatsApp-compression artifacts etc.)"""
    return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)


def _fix_gamma(img, gamma):
    """gamma < 1 brightens, gamma > 1 darkens"""
    table = np.array(
        [(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def _fix_contrast(img):
    """CLAHE on LAB L-channel: fixes shadows/faded print without blowout"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lum, ca, cb = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lum = clahe.apply(lum)
    return cv2.cvtColor(cv2.merge((lum, ca, cb)), cv2.COLOR_LAB2BGR)


def _fix_blur(img):
    """Unsharp mask: sharpen text edges. Always applied LAST"""
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.8, gaussian, -0.8, 0)


# ---------------- Dispatcher ----------------

def preprocess(image_path):
    """Diagnose and repair one image in place.
    Returns dict:
      applied  : list of corrections performed
      warnings : detect-only issues ('glare', 'severe_blur') for the UI
      rejected : True if the image is unrecoverable (severe blur)
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"applied": [], "warnings": [
            "unreadable_file"], "rejected": True}
    diag = diagnose(img)
    issues = diag["issues"]
    applied, warnings = [], []
    if "severe_blur" in issues:
        return {"applied": [], "warnings": ["severe_blur"], "rejected": True}
    if "glare" in issues:
        warnings.append("glare")
    if "skew" in issues:
        img = _fix_skew(img, diag["skew_angle"])
        applied.append("deskew")
    if "noise" in issues:
        img = _fix_noise(img)
        applied.append("denoise")
    if "dark" in issues:
        img = _fix_gamma(img, 0.6)
        applied.append("brighten")
    elif "bright" in issues:
        img = _fix_gamma(img, 1.5)
        applied.append("darken")
    if "low_contrast" in issues:
        img = _fix_contrast(img)
        applied.append("clahe")
    if "blur" in issues:
        img = _fix_blur(img)
        applied.append("sharpen")
    if applied:
        cv2.imwrite(image_path, img)
    return {"applied": applied, "warnings": warnings, "rejected": False}
