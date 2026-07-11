"""Individual photo extraction (OpenCV Haar cascade)"""

import os
import cv2

PHOTO_DIR = "photos"
os.makedirs(PHOTO_DIR, exist_ok=True)

_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def _detect_and_crop(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.35 * h)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    out = os.path.join(
        PHOTO_DIR, os.path.splitext(os.path.basename(image_path))[
            0] + "_photo.jpg"
    )
    cv2.imwrite(out, img[y1:y2, x1:x2])
    return out


def extract_photo(image_paths):
    """Try each page (photo is on the front side); return crop path or None"""
    for p in image_paths:
        photo = _detect_and_crop(p)
        if photo:
            return photo
    return None
