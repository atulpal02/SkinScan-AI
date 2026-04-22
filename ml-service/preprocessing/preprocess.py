import cv2
import numpy as np
from .enhance import fix_brightness, denoise, sharpen


def resize(image, size=(224, 224)):
    return cv2.resize(image, size)

def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100


def is_dark(image):
    return np.mean(image) < 50


def preprocess(image):
    info = {
        "was_blurry": False,
        "was_dark": False
    }

    if is_blurry(image):
        image = sharpen(image)
        info["was_blurry"] = True

    if is_dark(image):
        image = fix_brightness(image)
        info["was_dark"] = True

    image = denoise(image)
    image = resize(image)

    return image, info