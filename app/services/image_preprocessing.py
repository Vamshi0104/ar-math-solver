import io

import cv2
import numpy as np
from PIL import Image, ImageOps


def preprocess_image(image_bytes):
    def gamma_correction(image, gamma=1.2):
        table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    def sharpen(image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)

    def crop_largest_contour(image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            return image[y:y + h, x:x + w]
        return image

    def auto_deskew(image):
        coords = np.column_stack(np.where(image > 0))
        if coords.shape[0] < 10:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) < 1.0:
            return image
        (h, w) = image.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Load and fix orientation
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize large image for faster processing
    h, w = gray.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Denoise and enhance
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = cv2.medianBlur(blurred, 3)
    bilateral = cv2.bilateralFilter(median, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    gamma_corrected = gamma_correction(enhanced)
    sharpened = sharpen(gamma_corrected)
    norm = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)

    # Thresholding
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    # Crop + Deskew
    # cropped = auto_crop(morph)
    cropped = crop_largest_contour(morph)
    deskewed = auto_deskew(cropped)

    # Fallback for overly white or black
    mean_val = np.mean(deskewed)
    if mean_val > 245 or mean_val < 10:
        deskewed = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

    return deskewed
