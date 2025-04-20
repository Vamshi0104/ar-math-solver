import io

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.filters import sobel
from skimage.measure import shannon_entropy


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

    def is_image_clean(image):
        entropy = shannon_entropy(image)
        sobel_edges = sobel(image.astype(float) / 255.0)
        contrast = sobel_edges.std()
        brightness = np.mean(image)

        # Stroke density
        thresh = 200
        symbol_pixels = np.sum(image < thresh)
        density = symbol_pixels / image.size

        black_pct = np.sum(image < 50) / image.size
        white_pct = np.sum(image > 245) / image.size

        print(f"Entropy: {entropy:.2f} | Contrast: {contrast:.2f} | Brightness: {brightness:.2f}")
        print(f"Stroke Density: {density:.2%} |  Black: {black_pct:.2%} |  White: {white_pct:.2%}")

        is_clean = (
                entropy > 4.5 and
                0.05 < contrast < 0.25 and
                100 < brightness < 200 and
                0.02 < density < 0.25 and
                black_pct < 0.10 and
                white_pct < 0.50
        )

        print("Clean Image?", "YES" if is_clean else "NO")
        return is_clean

    def full_preprocess(gray):
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        median = cv2.medianBlur(blurred, 3)
        bilateral = cv2.bilateralFilter(median, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)
        gamma_corrected = gamma_correction(enhanced)
        sharpened = sharpen(gamma_corrected)
        norm = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        cropped = crop_largest_contour(morph)
        deskewed = auto_deskew(cropped)

        mean_val = np.mean(deskewed)
        if mean_val > 245 or mean_val < 10:
            print("Post-processing too light/dark, applying adaptive threshold...")
            deskewed = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
        return deskewed

    # Load and prepare
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.exif_transpose(image)
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize if large
    h, w = gray.shape
    if max(h, w) > 1500:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Run appropriate pipeline
    if is_image_clean(gray):
        print("Using original grayscale for OCR.")
        deskewed = auto_deskew(gray)
    else:
        print("Using full enhancement pipeline.")
        deskewed = full_preprocess(gray)

    return deskewed
