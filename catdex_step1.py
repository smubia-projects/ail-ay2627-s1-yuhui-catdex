"""
Catdex - Step 1: find the cat in a photo.

What this file does:
  1. Uses a pretrained YOLO model to find cats in a photo (no training needed).
  2. Rejects photos with no cat, or with more than one cat.
  3. Crops the cat out of the photo.
  4. Rejects the crop if it is too blurry (OpenCV "Laplacian variance" check).

Install once (in Colab, put a ! in front):
  pip install ultralytics opencv-python
"""

import cv2                      # OpenCV: reads images and does the blur check
import numpy as np              # number crunching (used to stretch contrast in the blur check)
from ultralytics import YOLO    # YOLO: the pretrained object detector

# ---- Settings you can tune -------------------------------------------------
CAT_CLASS_ID = 15        # In the COCO dataset YOLO was trained on, class 15 = "cat"
DOG_CLASS_ID = 16        # ...and class 16 = "dog"
MIN_CONFIDENCE = 0.5     # The MAIN cat must be at least 50% sure
EXTRA_CAT_CONFIDENCE = 0.25  # A SECOND cat only needs to be 25% sure to count as "another cat"
                             # (catches small, dark or half-hidden second cats)
BLUR_THRESHOLD = 30.0   # Crops scoring below this count as "too blurry" (tune on your photos!)
                         # (was 100 before the contrast-stretch fix; but based on the 20 lowest-scored pictures, 30 seems like an OK score,
                         #  but now dark photos score fairly instead of looking "blurry")
PADDING = 0.05           # Add 5% extra space around the box so ears/tail aren't cut off

# Which YOLO size to use. Bigger = more accurate but slower. Downloads automatically the first time.
#   "yolo11n.pt" nano (~6 MB, fastest)   "yolo11s.pt" small (~19 MB, more accurate)
MODEL_NAME = "yolo11s.pt"
model = YOLO(MODEL_NAME)


def blur_score(image):
    """Return a sharpness score. Higher = sharper. Low = blurry."""
    image = cv2.resize(image, (300, 300))              # same size every time, so scores are comparable
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     # colour isn't needed to judge sharpness
    # Stretch the contrast first, so a DARK photo (e.g. a black cat) isn't mistaken for a blurry one:
    # the darkest 2% of pixels become black, the brightest 2% become white, everything else spreads between.
    low, high = np.percentile(gray, (2, 98))
    if high > low:
        gray = np.clip((gray - low) * 255.0 / (high - low), 0, 255)
    return cv2.Laplacian(gray, cv2.CV_64F).var()       # measures how much the edges "jump"; blur smooths them


def find_cat(image_path):
    """
    Look for exactly one cat in the photo.
    Returns a dictionary: {"ok": True/False, "message": ..., "crop": ..., "box": ..., ...}
    """
    image = cv2.imread(image_path)                     # load the photo (OpenCV uses BGR colour order)
    if image is None:
        return {"ok": False, "message": f"Could not open {image_path}"}
    return find_cat_in(image)


def find_cat_in(image):
    """The same check on a photo already loaded as an OpenCV (BGR) image, e.g. one uploaded to the web API."""

    # Run YOLO, looking for cats AND dogs. We use the LOWER confidence level here so faint
    # second cats are found too, then apply the stricter level to the main cat below.
    # agnostic_nms=True: if YOLO draws both a "cat" box and a "dog" box on the SAME animal,
    # keep only the label it is more sure about, so one animal never counts as both.
    results = model(image, classes=[CAT_CLASS_ID, DOG_CLASS_ID], conf=EXTRA_CAT_CONFIDENCE,
                    agnostic_nms=True, verbose=False)
    boxes = results[0].boxes                           # every cat or dog box YOLO found
    labels = boxes.cls.tolist()                        # the class number of each box (15.0 or 16.0)
    scores = boxes.conf.tolist()                       # how sure YOLO is about each box (0 to 1)

    all_cats = [i for i, c in enumerate(labels) if int(c) == CAT_CLASS_ID]   # every cat, even faint ones
    sure_cats = [i for i in all_cats if scores[i] >= MIN_CONFIDENCE]          # cats YOLO is confident about
    dog_count = sum(1 for c, s in zip(labels, scores)                         # dogs it is confident about
                    if int(c) == DOG_CLASS_ID and s >= MIN_CONFIDENCE)

    if len(sure_cats) == 0 and dog_count > 0:
        return {"ok": False, "is_dog": True,
                "message": "That looks like a dog, not a cat! Catdex only collects cats."}
    if len(sure_cats) == 0:
        return {"ok": False, "message": "No cat found. Try another photo."}
    if len(all_cats) > 1:
        return {"ok": False, "message": f"Found {len(all_cats)} cats. Please photograph one cat at a time."}
    if dog_count > 0:
        return {"ok": False, "message": "There's a dog in the photo too. Please photograph the cat on its own."}

    # Exactly one cat and no dogs: get its box corners (x1, y1) = top-left, (x2, y2) = bottom-right
    i = sure_cats[0]
    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
    confidence = float(boxes.conf[i])

    # Add a little padding, but don't go outside the image edges
    h, w = image.shape[:2]
    pad_x, pad_y = (x2 - x1) * PADDING, (y2 - y1) * PADDING
    x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
    x2, y2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))

    crop = image[y1:y2, x1:x2]                         # cut the cat out of the photo
    sharpness = blur_score(crop)

    if sharpness < BLUR_THRESHOLD:
        return {"ok": False, "message": f"Photo looks blurry (score {sharpness:.0f}). Try holding still.",
                "sharpness": sharpness}

    return {"ok": True, "message": "Cat found!", "crop": crop, "box": (x1, y1, x2, y2),
            "confidence": confidence, "sharpness": sharpness}


# ---- Try it out -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "cat.jpg"   # pass a photo path, or use cat.jpg
    result = find_cat(path)
    print(result["message"])
    if result["ok"]:
        print(f"  YOLO confidence: {result['confidence']:.2f}")
        print(f"  Sharpness score: {result['sharpness']:.0f}")
        cv2.imwrite("cat_crop.jpg", result["crop"])          # save the crop so you can look at it
        print("  Saved the cropped cat as cat_crop.jpg")