import os
import imutils
import hashlib
import cv2
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import pandas as pd

def show_images(images):
    for i, img in enumerate(images):
        cv2.imshow("image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def aspect_ratio(width, height):
    return height / width if width != 0 else None

def measure_image(img_path):
    # Generate a unique_id based on the image path
    unique_id = hashlib.sha1(img_path.encode()).hexdigest()

    # Read image and preprocess
    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    if len(cnts) > 0:
        ref_object = cnts[0]
        box = cv2.minAreaRect(ref_object)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        dist_in_pixel = euclidean(tl, tr)
        dist_in_cm = 2
        pixel_per_cm = dist_in_pixel/dist_in_cm

        # Calculate height and width
        (x, y, w, h) = cv2.boundingRect(cnts[0])
        height = h / pixel_per_cm
        width = w / pixel_per_cm

        aspect_ratio_value = aspect_ratio(width, height)

        return unique_id, height, width, aspect_ratio_value
    else:
        print(f"Unable to measure image: {img_path}")
        return unique_id, None, None, None

folder_path = "Rice_Dataset/Paijam"
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
results = []

for img_path in image_files:
    img_full_path = os.path.join(folder_path, img_path)
    unique_id, height, width, aspect_ratio_value = measure_image(img_full_path)
    results.append({
        'unique_id': unique_id,
        'height': height,
        'width': width,
        'aspect_ratio': aspect_ratio_value,
        'image_path': img_full_path
    })

df = pd.DataFrame(results)
df.to_excel('Paijam_rice_image_measurements.xlsx', index=False)