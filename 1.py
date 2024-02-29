[ WARN:0@0.014] global loadsave.cpp:248 cv::findDecoder imread_('C:     est\mask.png'): can't open/read file: check file path/integrity
Traceback (most recent call last):
  File "c:\test\333.py", line 42, in <module>
    mask_resized = cv2.resize(mask_img, (w, h))
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.9.0) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\resize.cpp:4152: error: (-215:Assertion failed) !ssize.empty() in function 'cv::resize'

import cv2
import numpy as np
import os

# Load the mask image
mask_path = os.path.abspath('C:/test/mask.png')  # Update the path to your mask image
if not os.path.exists(mask_path):
    raise FileNotFoundError(f"Mask image not found at {mask_path}")

mask_img = cv2.imread(mask_path, -1)
if mask_img is None:
    raise ValueError("Unable to load mask image. Check file format and content.")

capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, img = capture.read()

    faces = face_cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5, minSize=(20, 20))

    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = img[y:y+h, x:x+w]

        # Resize the mask to match face size
        mask_resized = cv2.resize(mask_img, (w, h))

        # Create a mask for the face region
        mask = mask_resized[:, :, 3] / 255.0
        img[y:y+h, x:x+w, 0] = img[y:y+h, x:x+w, 0] * (1 - mask) + mask * mask_resized[:, :, 0]
        img[y:y+h, x:x+w, 1] = img[y:y+h, x:x+w, 1] * (1 - mask) + mask * mask_resized[:, :, 1]
        img[y:y+h, x:x+w, 2] = img[y:y+h, x:x+w, 2] * (1 - mask) + mask * mask_resized[:, :, 2]

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('From Camera', img)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
