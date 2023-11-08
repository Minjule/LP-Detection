from ultralytics import YOLO
import cv2
import numpy as np
from util import read_license_plate
import os
import easyocr
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

license_plate_detector = YOLO('runs\\detect\\train8\\weights\\best.pt')
image = cv2.imread("testimages\\1cropped.jpg")

if __name__ == '__main__':

    detections = license_plate_detector(image, device=0, stream = True)

    for idx, detection in enumerate(detections):
        boxes = detection.boxes.cpu().numpy()
        x1, y1, x2, y2, score, class_id = boxes.data[0]

        license_plate_crop = image[int(y1):int(y2), int(x1): int(x2), :]
        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

    rect_start = (x1, y1)
    rect_end = (x2, y2)

    #image = cv2.rectangle(image, rect_start, rect_end, (0, 0, 255), 2)
    #image = cv2.putText(image, license_plate_text, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

    cv2.imshow('Result Image', license_plate_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()