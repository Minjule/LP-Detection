from ultralytics import YOLO
import cv2
import numpy as np
from util import read_license_plate
import os
import easyocr
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('runs\\detect\\train8\\weights\\best.pt')
image = cv2.imread("testimages\\17.jpg")
vehicles = [2, 3, 5, 7]

if __name__ == '__main__':

    cars = coco_model(image, device=0, stream = True)
    cars_ = []
    for car in cars:
        cboxes = car.boxes.cpu().numpy()
        cx1, cy1, cx2, cy2, cscore, cclass_id = cboxes.data[0]
        if int(cclass_id) in vehicles:
            cars_.append([cx1, cy1, cx2, cy2, cscore])

    for car_ in cars_:
        car_crop = image[int(car_[1]):int(car_[3]), int(car_[0]): int(car_[2]), :]

        detections = license_plate_detector(car_crop, device=0, stream = True)

        for idx, detection in enumerate(detections):
            boxes = detection.boxes.cpu().numpy()
            x1, y1, x2, y2, score, class_id = boxes.data[0]

            license_plate_crop = car_crop[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

    rect_start = (x1, y1)
    rect_end = (x2, y2)

    #image = cv2.rectangle(image, rect_start, rect_end, (0, 0, 255), 2)
    #image = cv2.putText(image, license_plate_text, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2) 
    print(license_plate_text)
    cv2.imshow('Result Image', license_plate_crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()