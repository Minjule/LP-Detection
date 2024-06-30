from ultralytics import YOLO
import cv2
from util import read_license_plate
import numpy as np
#model.predict('testimages\\14cropped.jpg', save=True, imgsz=320, conf=0.5)

license_plate_detector = YOLO('runs\\detect\\train12\\weights\\best.pt')

image = cv2.imread("testimages\\11.jpg")
cap = cv2.VideoCapture("C:\\Users\\Acer\\Desktop\\New folder\\data\\license_plate.mp4")
vehicles = [2, 3, 5, 7]

cv2.namedWindow('License plate', cv2.WINDOW_NORMAL)
cv2.moveWindow('License plate', 50, 50)
cv2.resizeWindow('License plate', 1080, 620)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    turshilt = license_plate_detector(frame, stream=True)
    for idx, detection in enumerate(turshilt):
        boxes = detection.boxes.cpu().numpy()
        x1, y1, x2, y2, score, class_id = boxes.data[0]
        turshilt_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
        turshilt_text, turshilt_score = read_license_plate(turshilt_crop)

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    frame = np.array(frame) 
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(frame, turshilt_text, (x1, y2+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

    rect_start = (x1, y1)
    rect_end = (x2, y2)

    #image = cv2.rectangle(image, rect_start, rect_end, (0, 0, 255), 2)
    #image = cv2.putText(image, license_plate_text, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
    print(turshilt_text)
    cv2.imshow('License plate', frame)
    #cv2.imshow('Vehicle crop', car_crop)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# model.predict('testimages\\cropped7.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\cropped5.jpg', save=True, imgsz=320, conf=0.5)

# """model.predict('testimages\\11cropped.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\cropped3.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\2.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\3.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\4.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\5.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\6.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\7.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\8.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\9.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\10.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\1.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\1cropped.jpg', save=True, imgsz=320, conf=0.5)

# model.predict('testimages\\11.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\12.jpg', save=True, imgsz=320, conf=0.5)
# model.predict('testimages\\13.jpg', save=True, imgsz=320, conf=0.5)"""

