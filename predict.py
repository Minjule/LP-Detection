from ultralytics import YOLO
import easyocr
import cv2

model = YOLO('runs\\detect\\train8\\weights\\best.pt')
img = cv2.imread("testimages\\14cropped.jpg")
reader = easyocr.Reader(['ru'], gpu=True)

detections = reader.readtext(img)

model.predict('testimages\\14cropped.jpg', save=True, imgsz=320, conf=0.5)

for detection in detections:
    bbox, text, score = detection
    text = text.upper().replace(' ', '')

    print(text)

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

