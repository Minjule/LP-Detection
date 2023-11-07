from ultralytics import YOLO

model = YOLO('runs\\detect\\train7\\weights\\best.pt')

model.predict('testimages\\11cropped.jpg', save=True, imgsz=320, conf=0.5)

model.predict('testimages\\2.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\3.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\4.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\5.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\6.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\7.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\8.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\9.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\10.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\1.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\1cropped.jpg', save=True, imgsz=320, conf=0.5)

model.predict('testimages\\11.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\12.jpg', save=True, imgsz=320, conf=0.5)
model.predict('testimages\\13.jpg', save=True, imgsz=320, conf=0.5)

