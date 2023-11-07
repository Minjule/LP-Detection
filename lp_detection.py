from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def main():
    results = model.train(data='data.yaml', epochs=3, imgsz=640, batch = 4)

if __name__ == '__main__':
    main()