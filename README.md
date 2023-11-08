# LP-Detection
License plate detection using YOLOv8 and OCR.

Used DVC to track all the changes of dataset and training experiments.

Currently AdamW is way better than SGD. (Used second dataset, maybe gotta find more suitable dataset cuz model is also learning pattern of the plates such us general drawings and colors)

Sooooo, train8\weights\best.pt is the result of training third dataset. Recently, i'm REALIZING that it'd be the best if YOLOv8s own car detection is used as a base detection and then as its result we could go on.

Dataset:
1. https://universe.roboflow.com/peter-vala-eodzc/plates-7wtgj

2. https://universe.roboflow.com/maddi-w3masterio-tech-gmail-com/alpr-2-vlr0h/dataset/1

3. https://universe.roboflow.com/ru-anrp/russian-license-plates-detector
