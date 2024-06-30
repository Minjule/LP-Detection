import cv2

cap = cv2.VideoCapture("C:\\Users\\Acer\\Desktop\\New folder\\data\\license_plate.mp4")
cap.set(3, 640)
cap.set(4, 480)

cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.moveWindow('Webcam', 100, 100)
cv2.resizeWindow('Webcam', 1080, 620)

while True:
    ret, img= cap.read()
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()