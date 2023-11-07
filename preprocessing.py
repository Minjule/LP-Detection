import cv2 

image = cv2.imread("testimages\\7.jpg")
src = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equ = cv2.equalizeHist(src)
cv2.imshow("djf", equ)
cv2.waitKey(0)
cv2.destroyAllWwindows()