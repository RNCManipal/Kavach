import easyocr
import cv2
import imutils


img = cv2.imread('test.png',cv2.IMREAD_COLOR)


reader = easyocr.Reader(['en','en'])
result = reader.readtext(img)
print(result)

cv2.waitKey(0)
cv2.destroyAllWindows()