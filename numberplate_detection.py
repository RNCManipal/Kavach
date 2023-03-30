import cv2
import imutils
import numpy as np
import pytesseract
import easyocr

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

img = cv2.imread('test.png',cv2.IMREAD_COLOR)
img = imutils.resize(img, width=500 )
h, w, c = img.shape
img = img[int(h/2):int(h), 0:500]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 100) #Perform Edge detection

cnts,new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1=img.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)

cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:30]
screenCnt = None #will store the number plate contour
img2 = img.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3) 
result =[]

idx=7
# loop over contours
for c in cnts:
  # approximate the contours
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4: #chooses contours with 4 corners
                screenCnt = approx
                x,y,w,h = cv2.boundingRect(c) #finds co-ordinates of the plate
                new_img=img[y:y+h,x:x+w]
                cv2.imwrite('./'+str(idx)+'.png',new_img) #stores the new image
                text=pytesseract.image_to_string(new_img,lang='eng') #converts image characters to string

                reader = easyocr.Reader(['en','en'])
                result = reader.readtext(img)
                              
                idx+=1


reader = easyocr.Reader(['en','en'])
result = reader.readtext(img)

print(result)
cv2.imshow("img1",img1)
cv2.imshow('graycsale image',gray)
cv2.imshow('edged image',edged)

cv2.waitKey(0)
cv2.destroyAllWindows()


