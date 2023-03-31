import cv2
import imutils
import numpy as np
import pytesseract
import easyocr
import keras_ocr as kocr
import os

file = open("Numberplates_with_Index", "w")
file_paths = []
tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path
counter = 0
text = ''
reader = easyocr.Reader(['en','en'])
easy = []


print("Pytesseract")
file.write("Pytesseract")
for filename in os.listdir('./Output'):
  f = os.path.join('./Output', filename)
  file_paths.append(f)
  # checking if it is a file
  if os.path.isfile(f):
    img = cv2.imread(f,cv2.IMREAD_COLOR)
    img = imutils.resize(img, width=500 )
    h, w, c = img.shape
    # img = img[int(h/2):int(h), 0:500]

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
                    # cv2.imwrite('./'+str(idx)+'.png',new_img) #stores the new image
                    text=pytesseract.image_to_string(new_img,lang='eng')
                    if text != '':
                      file.writelines([str(counter), " : ", text])
                      print(counter, ' : ', text)
                      break
                    # cv2.imshow("img1",new_img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    idx+=1

    # if text == '':
    #   text=pytesseract.image_to_string(img,lang='eng')
    #   ("Pytersseract: ", text)
    counter = counter + 1
    
    result = reader.readtext(img, allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    easy.append(result)
    
print("EasyOcr: ")
file.write("EasyOcr: ")
for i in range(len(easy)):
  file.writelines([" ", str(i), " : "])
  print(i , ' : ')
  for j in range (len(easy[i])):
    file.write(easy[i][j][1])
    print(easy[i][j][1])

print("Keras OCR: ")
file.write("Keras OCR: ")
pipeline = kocr.pipeline.Pipeline()
images = [
         kocr.tools.read(plate) for plate in file_paths
]
prediction_groups = pipeline.recognize(images)

for i in range(len(prediction_groups)):
  print(i , " : " )
  file.writelines([" ", str(i) , " : " ])
  for j in range(len(prediction_groups[i])):
    print (prediction_groups[i][j][0])
    file.writelines(prediction_groups[i][j][0])
  print()

file.close()