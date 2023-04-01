#Importing modules
import cv2
import numpy as np
import face_recognition
import os
import time


#Constants
filepath = "Test4.mp4"


#input name:
detect = input("enter the suspect name: ")
detect = detect.upper()

#Taking the images from the given path and passing in myList
path='Images'
images=[]
ClassNames=[]
myList=os.listdir(path)
print(myList)





# passing images in image list and spliting names and appending them in ClassName
for cl in myList:
    img=cv2.imread(f'{path}/{cl}')
    images.append(img)
    ClassNames.append(os.path.splitext(cl)[0])
print(ClassNames)




#Function to append encodings of the images
def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown=findEncodings(images)

#Web-Cam
cap=cv2.VideoCapture(filepath)
cap.set(cv2.CAP_PROP_FPS, int(90))

#Main loop
while True:
    ret, img=cap.read()
    #imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#Finding Locations of frame and encoding each frame
    faceCurFrame= face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)

#Matching the encodings and getting min distance to get best match
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)


        #Matching with DataBase
        if matches[matchIndex]:
            name=ClassNames[matchIndex].upper()
            #color= 'green'


            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(imgS, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imgS, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgS, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
            print(name)
            if name == detect:
                print("Suspect Found")
                time.sleep(3)
                cap.release()
                cv2.destroyAllWindows()
                exit()


        else:
            #color= 'red'
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(imgS, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(imgS, (x1, y2-35), (x2, y2), (255, 0, 0), cv2.FILLED)
            print("Unrecognized")


    #Showing The original Image again.
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)
    imgS = cv2.resize(imgS, (0, 0), fx= 0.75, fy= 0.75)
    cv2.imshow('Attendance',imgS)
    if cv2.waitKey(1) == ord('q'):
        break


#releasing all cameras
print("Suspect Not Found")
cap.release()
cv2.destroyAllWindows()
