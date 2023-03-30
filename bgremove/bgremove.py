import cv2
import numpy as np
import time

def car_detect():

    cap=cv2.VideoCapture(r'test.mp4')
    car_locations = []

    min_width_react=90
    min_height_react=90

    count_line_popsition=600
    algo=cv2.createBackgroundSubtractorMOG2(100,200)

    def centre_handle(x,y,w,h):
        x1=int(w/2)
        y1=int(h/2)
        cx=x+x1
        cy=y+y1
        return cx,cy

    detect=[]
    offset=6
    counter=0
    
    try:
        while True:
            ret,frame1=cap.read()
            grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            blur=cv2.GaussianBlur(grey,(3,3),5)
            img_sub=algo.apply(blur)
            dilat =cv2.dilate(img_sub,np.ones((5,5)))
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            dilatada=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
            dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
            countershape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame1,(25,count_line_popsition),(1200,count_line_popsition),(255,127,0),3)

            for i,c in enumerate(countershape):
                (x,y,w,h)=cv2.boundingRect(c)
                validate_counter=(w>= min_width_react) and (h>=min_height_react)
                if not validate_counter:
                    continue
                
                cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)

                center=centre_handle(x,y,w,h)
                detect.append(center)
                cv2.circle(frame1,center,4,(0,0,255))
                for (x,y) in detect:
                    if y<(count_line_popsition+offset) and y>(count_line_popsition-offset):
                        counter+=1
                        car_locations.append([x, y, w, h])
                    cv2.line(frame1,(25,count_line_popsition),(1200,count_line_popsition),(0,127,255),3)
                    detect.remove((x,y))
                    print("Vehicle counter:"+str(counter))
                    
            cv2.putText(frame1,"VEHICLE COUNTER:"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)    

            #cv2.imshow('detecter',dilatada)
            cv2.imshow('frame',frame1)
            if cv2.waitKey(1)==13:
                break
        cv2.destroyAllWindows()
        cap.release()
        
    except:
        print(car_locations)
        return car_locations
    
if __name__ == '__main__':
    car_detect()