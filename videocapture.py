# from cmath import pi
import fractions
import cv2
from matplotlib import pyplot as plt
import numpy as np


def countingWhitePercentage(img):
    cntwhitepix= np.sum(img == 255)
    cntblackpix = np.sum(img == 0)
    
    # print('Number of white pixels:', cntwhitepix)
    # print('Number of black pixels:', cntblackpix)

    percentage = (cntwhitepix * 100) / (cntblackpix + cntwhitepix)

    # print('white percent:', percentage , "%")

    if percentage >= 0.3:
        cv2.putText(frame, str('%.7f'%(percentage)), (x,y), 1, 1,(0,0,255))
        # print(" -> ng")
    else:
        cv2.putText(frame, str(percentage), (x,y), 1, 1,(0,255,0))
        # print(" -> ok")

        
# videocapture from device

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    grayframe =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _,thresh = cv2.threshold(grayframe, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("threshold", thresh)


    #Object Detection
    imgblur = cv2.GaussianBlur(grayframe,(5,5),0)
    ret3,threshotsu = cv2.threshold(imgblur,160,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    cont,_ = cv2.findContours(threshotsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contimg = cv2.drawContours(frame,cont,-1,255,3)
    # c = max(cont,key = cv2.contourArea)

    for cnt in cont:
        x,y,w,h = cv2.boundingRect(cnt)
        for pic in cnt:
            pic = frame[y:y+h,x:x+w]
            cv2.imshow("pic", pic)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            r,c = pic.shape
            # piccrop = pic[25:100, 50:145]
            # ret, picbw = cv2.threshold(pic, 40, 255, cv2.THRESH_BINARY)
            # print(h,r,w,c)
            # print(type(y),type(h),type(r))
            if (r <= h and c <= w):
                # print("j")
                piccrop = pic[int(2*r/6):int(5*r/6), int(2*c/11):int(6*c/11)]
                
                piccrop2 = cv2.cvtColor(piccrop, cv2.COLOR_GRAY2BGR)
                
                ret, picbw = cv2.threshold(piccrop, 40, 255, cv2.THRESH_BINARY)

                cont2,_ = cv2.findContours(picbw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for pic2 in cont2:
                    x2,y2,w2,h2 = cv2.boundingRect(pic2)
                    cv2.rectangle( piccrop2,(x2,y2),(x2+w2,y2+h2), (0,255,0), 1)
                    # cv2.imshow("select", pic)
                    countingWhitePercentage(picbw)
                    
                cv2.imshow("thresh pic", picbw)
                cv2.imshow("pic crop",  piccrop2)    

        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,0,255), 2)
        

    cv2.imshow("camera frame", frame)
    cv2.imshow("otsu threshold", threshotsu)
    if cv2.waitKey(1) & 0xFF == ord('q'): #press a q on a keybord to exit
        break

cap.release()
cv2.destroyAllWindows()


