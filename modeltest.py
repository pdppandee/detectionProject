import cv2
from cv2 import resize
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf



#load model

model = tf.keras.models.load_model('models/detectmodel1.h5')

# Check its architecture
print(model.summary())

classes = ['ok','ng']



cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    grayframe =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    #Object Detection
    imgblur = cv2.GaussianBlur(grayframe,(5,5),0)
    ret3,threshotsu = cv2.threshold(imgblur,160,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    cont,_ = cv2.findContours(threshotsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #loop detect object
    for cnt in cont:
        x,y,w,h = cv2.boundingRect(cnt)
       
        
        # #crop each object
        # for objpic in cnt:

        objframe = frame[y:y+h,x:x+w]  

        objpic = objframe.copy()
        objpic = cv2.resize(objpic, (256,256))
        objpic = objpic.astype("float") / 255.0
        objpic = np.array(objpic)
        objpic = np.expand_dims(objpic, axis=0)


        predict = model.predict(objpic)[0] 
        # print(predict) output -> array([[0.31972545]], dtype=float32)

        # idx = np.argmax(predict)
        
        if predict > 0.5: 
            label = classes[0]
            print(predict[0])
            print(f'Predicted class is 0 == ok')
        else:
            label = classes[1]
            print(predict[0])
            print(f'Predicted class is 1 == ng')

        # cv2.imshow("pic", objframe)
        

        label = "{}: {:.5f}%".format(label, predict[0]*100) 
        cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 2)    
        cv2.putText(frame, label, (x,y), 1, 1,(255,0,0))

    cv2.imshow("camera frame", frame)

    #press a q on a keybord to exit
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()