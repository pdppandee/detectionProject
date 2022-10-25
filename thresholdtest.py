
from copy import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('imageraw/fulltest.jpg',3)
# img = img[500:3400,500:2500]
img = cv2.resize(img, (400, 580))
copyimg = img.copy()
img = cv2.medianBlur(img,3)

cv2.imshow("original", img)

rows, cols,_ = img.shape
print("Rows: ", rows)
print("Cols: ", cols)

# contour, _= cv2.findContours(img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# print("Contour:" , contour, len(contour))

# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# x = contour[0][:,0,0]
# y = contour[0][:,0,1]
# plt.scatter(x,y,s=100,c='m')
# plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# set to binary -> threshold(src, threshold, maxval(255), type)
ret,thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV) 
# cv2.imshow("bw", thresh)

'''
----#AdaptiveThreshold
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,11,2)
'''
'''
----#Contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Contour:", len(contours))
'''

# Otsu's thresholding after Gaussian filtering
imgblur = cv2.GaussianBlur(img,(5,5),0)
ret3,threshotsu = cv2.threshold(imgblur,120,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow("threshotsu", threshotsu)


#SELECT CONTOUR IMAGE -------------------------------------------------------------
cont,_ = cv2.findContours(threshotsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contimg = cv2.drawContours(img,cont,-1,255,3)
plt.imshow(contimg, cmap='gray')
plt.show()
c = max(cont,key = cv2.contourArea)

x,y,w,h = cv2.boundingRect(c)
cv2.rectangle(copyimg,(x,y),(x+w,y+h), (0,0,255), 2)
cv2.imshow("rectangle", copyimg)
cropimg = copyimg[y:y+h,x:x+w]

cropimg = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)
cv2.imshow("Crop", cropimg)

#DETECT IMAGE -------------------------------------------------------------

cropcropimg = cropimg[90:218, 40:152]
# rows, cols = cropcropimg.shape
# print("Rows cropcrop: ", rows)
# print("Cols cropcrop: ", cols)

cropcropimg = cv2.resize(cropcropimg,(336, 384))

cv2.imshow("CropCrop", cropcropimg)
ret, cropbwimg = cv2.threshold(cropcropimg, 90, 255, cv2.THRESH_BINARY)
# ret, cropbwimg = cv2.threshold(cropcropimg, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("ROI", cropbwimg)

# counting the number of pixels ----------------------------------
cntwhitepix= np.sum(cropbwimg == 255)
cntblackpix = np.sum(cropbwimg == 0)
  
print('Number of white pixels:', cntwhitepix)
print('Number of black pixels:', cntblackpix)

percentage = (cntwhitepix * 100) / (cntblackpix + cntwhitepix)

print('white percent:', percentage , "%")

if percentage >= 3.0:
    print(" -> ng")
else:
    print(" -> ok")


cv2.waitKey(0)
cv2.destroyAllWindows()
