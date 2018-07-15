import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

OUTPUT_PATH = '../image/'
OUTPUT_NUM = 5000
OUTPUT_PERIOD = 1000

count = 0
time1 = 0
time2 = 0

while count < 5000:
  count += 1

count = 0
while count < OUTPUT_NUM:
    time1 = cv2.getTickCount()
    
    timeP = math.floor(((time1 - time2) * 1000) / cv2.getTickFrequency())
    if timeP < OUTPUT_PERIOD:
        continue
    
    (grabbed, img) = cap.read()
    if not grabbed:
        print("no signal!\n")
        continue
    
    cv2.imshow("capture", img)
    fileName = OUTPUT_PATH + "yayoi_" + str(count) + '.png'
    cv2.imwrite(fileName, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    print("file " + fileName + " saved!\n")
    
    count += 1
    if count >= OUTPUT_NUM:
        print("capture complete!\n")
        break
        
    getKey = cv2.waitKey(10) & 0xFF
    if getKey is ord('q') or getKey is ord('Q'):
      break
    
    time2 = cv2.getTickCount()
    
cap.release()
cv2.destroyAllWindows()