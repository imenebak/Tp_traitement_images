import numpy as np
import cv2
import time

class BackGroundSubtractor:
	def __init__(self, alpha):
		self.alpha  = alpha
		self.backGroundModel = firstFrame

	def getForeground(self, firstFrame, secondFrame, thirdFrame, frame):
                # aplly the background median formula:
		# NEW_BACKGROUND = medianFrame * ALPHA + PREVIOUS_BACKGROUND * (1 - APLHA)
		finalFrame =  secondFrame * self.alpha + firstFrame * (1 - self.alpha)
		return cv2.absdiff(finalFrame.astype(np.uint8), secondFrame)
# Denoising filter we need to apply to all frames 
def denoise(frame):
    frame = cv2.medianBlur(frame,5)
    frame = cv2.GaussianBlur(frame,(5,5),0)
    
    return frame
def am(foreGround):
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel =cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    #foreGround = cv2.medianBlur(foreGround,5)
    foreGround = cv2.morphologyEx(foreGround, cv2.MORPH_CLOSE, kernel)
    return foreGround

def initSubtraction(firstFrame, secondFrame, thirdFrame, f):
    global resultat, resultat1
    # init our class instance
    backSubtractor = BackGroundSubtractor(0.01)
    
    frame = cv2.imread(f, 0)

    
    while True:
        # Show the filtered image
        #cv2.imshow('input',denoise(frame))
        """key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break"""
        
        resultat = np.append(resultat, denoise(frame))
        # get the foreground
        foreGround = backSubtractor.getForeground(denoise(firstFrame), denoise(secondFrame), denoise(thirdFrame),(frame) )
        #cv2.imshow("foreGround", foreGround)
        #foreGround = cv2.medianBlur(foreGround,5)
        #foreGround = cv2.cvtColor(foreGround, cv2.COLOR_BGR2GRAY)
        # Apply thresholding on the background and display the resulting mask
        ret, mask = cv2.threshold(am(foreGround), 15, 255, cv2.THRESH_BINARY)
    
        mask = cv2.medianBlur(mask,5)
        #cv2.imshow('mask',mask)
        resultat1.append(mask)
        return 0
        #cv2.destroyAllWindows()
        #cv2.imshow('gray mask', gray)
        """key = cv2.waitKey(10) & 0xFF
        if key == 27:
                break"""


import os
resultat = []
resultat1 = []
data = np.array([])
#Loading Data
for path, dirs, files in os.walk("input"):
    for filename in files:
        data = np.append(data, "input/"+filename)
        
#Buffer Managment
for gg in range(0, len(data)-3,3):
    for i in range (3):
            # Takes 3 arguments (frames) in a buffer
            print(i)
            m = int(gg)+int(i)
            if(i == 0):
                #print(data[m])
                firstFrame = cv2.imread(data[m], 0)
                secondFrame = firstFrame
                thirdFrame = secondFrame
            if(i == 1):
                #print(data[m])
                secondFrame = cv2.imread(data[m+1],0)
            if(i == 2):
                #print(data[m])
                thirdFrame = cv2.imread(data[m+2],0)
            else:
                #print(data[m+1])
                initSubtraction(firstFrame, secondFrame, thirdFrame, data[m+3])




#Display results
print("DEBUT DE LA VIDEO")
# Read until video is completed
for a in resultat1:
    cv2.imshow('Out',a)
    key = cv2.waitKey(10) & 0xFF
    if key == 27:
        break
    time.sleep(0.1)
