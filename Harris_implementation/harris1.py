#BAKOUR IMENE
#BOUCHAI NESMA HADIA

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import corner_peaks, corner_harris
from scipy import signal as sig
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
import cv2
import time

                   
def gradient_x(imggray):
    ##Sobel operator x kernels.
    kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    return sig.convolve2d(imggray, kernel_x, mode='same')
def gradient_y(imggray):
    ##Sobel operator y kernels
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return sig.convolve2d(imggray, kernel_y, mode='same')


#Chargement de l'image
img = imread('4.jpg')
#Conversion d'image couleur en niveaux de gris
imggray = rgb2gray(img)
img2 = np.copy(imggray)
#Calcul de dérivée spatiale
I_x = gradient_x(imggray)
I_y = gradient_y(imggray)

#cv2.imshow("test1", imggray)
#cv2.imshow("X", I_x)
#cv2.imshow("Y", I_y)

#Application des filtres gaussien Multidimensiona
Ixx = sc.ndimage.gaussian_filter(I_x**2, sigma=1)
Ixy = sc.ndimage.gaussian_filter(I_y*I_x, sigma=1)
Iyy = sc.ndimage.gaussian_filter(I_y**2, sigma=1)

#Configuration du tenseur de la structure
k = 0.05
# determinant
detA = Ixx * Iyy - Ixy ** 2
# trace
traceA = Ixx + Iyy
harris_response = detA - k * traceA ** 2

#Trouver les coins
img_copy_for_corners = np.copy(img)
temp1 = time.time()
for rowindex, response in enumerate(harris_response):
    for colindex, r in enumerate(response):
        #print(r)
        if r > 0:
            # this is a corner
            img_copy_for_corners[rowindex, colindex] = [255,0,0]

temp2 = time.time()
print('temps d\'execution harris : ', temp2-temp1)
operatedImage = np.float32(imggray) 
# apply the cv2.cornerHarris method 
# to detect the corners with appropriate 
# values as input parameters
temp1 = time.time()
dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07) 
  
# Results are marked through the dilated corners 
dest = cv2.dilate(dest, None) 
#cv2.imshow("dilat",dest)
# Reverting back to the original image, 
# with optimal threshold value 
img[dest > 0.01 * dest.max()]=[255,0,0]
temp2 = time.time()
print('temps d\'execution harris opencv: ', temp2-temp1)
# the window showing output image with corners 
#cv2.imshow('Image with Borders', img) 
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
ax[0].set_title("corners found ")
ax[0].imshow(img_copy_for_corners)
ax[1].set_title("opencv harris corners")
ax[1].imshow(img)
plt.show()
# De-allocate any associated memory usage  
if cv2.waitKey(0) & 0xff == 27: 
    cv2.destroyAllWindows() 



