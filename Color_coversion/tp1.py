import cv2
import matplotlib as plt
import numpy as np

'''R = [[228,14,204], [204,19,241],[188,23,175]]
G = [[34,30,167],[185,164,192],[29,84,149]]
B = [[189,248,94], [60,222,95], [188,22,175]]'''

Image = np.zeros((300, 300, 3), dtype= np.uint8)

Image[0:100, 0:100, 0:3] = list(np.random.choice(range(256), size = 3))
Image[0:100, 101:200, 0:3] = list(np.random.choice(range(256), size = 3))
Image[0:100, 201:300, 0:3] = list(np.random.choice(range(256), size = 3))

Image[101:200, 0:100, 0:3] = list(np.random.choice(range(256), size = 3))
Image[101:200, 101:200, 0:3] = list(np.random.choice(range(256), size = 3))
Image[101:200, 201:300, 0:3] = list(np.random.choice(range(256), size = 3))

Image[201:300, 0:100, 0:3] = list(np.random.choice(range(256), size = 3))
Image[201:300, 101:200, 0:3] = list(np.random.choice(range(256), size = 3))
Image[201:300, 201:300, 0:3] = list(np.random.choice(range(256), size = 3))
cv2.imshow("nom",Image)

image_gris = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray",image_gris)

image_hsv = cv2.cvtColor(Image, cv2.COLOR_BGR2HSV)

cv2.imshow("hsv",image_hsv)

image_xyz = cv2.cvtColor(Image, cv2.COLOR_BGR2XYZ)

cv2.imshow("xyz",image_xyz)
