import cv2
#import matplotlib as plt
from matplotlib import pyplot as plt
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
#cv2.imshow("Originale",Image)

#crop_img = Image[a-1:a+2, b-1:b+2]
a = Image[0:300, 0:300, 0]
b = Image[0:300, 0:300, 1]
c = Image[0:300, 0:300, 2]

plt.hist(a.ravel(),300,[0,256]);
plt.xlabel('Valeur couleur rouge')
plt.ylabel(u'pixels')
plt.show()

plt.hist(b.ravel(),300,[0,256]);
plt.xlabel('Valeur couleur verte')
plt.ylabel(u'pixels')
plt.show()

plt.hist(c.ravel(),300,[0,256]);
plt.xlabel('Valeur couleur bleue')
plt.ylabel(u'pixels')
plt.show()
plt.close()





