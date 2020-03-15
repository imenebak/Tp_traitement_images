"""
BAKOUR IMENE
BOUCHALI NESMA HADIA
K MEANS SEGMENTATION

STEPS :
First, randomly init a vector of centroids, assign each data point to any one of the k clusters
Calculate the centers of these clusters
Calculate the distance of all the points from the center of each cluster
repeat steps 
"""

import numpy as np
import cv2
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

K = 3
D = 3
Im = "4.jpg"
THRESH = 1E-4
useType = np.float

def findCentroids(feats, centroids):
    N = feats.shape[0]
    idx = np.zeros((N), dtype=np.int)
    for i in range(N):
        bestIdx = -1
        bestDist = -1
        for j in range(K):
            u = feats[i] - centroids[j]
            dist = np.inner(u, u)
            if (bestIdx == -1 or dist < bestDist):
                bestIdx = j
                bestDist = dist
        idx[i] = bestIdx
    return idx

if __name__ == '__main__':
    
    img = cv2.imread(Im)

    size = (img.shape[0] * img.shape[1], img.shape[2])
    #print(img.shape[0] , img.shape[1], img.shape[2])
    feats = np.reshape(img, size).astype(useType) / 256.0

    centroids = np.random.rand(K, D).astype(useType)
    newCentroids = np.zeros((K, D), dtype=useType)
    ITER = 1
    N = size[0]
    print ('Nombre de Features:', N)
    print ('Nombre de Clusters:', K)
    temp1 = time.time()
    while (True):
        idx = findCentroids(feats, centroids)
        for i in range(K):
            tmp = feats[idx == i]
            if len(tmp) == 0 or (tmp is None):
                tmp = centroids[i].copy()
            newCentroids[i] = np.mean(tmp, axis=0)
        #ERROR CALCUL
        error = 0.0
        for i in range(K):
            u = newCentroids[i] - centroids[i]
            error = error + np.sqrt(np.inner(u, u))

        print ('ITERATION:', ITER, 'Error Rate:', error)
        ITER = ITER + 1
        if (error < THRESH):
            break
        
        centroids = newCentroids.copy()
        if (ITER > 20):
            break

    idx = findCentroids(feats, centroids)
    centroids = centroids * 256.0
    idx = np.reshape(idx, (img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pix = centroids[idx[i][j]]
            img[i][j] = pix
    temp2 = time.time()
    print('temps d\'execution du k-means segmentation implementation: ', temp2-temp1)
    
    ################################################################################################################################################
    ##################################################################### UTILISANT LA FONCTION PREDEFINIE #########################################

    imgg = cv2.imread(Im)
    #cv2.imshow("original image",img)

    imgg1 = np.reshape(imgg, (imgg.shape[0]*img.shape[1], 3))
    cl = cluster.KMeans(n_clusters =3, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
    temp1 = time.time()
    d = cl.fit(imgg1)
    y = d.predict(imgg1)
    #print(y)

    m = np.where(y==2)
    imgg1[m, 0] = 0
    #img1[m, 1] = 0
    imgg1[m, 2] = 0
    temp2 = time.time()
    print('temps d\'execution du k-means segmentation implémentée en sklearn: ', temp2-temp1)
    imgg2 = np.reshape(imgg1,(imgg.shape[0], imgg.shape[1], 3))
    ################################################################################################################################################
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,10))
    ax[0].set_title("k means ségmentation ")
    ax[0].imshow(img)
    ax[1].set_title("Resultat utilisant la fonction predefinie")
    ax[1].imshow(imgg2)
    plt.show()
    # De-allocate any associated memory usage  
    if cv2.waitKey(0) & 0xff == 27: 
        cv2.destroyAllWindows() 
