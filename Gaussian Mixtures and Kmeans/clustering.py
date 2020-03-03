#!/usr/bin/python

#!/usr/bin/python

#!/usr/bin/python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal

#Your code here


def loadData(fileDj):
    X = np.loadtxt(fileDj)[:, :-1]
    y = np.loadtxt(fileDj)[:, -1]
    #Your code here
    return X, y

## K-means functions


def getInitialCentroids(X, k):
    initialCentroids = {}
    for i in range(k):
        initialCentroids[i] = X[np.random.choice(len(X), replace=False)]
    return initialCentroids


def getDistance(pt1, pt2):
    dist = np.linalg.norm(pt1 - pt2)
    return dist


def allocatePoints(X, clusters):
    distances = {}
    clust_range = range(len(clusters))
    #Get distances from cluster centroids
    for i in range(0, len(X)):
        distances[i] = [
            getDistance(X[i], clusters[k]) for k in clust_range
        ]

    #assign each datapoint to the closest centroid
    clusters_assignments = [np.argmin(distances[i]) for i in distances.keys()]

    return clusters_assignments, clusters


def updateCentroids(X, clusters, clusters_assignments, maxIter):
    diff = []
    df = pd.DataFrame(X, columns=['feat1', 'feat2'])
    df['cluster'] = clusters_assignments
    old = df.groupby(['cluster']).mean().reset_index()
    for i in range(maxIter):

        new_clusters = {}
        for i in range(len(clusters)):
            new_clusters[i] = df.groupby(['cluster']).mean().loc[i].values
        clusters_assignments, clusters = allocatePoints(X, new_clusters)
        df = pd.DataFrame(X, columns=['feat1', 'feat2'])
        df['cluster'] = clusters_assignments

    return new_clusters, clusters_assignments, diff


def visualizeClusters(X, clusters_assignments, new_clusters):
    values = np.array(list(new_clusters.values()))
    plt.scatter(X[:, 0], X[:, 1], c=clusters_assignments, s=50, cmap='viridis')
    plt.scatter(values[:, 0], values[:, 1], c='black', s=200, alpha=0.75)
    plt.title('Clusters: {}'.format(len(new_clusters)))
    plt.show()


def kmeans(X, k, maxIter=300):
    centroids_old = getInitialCentroids(X, k)
    clusters_assignments, _ = allocatePoints(X, centroids_old)
    new_centroids, clusters_assignments, diff = updateCentroids(
        X, centroids_old, clusters_assignments, maxIter=maxIter)
    return new_centroids, clusters_assignments, diff


def kneeFinding(X, kList, maxIter=300):
    sse = {}

    for k in kList:
        new_clusters, clusters_assignments, diff = kmeans(X,
                                                          k=k,
                                                          maxIter=10)
        data = np.insert(X, 2, np.array(clusters_assignments), axis=1)
        for i in new_clusters:
            temp_sse = 0
            for j in data:
                classif = j[-1]
                temp_sse += ((j[0] - new_clusters[classif][0]) +
                             (j[1] - new_clusters[classif][1]))**2
            sse[k] = temp_sse
        print('Clusters: {} ---- SSE: {}'.format(k, sse[k]))

    plt.plot([i for i in sse.keys()], list(sse.values()))
    plt.xlabel('Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Plot')
    plt.show()

    return sse


def purity(y, clusters):
    y_true = np.array(y - 1)
    y_pred = np.logical_not(clusters).astype(int)

    purity = np.zeros((2, 2))
    for i, j in zip(y_pred, y_true):
        if i == 0 and j == 0:
            purity[0][0] += 1
        elif i == 0 and j == 1:
            purity[1][0] += 1
        elif i == 1 and j == 0:
            purity[0][1] += 1
        else:
            purity[1][1] += 1
    purity = np.sum(np.amax(purity, axis=0)) / np.sum(purity)
    return purity



## GMM functions

def GMM(X,k,covType, num_iters = 100):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[:] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)

    #shape of cov matrix should be p x p x k
        # 2 x 2 x 2 for dataset1
        # 13 x 13 x 2 for dataset2
    cov = []
    for i in np.arange(k):
        cov.append(covMat)

    #proportions updated later. Randomly assume uniform dist. MSTEP
    mixing_proportions = np.repeat(1/k, k )

    #need initial means. Select Randomly
    n,d = X.shape
    means = []
    for i in np.arange(k):
        means.append(X[np.random.choice(n, size = 1)].flatten())
    means = np.array(means)

    initialClusters = {}
    densities = np.empty((n, k), np.float)

    #likelihood
    counter = 0
    while counter < num_iters:
        for i in np.arange(n):
            x = X[i] #extract single data point
            for j in np.arange(k):
                densities[i][j] = multivariate_normal.pdf(x, means[j], cov[j])

        #ESTEP for initial assignment
        z = np.empty((n, k), np.float)
        for i in np.arange(n):
            x = X[i]
            denominator = np.dot(mixing_proportions.T, densities[i])
            for j in np.arange(k):
                z[i][j] = mixing_proportions[j] * densities[i][j] / denominator
        #MSTEP
        for i in np.arange(k):
            z_t = (z.T)[i]
            denominator = np.dot(z_t.T, np.ones(n))
            means[i] = np.dot(z_t.T, X) / denominator
            difference = X - np.tile(means[i], (n, 1))
            cov[i] = np.dot(np.multiply(z_t.reshape(n,1), difference).T, difference) / denominator
            mixing_proportions[i] = denominator / n
        counter += 1
    for i in np.arange(n):
        initialClusters[i] = np.argmax(z[i])

    #Your code here
    return cov, means, mixing_proportions, densities, z, initialClusters




def main():
    #######dataset path
    try:
        datadir = sys.argv[1]
    except:
        datadir = os.getcwd()
    pathDataset1 = datadir+'\\data_sets_clustering\\humanData.txt'
    pathDataset2 = datadir+'\\data_sets_clustering\\audioData.txt'
    dataset1_X, dataset1_y = loadData(pathDataset1)
    dataset2_X, dataset2_y = loadData(pathDataset2)

    #Q5
    new_centroids, clusters_assignments, diff = kmeans(X = dataset1_X, k = 2, maxIter=300)
    visualizeClusters(dataset1_X, clusters_assignments, new_centroids)
    purity(y = dataset1_y ,clusters= clusters_assignments )
    print('Purity KMEANS Dataset1:', purity(y = dataset1_y, clusters= clusters_assignments))
    kneeFinding(X = dataset1_X,kList=range(1,7), maxIter=10)

    np.random.seed(1)
    cov, means, mixing_proportions, densities, z, initialClusters = GMM(X = dataset1_X,  k =2, covType= 'diag', num_iters = 10)
    print('Purity GMM Dataset1 DIAG:', purity(y = dataset1_y-1, clusters= list(initialClusters.values())))
    values = np.array(list(initialClusters.values()))
    plt.scatter(dataset1_X[:, 0], dataset1_X[:, 1], c=values, s=50, cmap='viridis')
    plt.show()

    cov, means, mixing_proportions, densities, z, initialClusters = GMM(X = dataset1_X,  k =2, covType= 'full', num_iters = 20)
    print('Purity GMM Dataset1 FULL:', purity(y = dataset1_y-1, clusters= list(initialClusters.values())))
    values = np.array(list(initialClusters.values()))
    plt.scatter(dataset1_X[:, 0], dataset1_X[:, 1], c=values, s=50, cmap='viridis')
    plt.show()


    cov, means, mixing_proportions, densities, z, initialClusters = GMM(X = dataset2_X,  k =2, covType= 'diag', num_iters = 1000)
    print('Purity GMM Dataset1 DIAG:', purity(y = dataset2_y-1, clusters= list(initialClusters.values())))
    values = np.array(list(initialClusters.values()))
    plt.scatter(dataset2_X[:, 0], dataset2_X[:, 1], c=values, s=50, cmap='viridis')
    plt.show()

    cov, means, mixing_proportions, densities, z, initialClusters = GMM(X = dataset2_X,  k =2, covType= 'full', num_iters = 1000)
    print('Purity GMM Dataset1 FULL:', purity(y = dataset2_y-1, clusters= list(initialClusters.values())))
    values = np.array(list(initialClusters.values()))
    plt.scatter(dataset2_X[:, 0], dataset2_X[:, 1], c=values, s=50, cmap='viridis')
    plt.show()

if __name__ == "__main__":
    main()
