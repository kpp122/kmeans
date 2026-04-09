import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt


def readDist():
    arr = []

    with open("dist.txt", "r") as file:
        for line in file:
            dat = line.split()
            arr.append(dat)
    arr = np.array(arr)
    file.close()

    return arr


def task1():
    data = readDist()
    print(data.shape)
    d = linkage(data, "complete")
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(d)
    plt.show()


def task2():
    data = []

    with open("data.txt", "r") as file:
        for line in file:
            data.append([float(f) for f in line.split()])
    file.close()

    centroids = [[2, 6], [4, 2], [6, 2]]
    partitions = optimalPartition(data, centroids)
    print(partitions)
    kmeans(data, centroids, partitions)


def euclideanDistance(obj1, obj2):
    return np.sqrt((obj1[0] - obj2[0]) ** 2 + (obj1[1] - obj2[1]) ** 2)


def nearestNeighbor(dataPoint, dataSpace):
    index = 0
    dist = float('inf')

    for i in range(len(dataSpace)):
        tmp = euclideanDistance(dataPoint, dataSpace[i])
        if tmp < dist:
            dist = tmp
            index = i
    return index


def optimalPartition(data, centroids):
    partitions = []

    for dataPoint in data:
        partition = nearestNeighbor(dataPoint, centroids)
        partitions.append(partition + 1)

    return partitions


def buildCentroid(dataSpace):
    centroid = [0] * len(dataSpace[0])

    for i in range(len(dataSpace)):
        for j in range(len(dataSpace[i])):
            centroid[j] = centroid[j] + dataSpace[i][j] / len(dataSpace)

    return centroid


def SSE(data, centroids, partitions):
    tmp_data = [[] for i in range(len(centroids))]
    partition_sse = [0] * len(centroids)

    for i in range(len(partitions)):  # Gather data from each partition
        tmp_data[partitions[i] - 1].append(data[i])

    for i in range(len(centroids)):
        for j in range(len(tmp_data[i])):
            partition_sse[i] = partition_sse[i] + euclideanDistance(tmp_data[i][j], centroids[i]) ** 2

    return partition_sse


def kmeans(data, centroids, partitions, iterations=5):
    k = len(centroids)
    centroids = centroids
    partitions = partitions
    sse = sum(SSE(data, centroids, partitions))

    for iteration in range(iterations):

        centroids_new = []
        tmp_data = [[] for i in range(k)]

        for i in range(len(partitions)):  # Gather data from each partition
            tmp_data[partitions[i] - 1].append(data[i])

        for i in range(k):  # Centroid step
            new_centroid = buildCentroid(tmp_data[i])
            centroids_new.append(new_centroid)

        centroids = centroids_new
        partitions = optimalPartition(data, centroids)

        sse_new = sum(SSE(data, centroids_new, partitions))

        _data = [[] for i in range(k)]
        for i in range(len(partitions)):
            _data[partitions[i] - 1].append(data[i])
        plot(_data, centroids, sse, iteration)

        if sse_new >= sse:
            print("converg")
            break
        else:
            sse = sse_new


def plot(data, centroids, sse, title):
    color = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    plt.title('SSE ' + str(sse))

    for i in range(len(data)):
        plt.scatter([j[0] for j in data[i]], [j[1] for j in data[i]], marker='o', c=color[i].reshape(1, -1), s=15)
    plt.scatter([i[0] for i in centroids], [i[1] for i in centroids], marker='o',  c='black', s=25)
    plt.savefig(str(title) + '.png')
    plt.clf()


if __name__ == '__main__':
    task2()
