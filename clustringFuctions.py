import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as shc
from pandas import DataFrame
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import math
from scipy.stats import skew, kurtosis
from dtaidistance import dtw, clustering
from dtaidistance import dtw_visualisation as dtwvis
import pandas as pd
from sktime.distances.elastic_cython import dtw_distance


# the class has functions which return extracted feature seperately
class featureExtraction():
    def mean(data):
        return round(data.mean(), 2)

    def median(data):
        return round(np.median(data), 2)

    def std(data):
        return round(data.std(), 2)

    def variance(data):
        return round(data.var(), 2)

    def sum(data):
        return round(data.sum(), 2)

    def skewness(data):
        return round(float(skew(data)), 2)

    def kurtosis(data):
        return round(float(kurtosis(data)), 2)

    def energy(data):
        return round(float(sum(map(lambda x: x * x, data))), 2)

    def rms(data):
        return round(math.sqrt(sum(map(lambda x: x * x, data)) / len(data)), 2)

    def crestFactor(data):
        return float(max(data) / min(data))


# plot extracted feature
def plotFeature(data):
    print("###    Data Features   ###")
    print("mean : ", featureExtraction.mean(data))
    print("median : ", featureExtraction.median(data))
    print("std : ", featureExtraction.std(data))
    print("variance : ", featureExtraction.variance(data))
    print("sum : ", featureExtraction.sum(data))
    print("skewness : ", featureExtraction.skewness(data))
    print("kurtosis : ", featureExtraction.kurtosis(data))
    print("energy : ", featureExtraction.energy(data))
    print("rms : ", featureExtraction.rms(data))
    print("crestFactor : ", featureExtraction.crestFactor(data))
    print("")


# prepare features use featureExtraction class given above and return the all feature in list
def getFeature(data):
    temp = []
    temp.append(featureExtraction.mean(data))
    temp.append(featureExtraction.median(data))
    temp.append(featureExtraction.std(data))
    temp.append(featureExtraction.variance(data))
    temp.append(featureExtraction.sum(data))
    temp.append(featureExtraction.skewness(data))
    temp.append(featureExtraction.kurtosis(data))
    temp.append(featureExtraction.energy(data))
    temp.append(featureExtraction.rms(data))
    return np.array(temp).reshape(len(temp), -1)


# get feature for each data packet seperately
def getFeaturedData(data):
    merged = list(getFeature(i) for i in data)
    return np.array(merged)


# windowing process for vibration data with initial window_size 3
def parseVib2Window(data, window_size=3):
    result = []
    k = 0
    temp = []
    for raw in data:
        k += 1
        temp.extend(raw)
        if (k % window_size == 0):
            result.append(temp)
            temp = []
    return np.array(result)


# windowing process for current data with initial window_size 3
def parseData2Window(data, window_size=3):
    k = 0
    temp = []
    result = []
    while k < len(data):
        temp.append(float(data[k]))
        k += 1
        if (k % window_size == 0):
            result.append(temp)
            temp = []

    return np.array(result)


# normalize given data and return it
def getNomalizedData(data):
    temp = []
    scaler = MinMaxScaler()
    for i in range(len(data)):
        scaler.fit(data[i])
        temp.append(scaler.transform(data[i]))

    result = []
    for i in temp:
        temp2 = []
        for j in i:
            temp2.append(float(j))
        result.append(temp2)

    return np.array(result)


# read given csv file and return data
def getCsvData(filename) -> list:
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data


# eliminate header name from main data and return just values
def prepareMainData(mainData: list) -> list:
    mainData = np.array(mainData)
    mainData = mainData[1:, :]
    return mainData


# get both wheel's current data from whole main data and return them
def prepareWheelCurrentData(mainData: list) -> list:
    wheelCurrent1, wheelTemp2 = [], []
    for i in mainData[:, 9]:
        if (i == ""):
            wheelCurrent1.append(0)
        else:
            wheelCurrent1.append(float(i))
    for i in mainData[:, 15]:
        if (i == ""):
            wheelTemp2.append(0)
        else:
            wheelTemp2.append(float(i))
    return [wheelCurrent1, wheelTemp2]


# get specified vibration data from given vibData and return it back as seperated and flatten format
def prepareVibrationData(vibData: list, tag: str) -> list:
    # tag can be VibrationX,VibrationY,VibrationZ
    vibData = np.array(vibData)
    vibData = vibData[1:, 1:]
    vibData = list(vibData)

    titX, titY, titZ = [], [], []

    for row in vibData:
        try:
            if (row[0] == "x"):
                temp = []
                for item in row[1:]:
                    if (item == ""):
                        temp.append(0)
                    else:
                        temp.append(float(item))
                titX.append(temp)
            elif (row[0] == "y"):
                temp = []
                for item in row[1:]:
                    if (item == ""):
                        temp.append(0)
                    else:
                        temp.append(float(item))
                titY.append(temp)
            elif (row[0] == "z"):
                temp = []
                for item in row[1:]:
                    if (item == ""):
                        temp.append(0)
                    else:
                        temp.append(float(item))
                titZ.append(temp)
            else:
                print("prepareVibrationData:: else error !! ")
        except:
            print("prepareVibrationData:: except error")
    del temp

    if tag == "VibrationZ":
        vibFlatten = np.array(titZ)
        vibFlatten = vibFlatten.flatten()
        vibFlatten = list(vibFlatten)
        return [titZ, vibFlatten]
    elif tag == "VibrationX":
        vibFlatten = np.array(titX)
        vibFlatten = vibFlatten.flatten()
        vibFlatten = list(vibFlatten)
        return [titX, vibFlatten]
    elif tag == "VibrationY":
        vibFlatten = np.array(titY)
        vibFlatten = vibFlatten.flatten()
        vibFlatten = list(vibFlatten)
        return [titY, vibFlatten]


# plot Current Data
def plotCurrent(wheel, title):
    plt.figure(figsize=(16, 9))
    plt.plot(wheel)
    plt.title(title)
    plt.xlabel("sample number")
    plt.ylabel("sample value")
    plt.legend()
    plt.savefig(title)


# plot Vibration Data
def plotVibration(wheel, title):
    plt.figure(figsize=(16, 9))
    plt.plot(wheel)
    plt.title(title)
    plt.xlabel("sample number")
    plt.ylabel("sample value")
    plt.legend()
    plt.savefig(title)

# return color name with specified number
def getColor(number):
    return np.array(["turquoise", "tomato", "violet", "blue", "wheat", "pink", "red"])[:number]

# given feature extracted and normalized as wheel1Normalized is clustered
# if n_cluster is specified or not, n_cluster is give as a input into clustering function
# plot anomaly score of clusters and return classses
def plotAnomalyScoreofFeatured(wheel1Normalized, windowData, method, metric, title, n_cluster=None):
    plt.figure(figsize=(16, 9))
    if n_cluster == None:
        cluster = AgglomerativeClustering(affinity=metric, linkage=method)
        cluster.fit_predict(wheel1Normalized)
    else:
        cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity=metric, linkage=method)
        cluster.fit_predict(wheel1Normalized)

    classes = list(cluster.labels_)
    score = {}
    for i in set(cluster.labels_):
        score[i] = round((len(classes) - classes.count(i)) / len(classes), 3)

    rgb = getColor(len(set(cluster.labels_)))
    k = len(windowData[0])

    isLabel = set(cluster.labels_)

    for raw in range(len(windowData)):
        if cluster.labels_[raw] in isLabel:
            isLabel.remove(cluster.labels_[raw])
            plt.plot(range(raw * k, raw * k + k), windowData[raw], color=rgb[cluster.labels_[raw]],
                     label="count:" + str(classes.count(cluster.labels_[raw])) + " score:" + str(
                         score[cluster.labels_[raw]]))
            if raw != 0:
                plt.plot(range(raw * k - 1, raw * k + 1), [windowData[raw - 1][k - 1], windowData[raw][0]],
                         color=rgb[cluster.labels_[raw]])
        else:
            plt.plot(range(raw * k, raw * k + k), windowData[raw], color=rgb[cluster.labels_[raw]])
            if raw != 0:
                plt.plot(range(raw * k - 1, raw * k + 1), [windowData[raw - 1][k - 1], windowData[raw][0]],
                         color=rgb[cluster.labels_[raw]])

    plt.xlabel("sample number")
    plt.ylabel("sample value")
    plt.title(title)
    plt.legend()
    plt.savefig(title)

    return classes


# return the distance matrix of give data
# using this in calculate DTW distance
def getDTWDistMatrix(data):
    series_list = []

    distfast = dtw.distance_matrix_fast(data)
    distfast = lower(distfast)

    return distfast

# given feature extracted as wheel1Parsed is clustered
# if n_cluster is specified or not, n_cluster is give as a input into clustering function
# plot anomaly score of clusters and return classses
def plotAnomalyScoreofDTW(wheel1Parsed, windowData, method, title, n_clusters):
    plt.figure(figsize=(16, 9))
    if n_clusters == None:
        cluster = AgglomerativeClustering(affinity="precomputed", linkage=method)
    else:
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage=method)

    cluster.fit_predict(getDTWDistMatrix(wheel1Parsed))

    classes = list(cluster.labels_)
    score = {}
    for i in set(cluster.labels_):
        score[i] = round((len(classes) - classes.count(i)) / len(classes), 3)

    rgb = getColor(len(set(cluster.labels_)))
    k = len(windowData[0])

    isLabel = set(cluster.labels_)

    for raw in range(len(windowData)):
        if cluster.labels_[raw] in isLabel:
            isLabel.remove(cluster.labels_[raw])
            plt.plot(range(raw * k, raw * k + k), windowData[raw], color=rgb[cluster.labels_[raw]],
                     label="count:" + str(classes.count(cluster.labels_[raw])) + " score:" + str(
                         score[cluster.labels_[raw]]))
            if raw != 0:
                plt.plot(range(raw * k - 1, raw * k + 1), [windowData[raw - 1][k - 1], windowData[raw][0]],
                         color=rgb[cluster.labels_[raw]])
        else:
            plt.plot(range(raw * k, raw * k + k), windowData[raw], color=rgb[cluster.labels_[raw]])
            if raw != 0:
                plt.plot(range(raw * k - 1, raw * k + 1), [windowData[raw - 1][k - 1], windowData[raw][0]],
                         color=rgb[cluster.labels_[raw]])

    plt.xlabel("sample number")
    plt.ylabel("sample value")
    plt.title(title)
    plt.legend()
    plt.savefig(title)

    return list(cluster.labels_)

# calculate transpose matrix and return
def lower(matrix):
    result = np.zeros(shape=(len(matrix), len(matrix[0])))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i < j:
                result[j][i] = matrix[i][j]
                result[i][j] = matrix[i][j]
    return result

# get score1 and score2 as input for calculate merge anomaly score of both two method
def getMergeAnomalyScore(score1: list, score2: list) -> list:
    result = []
    for i in range(len(score1)):
        s1 = (len(score1) - score1.count(score1[i])) / len(score1)
        s2 = (len(score2) - score2.count(score2[i])) / len(score2)
        result.append(round((s1 + s2) / 2, 3))
    return result

# plot merge anomaly score
def plotMergeAnomalyScore(data, score: list, title):
    plt.figure(figsize=(16, 9))
    rgb = getColor(len(set(score)))
    k = len(data[0])

    isLabel = list(set(score))

    for raw in range(len(data)):
        if score[raw] in isLabel:
            isLabel.remove(score[raw])
            plt.plot(range(raw * k, raw * k + k), data[raw], color=rgb[list(set(score)).index(score[raw])],
                     label="count:" + str(score.count(score[raw])) + " score:" + str(score[raw]))
            if raw != 0:
                plt.plot(range(raw * k - 1, raw * k + 1), [data[raw - 1][k - 1], data[raw][0]],
                         color=rgb[list(set(score)).index(score[raw])])
        else:
            plt.plot(range(raw * k, raw * k + k), data[raw], color=rgb[list(set(score)).index(score[raw])])
            if raw != 0:
                plt.plot(range(raw * k - 1, raw * k + 1), [data[raw - 1][k - 1], data[raw][0]],
                         color=rgb[list(set(score)).index(score[raw])])

    plt.xlabel("sample number")
    plt.ylabel("sample value")
    plt.title(title)
    plt.legend()
    plt.savefig(title)
