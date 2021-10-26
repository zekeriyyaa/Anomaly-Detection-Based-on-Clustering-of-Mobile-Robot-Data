from clustringFuctions import *

if __name__ == '__main__':

    ### Parameter Initialization - Process start ###
    window_size = 2  # for data preparing
    method = "single"  # for all hierarchical clusterig process
    metric = "euclidean"  # for featured extraction process
    ### Parameter Initialization - Process end ###

    ### Data Preparing - Process start ###
    fileName1 = "motor1_vib.csv"
    fileName2 = "motor2_vib.csv"

    # get whole data from csv for wheel-1
    data = getCsvData(fileName1)
    # parse whole data and get just vibration data
    # this return list with 2 size
    data = prepareVibrationData(data, "VibrationZ")
    # seperated vibration data
    wheel1 = data[0]
    # vibration data which added one after the other
    wheel1Flatten = data[1]

    # get whole data from csv for wheel-2
    data = getCsvData(fileName2)
    # parse whole data and get just vibration data
    # this return list with 2 size
    data = prepareVibrationData(data, "VibrationZ")
    # seperated vibration data
    wheel2 = data[0]
    # vibration data which added one after the other
    wheel2Flatten = data[1]


    # reshape wheel1-vibration data
    wheel1 = np.array(wheel1).reshape(len(wheel1), -1)
    # reshape wheel2-vibration data
    wheel2 = np.array(wheel2).reshape(len(wheel2), -1)


    # wheel 1
    # windowing process
    wheel1Parsed = parseVib2Window(wheel1, window_size)
    # feature extraction process
    wheel1Feature = getFeaturedData(wheel1Parsed)
    # normalization process
    wheel1Normalized = getNomalizedData(wheel1Feature)

    # wheel 2
    # windowing process
    wheel2Parsed = parseVib2Window(wheel2, window_size)
    # feature extraction process
    wheel2Feature = getFeaturedData(wheel2Parsed)
    # normalization process
    wheel2Normalized = getNomalizedData(wheel2Feature)
    ### Data Preparing - Process end ###


    ### Default Plotting - Process start ###
    plotVibration(wheel1Flatten, "Wheel1 Vibration")
    plotVibration(wheel2Flatten, "Wheel2 Vibration")
    ### Default Plotting - Process end ###


    # get clustering results for #cluster 2,3,4
    for n_cluster in range(2, 5):
        # wheel1
        title = "Wheel1 - Dynamic Time Warping - Anomaly Score - n_clusters " + str(n_cluster) + " - window_size " + str(window_size)
        # get anomaly score of dynamic time warping method's result
        w1AnomalyScoreDTW = plotAnomalyScoreofDTW(wheel1Parsed, wheel1Parsed, method, title, n_cluster)

        title = "Wheel1 - Feature Extracted - Anomaly Score with - n_clusters " + str(n_cluster) + " - window_size " + str(window_size)
        # get anomaly score of feature extraction method's result
        w1AnomalyScoreFeatured = plotAnomalyScoreofFeatured(wheel1Normalized, wheel1Parsed, method, metric, title,n_cluster)

        title = "Wheel1 - Merged Anomaly Score - n_clusters " + str(n_cluster) + " - window_size " + str(window_size)
        # get merged anomaly score
        score = getMergeAnomalyScore(w1AnomalyScoreDTW, w1AnomalyScoreFeatured)
        # plot total results
        plotMergeAnomalyScore(wheel1Parsed, score, title)

        #wheel2
        title = "Wheel2 - Dynamic Time Warping - Anomaly Score - n_clusters " + str(n_cluster) + " - window_size " + str(window_size)
        # get anomaly score of dynamic time warping method's result
        w2AnomalyScoreDTW = plotAnomalyScoreofDTW(wheel2Parsed, wheel2Parsed, method, title, n_cluster)

        title = "Wheel2 - Feature Extracted - Anomaly Score with - n_clusters " + str(n_cluster) + " - window_size " + str(window_size)
        # get anomaly score of feature extraction method's result
        w2AnomalyScoreFeatured = plotAnomalyScoreofFeatured(wheel2Normalized, wheel2Parsed, method, metric, title,n_cluster)

        title = "Wheel2 - Merged Anomaly Score - n_clusters " + str(n_cluster) + " - window_size " + str(window_size)
        # get merged anomaly score
        score = getMergeAnomalyScore(w2AnomalyScoreDTW, w2AnomalyScoreFeatured)
        # plot total results
        plotMergeAnomalyScore(wheel2Parsed, score, title)


    print("End Of App")
