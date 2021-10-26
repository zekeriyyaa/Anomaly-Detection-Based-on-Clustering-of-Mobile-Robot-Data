## Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data
### 1. Introduction
This report is present an approach to detect anomaly of mobile robot's current and vibration data.  The main idea is examine all data, separate them into two cluster as normal and anomaly and then  using these clustering results figure out the merged anomaly score for each data sample. For this purpose, both of current and vibration data are cluster by using Hierarchical clustering algorithm. Before the clustering there are several preprocessing step that are windowing, feature extraction, dynamic time warping and min-max normalization.  

### 2. Interested Data
There are two different types of data that are coming from mobile robots sensors as current and vibration data. Both of them are produce at same frequency but they have different characteristic. Although the current data is numeric data, the vibration data is time series data. So, current data has a single value per each data packet but vibration data has much more value per each data packet.

Current Data Sample        |  Vibration Data Sample
:-------------------------:|:-------------------------:
![](https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/images_current/wheel2%20Current.png) | ![](https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/image_vibration/Wheel2%20Vibration.png)

### 3. Proposed Method
There are two different method are proposed to detect anomaly on data. They have common step as windowing. And also they have some other different steps like feature extraction, normalization and dynamic time warping. These all are about preprocessing steps. After the preprocessing steps data is clustering into two subset by using hierarchical clustering as normal and anomaly. The anomaly scores of each data sample are produces as a result of clustering. And then, the results of two method are collect and anomaly scores are merge for each same data sample. While merging anomaly score, the mean of them are take. Given two method is perform separately using both current and vibration data. Proposed method is shown as below.

![](https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/System-Architecture.PNG)

Rest of here, method 1 is represent a method which is use feature extraction and method 2 is also represent a method which is use DTW. Remember that both of these methods have also common steps.

#### 3.1 Preprocessing Steps

**A. Windowing** <br>
In this process, the data are parsed into subsets named as window with same size. For the extract of features of data, the data must be a time series data. In this way, the data are converted time series data. In this project, window size is 3. This step is implement for both two methods. Sample windowing process output is shown as below:

<img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/windowing.PNG" align="center" width="600" height="300">

**B. Feature Extraction** <br>
The features are extracted separately for each window. There are nine different feature as given below:

**C. Dynamic Time Warping** <br>
In method 2, DTW is used for calculate similarity instead of Euclidean distance. After the windowing process, the data was converted time series data. So now, it is possible to use DTW on data. 

Feature Extraction         |  Dynamic Time Warping
:-------------------------:|:-------------------------:
<img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/features.PNG" width="400" height="300"> | <img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/dtw.png" width="400" height="300">

**D. Min-Max Normalization** <br>
Min-max normalization is one of the most common ways to normalize data. For every feature, the minimum value of that feature gets transformed into a 0, the maximum value gets transformed into a 1, and every other value gets transformed into a decimal between 0 and 1. Min-max normalization is executed on features that extracted from window. This step is implement only for method 1.

#### 3.2 Hierarchical Clustering
This clustering technique is divided into two types as agglomerative and divisive. In this method, agglomerative approach is used. At this step, preprocessing steps is already done for method 1 and method 2 and the windows are ready to clustering. These windows are put into hierarchical algorithm to find clusters. As a result, the clusters which windows are belong to are found. They are used for calculate the anomaly score for whole data. This step is implemented for both two methods. And, the dendrogram which is represent the clustering result is produce.

#### 3.3 Find Anomaly Score
The anomaly score is calculated separately from result of hierarchical clustering of both method 1 and method 2. The hierarchical clustering algorithm is produce clusters for each window. With use these clusters, the anomaly score is calculated for each cluster as given below (C: interested cluster, #All window: number of all window, #C window: number of window that belong to cluster C): ```C_anomaly=(#All Window - #C Window)/(#All Window)```
<br<
After the calculation of anomaly score for each method, the merged anomaly score is generate from mean of them. The formula is as follows for generate merged score:
```C_(merged anomaly score)=(C_(anomaly of method1)+ C_(anomaly of method2))/2``` 
<br>
The anomaly score which is higher mean it is highly possible to be anomaly.

### 4. Experiments
An anomaly score is located right-top of figure. Different clusters are shown with different color.
#### Current Data Results

Feature Extracted Clustering Anomaly Score|  DTW Clustering Anoamly Score
:----------------------------------------:|:---------------------------------:
<img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/images_current/Wheel2%20-%20Feature%20Extracted%20-%20Anomaly%20Score%20with%20-%20n_clusters%202%20-%20window_size%202.png" width="600" height="300"> | <img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/images_current/Wheel2%20-%20Dynamic%20Time%20Warping%20-%20Anomaly%20Score%20-%20n_clusters%202%20-%20window_size%202.png" width="600" height="300">

Merged Anomaly Score
:---------------------------:
<img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/images_current/Wheel2%20-%20Merged%20Anomaly%20Score%20-%20n_clusters%202%20-%20window_size%202.png" width="800" height="400">

#### Vibration Data Results

Feature Extracted Clustering Anomaly Score|  DTW Clustering Anoamly Score
:----------------------------------------:|:---------------------------------:
<img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/image_vibration/Wheel2%20-%20Feature%20Extracted%20-%20Anomaly%20Score%20with%20-%20n_clusters%202%20-%20window_size%202.png" width="600" height="300"> | <img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/image_vibration/Wheel2%20-%20Dynamic%20Time%20Warping%20-%20Anomaly%20Score%20-%20n_clusters%202%20-%20window_size%202.png" width="600" height="300">

Merged Anomaly Score
:---------------------------:
<img src="https://github.com/zekeriyyaa/Anomaly-Detection-Based-on-Hierarchical-Clustering-of-Mobile-Robot-Data/blob/main/image_vibration/Wheel2%20-%20Merged%20Anomaly%20Score%20-%20n_clusters%202%20-%20window_size%202.png" width="800" height="400">



