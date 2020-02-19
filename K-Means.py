import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans

def read_files(directory_path):
    path_list = glob.glob(directory_path)
    files = []
    for path in path_list:
        img = cv2.imread(path)
        #img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)# GRAY_SCALE
        files.append(img)

    return files

def extruct_features(files):

    """
    ref https://qiita.com/hitomatagi/items/62989573a30ec1d8180b
    ORB, GFTT, AKAZE, KAZE, BRISK, SIFTは、特徴点だけではなく、特徴量も計算しています。

    # AgastFeatureDetector
    detector = cv2.AgastFeatureDetector_create()

     # FAST
     detector = cv2.FastFeatureDetector_create()

    # MSER
    detector = cv2.MSER_create()

    # AKAZE
    detector = cv2.AKAZE_create()

    # BRISK
    detector = cv2.BRISK_create()

    # KAZE
    detector = cv2.KAZE_create()

    # ORB (Oriented FAST and Rotated BRIEF)
    detector = cv2.ORB_create()

    # SimpleBlobDetector
    detector = cv2.SimpleBlobDetector_create()

    """
    detector = cv2.ORB_create()

    #features = np.array([])
    features = np.empty((0,16000))
    for img in files:
        kp_base, des_base = detector.detectAndCompute(img, None)
        print(des_base)
        des_base_np = np.array(des_base)
        print(des_base_np.shape)
        des_base_np =des_base_np.reshape(1,-1)
        print(des_base_np.shape)
        features = np.append(features,des_base_np,axis=0)


    print(features)
    return features

def k_means(features):
    n = 4 #クラスター数
    model = KMeans(n_clusters=n,init='k-means++',n_init=10,verbose = 0,random_state=19,n_jobs=-1)

    result = model.fit(features)

    labels = result.labels_

    print(labels)

    return result

if __name__ == '__main__':
    directory_path = "picture/*.*"
    read_files = read_files(directory_path)
    features = extruct_features(read_files)
    result_clustering = k_means(features)





