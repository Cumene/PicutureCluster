import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sklearn
from sklearn.cluster import KMeans
import pickle
import time

def read_files(directory_path,gray_scale=False,resize=True):
    path_list = glob.glob(directory_path)
    files = []
    filename = 'picture'
    for path in path_list:
        if gray_scale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # GRAY_SCALE
        else:
            img = cv2.imread(path)

        if resize:
            img = cv2.resize(img,dsize=(640,480))

        files.append(img)

    if gray_scale:
        filename = filename + "-gray"
    else:
        filename = filename + "-color"

    if resize:
        filename = filename + "-size-640-480"
    else:
        filename = filename + "-origin"

    filename = filename + "-" + format_str_time() + ".pkl"
    save_pkl(filename,files)

    return files

def read_files_pke(pkl_file):
    load_file = load_pkl(pkl_file)
    return load_file

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
        #print(des_base)
        des_base_np = np.array(des_base)
        #print(des_base_np.shape)
        des_base_np =des_base_np.reshape(1,-1)
        #print(des_base_np.shape)
        features = np.append(features,des_base_np,axis=0)




    print(features)
    return features

def extruct_features_dense(files):


    return features

def k_means(features):

    k = 1000 #クラスター数

    for n in range(2,k+1):
        print("cluster:"+str(n))
        model = KMeans(n_clusters=n, init='k-means++', n_init=10, verbose=0, random_state=19, n_jobs=-1)

        result = model.fit(features)

        filename = 'model-cluster-' + str(n) + +"-" + format_str_time() + ".pkl"

        save_pkl(filename, model)


    #result_load = load_pkl(filename)

    #labels = result_load.labels_
    #inertia = result_load.inertia_

    #print(labels)

    return result

def format_str_time():
    now = time.ctime()
    cnvtime = time.strptime(now)
    format_time = time.strftime("%Y-%m-%d-%H-%M-%S", cnvtime)
    return str(format_time)

def save_pkl(filename,data):
    with open(filename, 'wb') as save_pkl:
        pickle.dump(data,save_pkl)

def load_pkl(filename):
    with open(filename,'rb') as load_pkl:
        load_file = pickle.load(load_pkl)
        return load_file

if __name__ == '__main__':
    directory_path = "picture/*.*"
    read_files = read_files(directory_path,False,False)
    #read_files = read_files_pke("picture-color-size-640-480-2020-02-20-01-52-41.pkl")
    features = extruct_features(read_files)
    result_clustering = k_means(features)





