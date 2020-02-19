import numpy as np
import cv2
from matplotlib import pyplot as plt

file_path_base = "picture/1.jpg" ##画像の場所
file_path_comp = "picture/5.jpg"
print(file_path_base)
print(file_path_comp)

img_base = cv2.imread(file_path_base)
img_comp = cv2.imread(file_path_comp)

# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints and detect with ORB
kp_base,des_base = orb.detectAndCompute(img_base, None)
kp_comp,des_comp = orb.detectAndCompute(img_comp, None)

print("base:: orb-des")
print(des_base)

# Brute-Force Matcher生成
bf = cv2.BFMatcher()

# 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
matches = bf.knnMatch(des_base, des_comp, k=2)

# データを間引きする
ratio = 1.0
good = []
for m, n in matches:
    if m.distance < ratio * n.distance:
        good.append([m])

# 対応する特徴点同士を描画
img_result = cv2.drawMatchesKnn(img_base, kp_base, img_comp, kp_comp, good, None, flags=2)

#ref https://qiita.com/kenfukaya/items/dfa548309c301c7087c4
height = img_result.shape[0]
width = img_result.shape[1]

img_result_resize = cv2.resize(img_result, (int(width*0.1), int(height*0.1)))

# 画像表示
cv2.imshow('img', img_result_resize)

# キー押下で終了
cv2.waitKey(0)
cv2.destroyAllWindows()



