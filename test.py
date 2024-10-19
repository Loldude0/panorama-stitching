import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
from scipy import ndimage
import random



def detect_corner(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 5000, 0.01, 10)
    cmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in corners:
        y, x = map(int, i.ravel())
        cmap[y, x] = gray[y, x]
    return cmap, corners

def ANMS(cmap, num_best):
    local_max = ndimage.maximum_filter(cmap, size=10)
    max = (cmap == local_max) & (cmap > 0)
    y, x = np.where(max)

    r = np.inf * np.ones(len(y))

    for i in range(len(y)):
        for j in range(len(y)):
            if cmap[y[j], x[j]] > cmap[y[i], x[i]]:
                euclidean_dist = np.sqrt((y[j]-y[i])**2 + (x[j]-x[i])**2)
                if euclidean_dist < r[i]:
                    r[i] = euclidean_dist

    sorted_corners = np.argsort(r)[::-1]
    return [(y[i], x[i]) for i in sorted_corners[:num_best]]

def feature_descript(gray_img, corners):
    feature_descriptors = []

    for corner in corners:
        y, x = corner[0], corner[1]

        if(y - 20 >= 0 and x - 20 >= 0 and y + 20 < gray_img.shape[0] and x + 20 < gray_img.shape[1]):
            patch = gray_img[y - 20:y + 20, x - 20:x + 20]

            blurred_patch = cv2.GaussianBlur(patch, (5, 5), 0)
            blurred_patch = cv2.resize(blurred_patch, (8, 8))

            current_feature = blurred_patch.reshape(64, 1)

            mean = np.mean(current_feature)
            std_dev = np.std(current_feature)
            if(std_dev == 0):  # check for division by 0
                current_feature = (current_feature - mean)
            else:
                current_feature = (current_feature - mean) / std_dev

            feature_descriptors.append(current_feature)

    return np.array(feature_descriptors)

def feature_match(img1, img2, corner1, corner2):
    threshold = 0.8
    desc1 = feature_descript(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), corner1)
    desc2 = feature_descript(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corner2)
    
    matches = []
    for i, desc in enumerate(desc1):
        distances = (np.sum((desc2 - desc) ** 2, axis=1)).flatten()
        sorted_idx = np.argsort(distances)
        calc_ratio = distances[sorted_idx[0]] / distances[sorted_idx[1]]
        if(calc_ratio < threshold):
            matches.append(cv2.DMatch(i, sorted_idx[0], distances[sorted_idx[0]]))
    
    matches.sort(key=lambda x: x.distance)
    print(len(matches))

    keypoints1 = [cv2.KeyPoint(float(x), float(y), 1) for y, x in corner1]  
    keypoints2 = [cv2.KeyPoint(float(x), float(y), 1) for y, x in corner2]  

    img_matches = cv2.drawMatches(img1, keypoints1,
                                  img2, keypoints2,
                                  matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return matches, img_matches, keypoints1, keypoints2

for k in range(1,4):  # 1,4
    image_files = sorted(os.listdir(f'./train_images/Set{k}'))    
    
    for i in range(len(image_files) - 1): 
        img1 = cv2.imread(f'./train_images/Set{k}/{image_files[i]}')
        img2 = cv2.imread(f'./train_images/Set{k}/{image_files[i+1]}')
        
        cmap1, corners1 = detect_corner(img1)
        cmap2, corners2 = detect_corner(img2)
        
        
        sorted_corners1 = ANMS(cmap1, 100)
        sorted_corners2 = ANMS(cmap2, 100)
        
        for y, x in sorted_corners1:
            cv2.circle(img1, (x, y), 3, (0, 255, 0), -1)
        for y, x in sorted_corners2:
            cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)
        
        matches, img_matches, keypoints1, keypoints2 = feature_match(img1, img2, sorted_corners1, sorted_corners2)
        print(matches[0].queryIdx)
        print(matches[0].trainIdx)
        
        plt.figure(figsize=(20, 10))
        plt.subplot(133), plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)), plt.title('Feature Matches'), plt.axis('off')
        plt.tight_layout()
        plt.show()