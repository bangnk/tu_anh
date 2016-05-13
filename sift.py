import glob
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageFilter
import json
import cv2
import matplotlib.pyplot as plt


def sift():
    file = 'shopping/images/img_bags_clutch_995.jpg'
    img = cv2.imread(file)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)

    # file1 = 'shopping/images/img_bags_clutch_10.jpg'
    # img1 = cv2.imread(file1)
    # gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # kp1, des1 = sift.detectAndCompute(gray1,None)
    #
    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1,des)
    # matches = sorted(matches, key = lambda x:x.distance)
    # img3 = cv2.drawMatches(img1,kp1,img,kp,matches[:10], flags=2)
    # plt.imshow(img3),plt.show()

    img=cv2.drawKeypoints(img,kp,1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(file.split('/')[-1],img)


if __name__ == "__main__":
    sift()