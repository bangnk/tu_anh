import glob
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageFilter
import json
import cv2


def sift():
    img = cv2.imread('shopping/images/img_bags_clutch_653.jpg')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,1)
    cv2.imwrite('tt.png',img)


if __name__ == "__main__":
    sift()