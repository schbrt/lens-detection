import cv2
import numpy as np
import sys
import os

def runSift(f):
    #img = cv2.imread(str(f))
    img = cv2.imread(f.name)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if output has 'order' BGR, otherwise RGB

    sift = cv2.SIFT()

    kp = sift.detect(out, None)

    img = cv2.drawKeypoints(out, kp)
    name = 'sifted' + str(f.name)[5:]
    path  = os.path.dirname(os.path.abspath(__file__)) + '/processed_data/'
    fullpath = os.path.join(path, name)

    cv2.imwrite(fullpath, img)

newpath = os.path.dirname(os.path.abspath(__file__)) + '/processed_data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

for filename in os.listdir('data/'):
    with open('data/' + filename) as f:
        runSift(f)