import cv2
import numpy as np
import sys
import os
import copy
from astropy.io import fits

def runSiftJPG(f):
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

def runSiftFits(f,filename):

    mat = f[0].data

    #out = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    mat = mat.astype(np.uint8)

    sift = cv2.SIFT()

    kp = sift.detect(mat, None)

    img = cv2.drawKeypoints(mat, kp)
    name = 'sifted' + str(filename)
    print name
    path  = os.path.dirname(os.path.abspath(__file__)) + '/processed_data/'
    fullpath = os.path.join(path, name)

    #cv2.imwrite(fullpath, img)
    #fcopy = copy.deepcopy(f)
    #fcopy[0].data = img
    fits.writeto(filename, img)



newpath = os.path.dirname(os.path.abspath(__file__)) + '/processed_data'
if not os.path.exists(newpath):
    os.makedirs(newpath)
'''
for filename in os.listdir('data/'):
    with open('data/' + filename) as f:
        runSift(f)
'''

for filename in os.listdir('negative/g/'):
    with fits.open('negative/g/' + filename) as f:
        runSiftFits(f, filename)