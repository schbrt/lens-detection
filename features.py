import cv2
import numpy as np
import sys
import os
import copy
from astropy.io import fits

class Image:
    def __init__(self, name, mat, des, label=None):
        self.name = name
        self.label = label
        self.matrix = mat
        self.descriptor = des


def runSiftJPG(f, count, tar):
    img = cv2.imread(f.name)

    #is grayscale conversion necessary?
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if output has 'order' BGR, otherwise RGB

    #instantiate feature detector
    #sift = cv2.SIFT()
    surf = cv2.SURF(hessianThreshold =500.0)


    #kp, desc = sift.detectAndCompute(out, None)
    kp, desc = surf.detectAndCompute(img, None)

    #change img to out if grayscale
    img = cv2.drawKeypoints(out, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    name = 'sifted' + str(count) + '.jpg'


    path  = os.path.dirname(os.path.abspath(__file__)) + '/' + str(tar)
    fullpath = os.path.join(path, name)
    cv2.imwrite(fullpath, img)

    ret = Image(f.name, img, desc)
    return ret

'''
Returns an array of descriptors
'''
def getDescriptors(src, tar):
    newpath = os.path.dirname(os.path.abspath(__file__)) + '/' + str(tar)
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    imageList = []
    count = 0
    srcDir = src + '/'
    for filename in os.listdir(str(srcDir)):
        count += 1
        if filename.endswith('.jpeg') or filename.endswith('.jpg'):
            with open(str(srcDir) + filename, 'rb') as f:
                image = runSiftJPG(f, count, tar)
                imageList.append(image)
                print filename

    return imageList

'''
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
    fits.writeto('processed_data/' + filename, img)


    for filename in os.listdir('FITS files/positive/g'):
        with fits.open('FITS files/positive/g/' + filename) as f:
            runSiftFits(f, filename)


'''

