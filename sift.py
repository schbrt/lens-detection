import cv2
import numpy as np
import sys
import os
import copy
from astropy.io import fits

def runSiftJPG(f, count):
    img = cv2.imread(f.name)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #if output has 'order' BGR, otherwise RGB
    sift = cv2.SIFT()


    kp = sift.detect(out, None)

    #img = cv2.drawKeypoints(out, kp)
    img = cv2.drawKeypoints(out,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    name = 'sifted' + str(count) + '.jpg'



    path  = os.path.dirname(os.path.abspath(__file__)) + '/processed_data/'
    fullpath = os.path.join(path, name)
    cv2.imwrite(fullpath, img)



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
'''

newpath = os.path.dirname(os.path.abspath(__file__)) + '/processed_data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

count = 0
for filename in os.listdir('caslenses/'):
    count += 1
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        with open('caslenses/' + filename, 'rb') as f:
            runSiftJPG(f, count)
            print filename
'''
for filename in os.listdir('FITS files/positive/g'):
    with fits.open('FITS files/positive/g/' + filename) as f:
        runSiftFits(f, filename)
'''





