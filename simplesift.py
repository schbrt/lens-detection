import os
import numpy as np
import SimpleCV as cv
import sys

def sift(f, count):
    img = cv.Image(f)
    feats = img.findKeypoints(min_quality=1000.0, flavor='SURF', highQuality=True)
    #rank features in ascending order
    feats = feats.sortArea()

    feats.draw()

    name = 'sifted' + str(count) + '.jpg'
    path  = os.path.dirname(os.path.abspath(__file__)) + '/processed_data/'
    fullpath = os.path.join(path, name)
    img.save(fullpath)
    des = []
    for feat in feats:
        des.append(feat.descriptor())
    return des

newpath = os.path.dirname(os.path.abspath(__file__)) + '/processed_data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

count = 0
for filename in os.listdir('caslenses/'):
    count += 1
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        #with open('caslenses/' + filename, 'rb') as f:
        f = 'caslenses/' + filename
        img = sift(f, count)
        print filename
        print img