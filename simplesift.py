import os
import numpy as np
import SimpleCV as cv
import sys

def sift(f, count):
    img = cv.Image(f)
    feats = img.findKeypoints(flavor='SIFT', highQuality=True)
    feats = feats.sortArea()
    #get 10 most important features
    #top_feats = feats[-10:]
    #top_feats.draw()
    feats.draw()

    name = 'sifted' + str(count) + '.jpg'
    path  = os.path.dirname(os.path.abspath(__file__)) + '/processed_data/'
    fullpath = os.path.join(path, name)
    img.save(fullpath)


newpath = os.path.dirname(os.path.abspath(__file__)) + '/processed_data'
if not os.path.exists(newpath):
    os.makedirs(newpath)

count = 0
for filename in os.listdir('caslenses/'):
    count += 1
    if filename.endswith('.jpeg') or filename.endswith('.jpg'):
        #with open('caslenses/' + filename, 'rb') as f:
        f = 'caslenses/' + filename
        sift(f, count)
        print filename