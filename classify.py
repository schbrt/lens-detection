import features as ft
from sklearn import svm
import cv2
import numpy as np
import bow
#import svm

lens_train = ft.getFeats('caslenses/training', 'lens_training_out')
lens_test = ft.getFeats('caslenses/test', 'lens_test_out')
gal_train = ft.getFeats('galaxies/training', 'gal_training_out')
gal_test = ft.getFeats('galaxies/test', 'gal_test_out')


for item in lens_train:
    item.label = 'lens'

for item in gal_train:
    item.label = 'galaxy'

#sklearn version
#cl = svm.SVC()

#
images = lens_train + gal_train

#playing around with BOW
dlist = bow.feat_list(images)
km = bow.cluster(dlist)


train = []
labels = []
for item in images:
    feats = km.predict(item.features)
    train.append(feats)
    if item.label == 'galaxy':
        labels.append(0)
    else:
        labels.append(1)


#cl.fit(train, labels)

#opencv version
cl = cv2.SVM()
cl.train(np.asarray(train), np.asarray(labels))

print '\n'
print 'Lenses:'
avg1,avg2 = 0, 0
for item in lens_test:
    curr = item.features
    n = len(curr)
    m = len(curr[0])
    curr = np.asarray(curr)
    curr =curr.reshape(1, n * m)
    print curr.shape
    res = cl.predict(curr)
    print res


print '\n'
print 'Galaxies'
avg1, avg2 = 0, 0
for item in gal_test:
    res = cl.predict(item.features)
    print res

print cl.n_support_