from features import *
from sklearn import svm


lens_train = getDescriptors('caslenses/training', 'lens_training_out')
lens_test = getDescriptors('caslenses/test', 'lens_test_out')
gal_train = getDescriptors('galaxies/training', 'gal_training_out')
gal_test = getDescriptors('galaxies/test', 'gal_test_out')


for item in lens_train:
    item.label = 'lens'

for item in gal_train:
    item.label = 'galaxy'

cl = svm.SVC(kernel = 'linear')

des = lens_train + gal_train

train = []
labels = []
for item in des:
    train.append(item.descriptor[0])
    if item.label == 'galaxy':
        labels.append(0)
    else:
        labels.append(1)

cl.fit(train, labels)

print '\n'
print 'Lenses:'
for item in lens_test:
    res = cl.predict(item.descriptor[0])
    print res

print '\n'
print 'Galaxies'
for item in gal_test:
    res = cl.predict(item.descriptor[0])
    print res