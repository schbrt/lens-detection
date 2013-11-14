from features import *
from sklearn import svm


lens_training_data = getDescriptors('caslenses/training', 'lens_training_out')
lens_test_data = getDescriptors('caslenses/test', 'lens_test_out')
gal_training_data = getDescriptors('galaxies/training', 'gal_training_out')
gal_test_data = getDescriptors('galaxies/test', 'gal_test_out')


for item in lens_training_data:
    item.label = 'lense'

for item in gal_training_data:
    item.label = 'galaxy'

classifier = svm.SVC()

training_data = lens_training_data + gal_training_data