from features import *
from sklearn import svm


lens_training_des = getDescriptors('caslenses/training', 'lens_training_out')
lens_test_des = getDescriptors('caslenses/test', 'lens_test_out')
gal_training_des = getDescriptors('galaxies/training', 'gal_training_out')
gal_test_des = getDescriptors('galaxies/test', 'gal_test_out')

classifier = svm.SVC()
