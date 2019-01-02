#!/usr/bin/env python
"""
分类器
"""
import os 
import util
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import glob
import time
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from skimage.feature import hog
from sklearn.externals import joblib
import pickle
from sklearn.grid_search import GridSearchCV

notcars = glob.glob('non-vehicles/*/*.png')
cars = glob.glob('vehicles/*/*.png')


colorspace = 'YUV'
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
t = time.time()
car_features = util.extract_features(cars, cspace=colorspace, orient=orient,
									 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
									 hog_channel=hog_channel)

notcar_features = util.extract_features(notcars, cspace=colorspace, orient=orient,
										pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
										hog_channel=hog_channel)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extrace features...')

X = np.vstack((car_features, notcar_features))
X = X.astype(np.float64)

#数据缩放与转换
#X_scaler = StandardScaler().fit(X)
#scaled_X = X_scaler.transform(X) 

y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

#数据划分
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

print('Feature vector length:', len(X_train[0]))


svc = LinearSVC()

t3 = time.time()
svc.fit(X_train, y_train)
t4 = time.time()
print(round(t4-t3, 2), 'Seconds to train classfier...')

print('Test Accuracy of classfier = ', round(svc.score(X_test, y_test),  4))

t5 = time.time()

n_predict = 10

print('My classfier predicts:', svc.predict(X_test[0:n_predict]))
print('For these ', n_predict, 'labels: ', y_test[0:n_predict])
t6 = time.time()
print(round(t6-t5, 5), 'Seconds to predict ', n_predict, 'labels with classfier')

train_dist={}
train_dist['clf'] = svc
train_dist['scaler'] = None 
train_dist['orient'] = orient
train_dist['pix_per_cell'] = pix_per_cell
train_dist['cell_per_block'] = cell_per_block
train_dist['hog_channel'] = hog_channel
train_dist['spatial_size'] = None
train_dist['hist_bins'] = None

output = open('train_dist.p', 'wb')
pickle.dump(train_dist, output)























