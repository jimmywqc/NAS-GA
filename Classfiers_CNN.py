# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 17:00:13 2022

@author: Jimmy
"""

from __future__ import print_function
import numpy as np
from scipy.io import loadmat 

#load data
DataBase = loadmat('PQD_with_noise.mat')
X = []
Y = []

for i in range (0,16*700):
    X.append(DataBase ['SignalsDataBase'][0][i]['signals'][0])
    Y.append(DataBase ['SignalsDataBase'][0][i]['labels'][0])

x = np.array(X)
x = np.expand_dims(x,axis=2)
y = np.array(Y)
y = y.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder  
OHE = OneHotEncoder()
y_label = OHE.fit_transform(y).toarray()

num = int(x.shape[0]*0.9)
val_num = x.shape[0]-num

x_train = x[0:num,:,:]
y_train = y_label[0:num,:]
x_test = x[val_num:x.shape[0],:,:]
y_test = y_label[val_num:x.shape[0],:]
dataset = ((x_train,y_train),(x_test,y_test))


import numpy as np
from keras import backend as K
import Devol, Genome_handler
import os
import tensorflow as tf
from keras.models import load_model

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#max_dense_nodes>=16  max_filters>=8
genome_handler = Genome_handler.GenomeHandler(max_conv_layers=10, max_dense_layers=10, max_filters=128,
                 max_dense_nodes=256, input_shape=x_train.shape[1:], n_classes=16)
devol = Devol.DEvol(genome_handler)
_,average_acc,best_acc,acc = devol.run(dataset=dataset,num_generation=10,pop_size=10,epochs=50)

best_model = load_model('best_model.h5')
print(best_model.summary())

np.save('acc.npy',acc)

# import matplotlib.pyplot as plt
# x = np.arange(1, 11)
# plt.figure()
# plt.plot(x,average_acc,label='average_acc')
# plt.plot(x,best_acc,label='best_acc')
# plt.legend(loc='upper left')
# plt.show()
print(average_acc)
print(best_acc)
print(acc)

