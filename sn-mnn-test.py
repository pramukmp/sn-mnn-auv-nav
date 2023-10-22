from MemoryNetwork import MemoryNeuralNetwork
import numpy as np
import pandas as pd
import os
from numpy import load
from numpy import transpose
from numpy import cos
from numpy import sin
from numpy.random.mtrand import random
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import sys
import pickle
import matplotlib.pyplot as plt

print("Loading Data...")

neeta=1e-5
neeta_dash=5e-7
lipschitz_constant = 1.0
epochs = 50

filename = open("/home/pramuk/IISC/sn-mnn-auv-nav/trained_models_mnn/sn-mnn.obj", "rb")
mnn = pickle.load(filename)
filename.close()

IMU_in = load('/home/pramuk/IISC/sn-mnn-auv-nav/dataset/Test/IMU_in_test.npy')
V = load('/home/pramuk/IISC/sn-mnn-auv-nav/dataset/Test/V_test.npy')

b1 = np.array([cos((45 + 0 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 0 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
b2 = np.array([cos((45 + 1 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 1 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
b3 = np.array([cos((45 + 2 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 2 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])
b4 = np.array([cos((45 + 3 * 90) * np.pi / 180) * sin(20 * np.pi / 180),
               sin((45 + 3 * 90) * np.pi / 180) * sin(20 * np.pi / 180), cos(20 * np.pi / 180)])

A = np.array([b1, b2, b3, b4]).reshape((4, 3))
p_inv = np.matmul(inv(np.matmul(transpose(A), A)), transpose(A))
beams = np.zeros((len(V[0, :]), 4))
for i in range(0, len(V[0, :])):
    beams[i, :] = np.matmul(A, (V[:, i]) * (1 + 0.007))  # scale factor 0.7%

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(V[0, :]), 4) + \
              0.001 * np.ones((len(V[0, :]), 4))

T = 100
X_gyro = np.zeros((len(IMU_in[0, :, 0]) // T, 3))
X_acc = np.zeros((len(IMU_in[0, :, 0]) // T, 3))
Y = np.zeros((len(IMU_in[0, :, 0]) // T, 4))
Z = np.zeros((len(IMU_in[0, :, 0]) // T, 3))

n = 0
V = V.T

for t in range(0, len(IMU_in[0, :, 0]) - 1, T):
    X_acc[n, :] = IMU_in[:, t, 0]
    X_gyro[n, :] = IMU_in[:, t, 1]
    y = beams_noise[n, :]
    Y[n, :] = y
    z = V[n, :]
    Z[n, :] = z
    n = n + 1
#print(Y)
# Num of DVL samples
N = len(IMU_in[0, :, 0]) // T
print(X_acc.shape,X_gyro.shape,Y.shape)

test_data=np.concatenate((X_acc,X_gyro,Y),axis=1)
test_data_df=pd.DataFrame(test_data)
print(test_data_df.head())

actual_data = np.zeros((len(test_data)-1,3))
predicted_data = np.zeros((len(test_data)-1,3))

for i in range(1, len(test_data)):
    mnn.feedforward(test_data[i-1,:])    
    actual_data[i-1,:] = Z[i-1,:]
    predicted_data[i-1,:] = mnn.output_nn

avg_rmse=(mean_squared_error(actual_data,predicted_data,squared=False))
rmse_x = (np.sum(np.square(actual_data[:,0] - predicted_data[:,0])) / len(test_data)) ** 0.5
rmse_y = (np.sum(np.square(actual_data[:,1] - predicted_data[:,1])) / len(test_data)) ** 0.5
rmse_z = (np.sum(np.square(actual_data[:,2] - predicted_data[:,2])) / len(test_data)) ** 0.5

print("========================\n RMSE x (m/s): %5.5f\n RMSE y (m/s): %5.5f\n RMSE z (m/s): %5.5f\n ========================" % (rmse_x, rmse_y, rmse_z))
print("overall rmse",avg_rmse)
np.savetxt("/home/pramuk/IISC/sn-mnn-auv-nav/outputs/predicted_position.csv", predicted_data, delimiter=",")