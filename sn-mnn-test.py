from mnn_torch import MemoryNeuralNetwork
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

filename = open("/home/pramuk/IISC/sn-mnn-auv-nav/trained_models_mnn/sn-mnn.obj", "rb")
mnn = pickle.load(filename)
filename.close()

IMU_in = load('/home/pramuk/IISC/sn-mnn-auv-nav/dataset/Test/IMU_in_test.npy')
V = load('/home/pramuk/IISC/sn-mnn-auv-nav/dataset/Test/V_test.npy')

resampled_dvl_data = np.zeros((len(IMU_in[0, :, 0]), 3))
X_gyro = np.zeros((len(IMU_in[0, :, 0]), 3))
X_acc = np.zeros((len(IMU_in[0, :, 0]), 3))
Y = np.zeros((len(IMU_in[0, :, 0]), 4))
Z = np.zeros((len(IMU_in[0, :, 0]), 3))

#print(np.linspace(df[0][0],df[0][1],num=100))
n=0

for i in range(0,len(IMU_in[0,:,0])-100,100):
    for axis in [0,1,2]:                
        resampled_dvl_data[i:i+100,axis]=np.linspace(V[axis][n],V[axis][n+1],100)
    n=n+1
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
beams = np.zeros((len(resampled_dvl_data[:, 0]), 4))

for i in range(0, len(resampled_dvl_data[:, 0])):
    beams[i, :] = np.matmul(A, (resampled_dvl_data[i, :])
                            * (1 + 0.007))  # scale factor 0.7%

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(resampled_dvl_data[:,0]), 4) + \
    0.001 * np.ones((len(resampled_dvl_data[:,0]), 4))

X_acc[:,:]=IMU_in[:,:,0].transpose()
X_gyro[:,:]= IMU_in[:,:, 1].transpose()
Y=beams_noise[:,:]
Z=resampled_dvl_data[:, :]
#print(Y)
# Num of DVL samples
N = len(IMU_in[0, :, 0])
print(X_acc.shape,X_gyro.shape,Y.shape)

test_data=np.concatenate((X_acc,X_gyro,Y),axis=1)
test_data_df=pd.DataFrame(test_data)
output_data_df=pd.DataFrame(Z)
print(test_data_df.head())

input_data_csv = test_data_df.to_csv(
    '/home/pramuk/IISC/sn-mnn-auv-nav/inputs/Test/input_data_test.csv')
target_var_csv=output_data_df.to_csv('/home/pramuk/IISC/sn-mnn-auv-nav/inputs/Test/target_var_test.csv')

actual_data = np.zeros((len(test_data)-1,3))
predicted_data = np.zeros((len(test_data)-1,3))

for i in range(1, len(test_data)):
    mnn.feedforward(test_data[i-1,:])    
    actual_data[i-1,:] = Z[i-1,:]
    predicted_data[i-1,:] = mnn.output_nn.cpu().detach().numpy()

avg_rmse=(mean_squared_error(actual_data,predicted_data,squared=False))
rmse_x = (np.sum(np.square(actual_data[:,0] - predicted_data[:,0])) / len(test_data)) ** 0.5
rmse_y = (np.sum(np.square(actual_data[:,1] - predicted_data[:,1])) / len(test_data)) ** 0.5
rmse_z = (np.sum(np.square(actual_data[:,2] - predicted_data[:,2])) / len(test_data)) ** 0.5

print("========================\n RMSE x (m/s): %5.5f\n RMSE y (m/s): %5.5f\n RMSE z (m/s): %5.5f\n ========================" % (rmse_x, rmse_y, rmse_z))
print("overall rmse",avg_rmse)
np.savetxt("/home/pramuk/IISC/sn-mnn-auv-nav/outputs/predicted_position.csv", predicted_data, delimiter=",")
plt.plot(predicted_data[:,0])
plt.plot(actual_data[:,0])