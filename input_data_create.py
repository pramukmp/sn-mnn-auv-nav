import numpy as np
import pandas as pd
import os
import sys
from numpy import cos
from numpy import sin
from numpy import transpose
from numpy.linalg import inv
from numpy import load

IMU_in = load(
    '/home/pramuk/IISC/sn-mnn-auv-nav/dataset/TrainAndValidation/IMU_in.npy')
V = load('/home/pramuk/IISC/sn-mnn-auv-nav/dataset/TrainAndValidation/V.npy')

print(IMU_in[2, :, 0].max(), IMU_in[2, :, 1].max())


print(V)
n = 0
resampled_dvl_data = np.zeros((len(IMU_in[0, :, 0]), 3))

for axis in [0, 1, 2]:
    resampled_dvl_data[0:100, axis] = np.linspace(V[axis][0], V[axis][0], 100)

for i in range(100, len(IMU_in[0, :, 0]), 100):
    for axis in [0, 1, 2]:
        resampled_dvl_data[i:i+100,
                           axis] = np.linspace(V[axis][n], V[axis][n+1], 100)
    n = n+1

# DVL speed to beams  - pitch=20 deg eqn 4 substitution from paper
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

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(resampled_dvl_data[:, 0]), 4) + \
    0.001 * np.ones((len(resampled_dvl_data[:, 0]), 4))

T = 100
X_gyro = np.zeros((len(IMU_in[0, :, 0]), 3))
X_acc = np.zeros((len(IMU_in[0, :, 0]), 3))
Y = np.zeros((len(IMU_in[0, :, 0]), 4))
Z = np.zeros((len(IMU_in[0, :, 0]), 3))

n = 0
X_acc = IMU_in[:, :, 0]
X_gyro = IMU_in[:, :, 1]
y = beams_noise[:, :]
Z = resampled_dvl_data[:, :]

print(X_acc.shape, X_gyro.shape, y.shape)

print(Z)
# Num of DVL samples

N = len(IMU_in[0, :, 0])

# print(X_acc[:, 2].max(), X_gyro[:, 2].max(), Y.max())
input_data = np.concatenate((X_acc.T, X_gyro.T, y), axis=1)
input_data_df = pd.DataFrame(input_data)
output_data_df = pd.DataFrame(Z)
print(input_data_df.head())

input_data_csv = input_data_df.to_csv(
    '/home/pramuk/IISC/sn-mnn-auv-nav/inputs/input_data.csv')
target_var_csv=output_data_df.to_csv('/home/pramuk/IISC/sn-mnn-auv-nav/inputs/target_var.csv')