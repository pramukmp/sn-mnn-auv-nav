################## Imports ##################

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
from MemoryNetwork import MemoryNeuralNetwork
import matplotlib.pyplot as plt
from sklearn.utils import resample
# Seeds

np.random.seed(0)


################## Functions ##################

def RMSE(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    rmse_ls = np.sqrt(mean_squared_error(true, LS))
    rmse_predicted = np.sqrt(mean_squared_error(true, predicted))
    improv = 100 * (1 - (rmse_predicted / rmse_ls))
    return rmse_ls, rmse_predicted, improv


def MAE(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    mae_ls = np.sum(np.abs(LS - true)) / len(true)
    mse_predicted = np.sum(np.abs(predicted - true)) / len(true)
    return mae_ls, mse_predicted


def NSE_R2(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    true_avg = np.mean(true)
    temp_ls = np.sum((LS - true) ** 2) / np.sum((true - true_avg) ** 2)
    r2_ls = 1 - temp_ls
    temp_ls = np.sum((predicted - true) ** 2) / np.sum((true - true_avg) ** 2)
    r2_predicted = 1 - temp_ls
    return r2_ls, r2_predicted


def VAF(true, predicted, LS):
    true = LA.norm(true, axis=1)
    predicted = LA.norm(predicted, axis=1)
    LS = LA.norm(LS, axis=1)
    true_var = np.var(true)
    temp_ls = np.var(true - LS)
    r2_ls = (1 - temp_ls / true_var) * 100
    temp_predicted = np.var(true - predicted)
    r2_predicted = (1 - temp_predicted / true_var) * 100
    return r2_ls, r2_predicted


################## DNN ##################
neeta = 0.0012
neeta_dash = 0.0005
lipschitz_constant = 1.2
epochs = 50
pred_list = []

mnn = MemoryNeuralNetwork(10, 20, 3, neeta=neeta, neeta_dash=neeta_dash,
                          lipschitz_norm=lipschitz_constant, spectral_norm=True)

################# MAIN ################

# load
path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
IMU_in = load(
    '/home/pramuk/IISC/sn-mnn-auv-nav/dataset/TrainAndValidation/IMU_in.npy')
V = load('/home/pramuk/IISC/sn-mnn-auv-nav/dataset/TrainAndValidation/V.npy')

print(IMU_in[2, :, 0].max(), IMU_in[2, :, 1].max())


print(V)
n=0
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

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(resampled_dvl_data[:,0]), 4) + \
    0.001 * np.ones((len(resampled_dvl_data[:,0]), 4))

T = 100
X_gyro = np.zeros((len(IMU_in[0, :, 0]), 3))
X_acc = np.zeros((len(IMU_in[0, :, 0]), 3))
Y = np.zeros((len(IMU_in[0, :, 0]), 4))
Z = np.zeros((len(IMU_in[0, :, 0]), 3))

n = 0
X_acc=IMU_in[:,:,0]
X_gyro= IMU_in[:,:, 1]
y=beams_noise[:,:]
Z=resampled_dvl_data[:, :]
print(X_acc.shape,X_gyro.shape,y.shape)

print(Z)
# Num of DVL samples

N = len(IMU_in[0, :, 0])

# print(X_acc[:, 2].max(), X_gyro[:, 2].max(), Y.max())
input_data = np.concatenate((X_acc.T, X_gyro.T, y), axis=1)
input_data_df = pd.DataFrame(input_data)
print(input_data_df.head())

input_data_csv = input_data_df.to_csv(
    '/home/pramuk/IISC/sn-mnn-auv-nav/inputs/input_data.csv')

error_sum = 0.0
error_list = np.zeros(epochs)
rmse_list = np.zeros(epochs)
unstable_flag = False
rmse_sum = 0.0

try:
    for _ in range(0, 2000):
        mnn.feedforward(np.zeros(10))
        mnn.backprop(np.zeros(3))
    # for i in range(10):
    #     mnn.feedforward(input_data[1, :])
    #     mnn.backprop(Z[1, :])
    for epoch in range(0, epochs):
        pred_list = []
        if (epoch != 0):
            print("Training for epoch %2d finished with average rmse loss of %.5f\n" % (
                epoch-1, rmse_sum/len(input_data)))
            error_list[epoch] = error_sum / len(input_data)
            rmse_list[epoch] = rmse_sum/len(input_data)
        error_sum = 0
        rmse_sum = 0
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        for i in range(1, len(input_data)):

            if (mnn.squared_error > 1e30):

                unstable_flag = True
                # double break
                i = sys.maxint
                epoch = sys.maxint

            pred = mnn.feedforward(input_data[i-1, :])
            mnn.backprop(Z[i-1, :])

            # print(pred,Z[i-1,:])
            if (epoch > 45):
                pred_list.append(pred)

            error_sum += mnn.squared_error
            rmse_sum += mnn.rmse

            print("Training for epoch %2d, progress %5.2f%% with squared loss: %.5f with rmse: %.5f rmse x: %.5f rmse y: %.5f rmse z: %.5f" % (
                epoch, (i/len(input_data)) * 100, mnn.squared_error, mnn.rmse,mnn.rmse_x,mnn.rmse_y,mnn.rmse_z), end="\r")
        # print(pred_list)

    print("=====================================================================================================")

except Exception as e:
    # print(e.__doc__)
    print(e)

finally:
    save_path = '/home/pramuk/IISC/sn-mnn-auv-nav/trained_models_mnn/sn-mnn.obj'
    if not unstable_flag:
        print("Done! saving model as " + save_path + " ...")

        dfpred = pd.DataFrame(pred_list)
        dfpred.to_csv(
            '/home/pramuk/IISC/sn-mnn-auv-nav/predictions.csv', index=False)

        filename = open(save_path, "wb")
        pickle.dump(mnn, filename)
        filename.close()

        plt.figure(0)
        plt.plot(error_list[1:])
        plt.xlabel("Epochs")
        plt.ylabel("Squared Loss")
        plt.title("Squared Loss vs Epochs")
        plt.grid(True)

        plt.figure(1)
        plt.plot(rmse_list[1:])
        plt.xlabel("Epochs")
        plt.ylabel("Root Mean Squared Loss")
        plt.title("Root mean Squared Loss vs Epochs")
        plt.grid(True)

        plt.figure(2)
        plt.plot(dfpred[0])
        plt.plot(Z[:, 0])
        plt.legend(['predicted y', 'true y'])
        plt.xlabel("Time")
        plt.ylabel("Predcition and Desired speed(x component) (m/s)")

        plt.figure(3)
        plt.plot(dfpred[1])
        plt.plot(Z[:, 1])
        plt.title("Predcition and Desired output(y component) vs Time")
        plt.legend(['predicted y', 'true y'])
        plt.xlabel("Time")
        plt.ylabel("Predcition and Desired speed(x component) (m/s)")

        plt.figure(4)
        plt.plot(dfpred[2])
        plt.plot(Z[:, 2])
        plt.title("Predcition and Desired output (z component) vs Time")
        plt.legend(['predicted z', 'true z'])
        plt.xlabel("Time")
        plt.ylabel("Predcition and Desired speed(x component) (m/s)")

        # plt.savefig("trained_models/loss_" + str(neeta) + "_" + str(neeta_dash) + "_" + str(epochs) + ".eps", format="eps", dpi=1000)
        plt.show()
    else:
        print("Network Unstable! Quitting...")