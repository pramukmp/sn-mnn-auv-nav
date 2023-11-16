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
neeta=1e-3
neeta_dash=5e-4
lipschitz_constant = 1.2
epochs = 70
pred_list=[]

mnn = MemoryNeuralNetwork(10, 50, 3, neeta=neeta, neeta_dash=neeta_dash, lipschitz_norm=lipschitz_constant, spectral_norm=True) 
# Inputs - 4 beams, 3 accelero, 3 gyro, 
# Hidden neurons - 100

################# MAIN ################

# load
path = os.getcwd()
path = os.path.abspath(os.path.join(path, os.pardir))
IMU_in = load('/home/airl/auvnav/sn-mnn-auv-nav/dataset/TrainAndValidation/IMU_in.npy')
V = load('/home/airl/auvnav/sn-mnn-auv-nav/dataset/TrainAndValidation/V.npy')

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
beams = np.zeros((len(V[0, :]), 4))
for i in range(0, len(V[0, :])):
    beams[i, :] = np.matmul(A, (V[:, i]) * (1 + 0.007))  # scale factor 0.7%

beams_noise = beams + (0.042 ** 2) * np.random.randn(len(V[0, :]), 4) + \
              0.001 * np.ones((len(V[0, :]), 4))

T = 100 # Sampling rate IMU - 100Hz, DVL - 1Hz
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
    Y[n, :] = y # DVL
    z = V[n, :] # velocities target variable
    Z[n, :] = z
    n = n + 1
#print(Y)
# Num of DVL samples
N = len(IMU_in[0, :, 0]) // T
print(X_acc.shape,X_gyro.shape,Y.shape)

input_data=np.concatenate((X_acc,X_gyro,Y),axis=1)
input_data_df=pd.DataFrame(input_data)
print(len(input_data))

error_sum = 0.0
error_list = np.zeros(epochs)
rmse_list=np.zeros(epochs)
unstable_flag = False
rmse_sum=0.0

try:
    for _ in range(0, 2000):
        mnn.feedforward(np.zeros(10))
        mnn.backprop(np.zeros(3))
    for epoch in range(0, epochs):
        pred_list=[]
        if(epoch != 0):
            print("Training for epoch %2d finished with average loss of %.5f\n" % (epoch-1, error_sum/len(input_data)))
            error_list[epoch] = error_sum / len(input_data)
            rmse_list[epoch]=rmse_sum/len(input_data)
        error_sum = 0
        rmse_sum=0
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        for i in range(1, len(input_data)):
            
            if(mnn.squared_error > 1e30):
                
                unstable_flag = True
                #double break
                i = sys.maxint
                epoch = sys.maxint
            
            pred=mnn.feedforward(input_data[i-1,:])
            mnn.backprop(Z[i-1,:])
            # print(pred,Z[i-1,:])
            if(epoch>45):
                pred_list.append(pred)
            
            error_sum += mnn.squared_error
            rmse_sum+=mnn.rmse
            
            print("Training for epoch %2d, progress %5.2f%% with squared loss: %.5f with rmse: %.5f" % (epoch, (i/len(input_data)) * 100, mnn.squared_error,mnn.rmse), end="\r")
        #print()
        

    print("=====================================================================================================")    
    
except Exception as e:
    #print(e.__doc__)
    print(e)    

finally:
    save_path='/home/airl/auvnav/sn-mnn-auv-nav/trained_models_mnn/sn-mnn.obj'
    if not unstable_flag:
        print("Done! saving model as " + save_path + " ...")

        dfpred=pd.DataFrame(pred_list)
        dfpred.to_csv('predictions.csv', index=False)

        filename = open(save_path, "wb")
        pickle.dump(mnn, filename)
        filename.close()

        plt.figure(0)
        plt.plot(error_list[1:])
        plt.xlabel("Epochs");plt.ylabel("Squared Loss");plt.title("Squared Loss vs Epochs")
        plt.grid(True)

        plt.figure(1)
        plt.plot(rmse_list[1:])
        plt.xlabel("Epochs");plt.ylabel("Root Mean Squared Loss");plt.title("Root mean Squared Loss vs Epochs")
        plt.grid(True)

        plt.figure(2)
        plt.plot(pred_list[:,0],'g',Z[:,0],'r')
        plt.title("Predcition and Desired output(x component) vs Time")
        plt.legend(['Predcited output','Desired output'])

        plt.figure(3)
        plt.plot(pred_list[:,1],'g',Z[:,1],'r')
        plt.title("Predcition and Desired output(y component) vs Time")
        plt.legend(['Predcited output','Desired output'])

        plt.figure(4)
        plt.plot(pred_list[:,2],'g',Z[:,2],'r')
        plt.title("Predcition and Desired output (z component) vs Time")
        plt.legend(['Predcited output','Desired output'])

        #plt.savefig("trained_models/loss_" + str(neeta) + "_" + str(neeta_dash) + "_" + str(epochs) + ".eps", format="eps", dpi=1000)
        plt.show()
    else:
        print("Network Unstable! Quitting...")






