################## Imports ##################

import numpy as np
import pandas as pd
import os
import sys
import pickle
# from MemoryNetwork import MemoryNeuralNetwork
from mnn_torch import MemoryNeuralNetwork
import matplotlib.pyplot as plt
import torch
# Seeds

np.random.seed(0)

################## DNN ##################
neeta=1e-3
neeta_dash=5e-4
lipschitz_constant = 1.2
epochs = 70
pred_list = []

mnn = MemoryNeuralNetwork(10, 50, 3, neeta=neeta, neeta_dash=neeta_dash,
                          lipschitz_norm=lipschitz_constant, spectral_norm=True)

################# MAIN ################

# load
inputcsvpath = '/home/pramuk/IISC/sn-mnn-auv-nav/inputs/input_data.csv'
targetcsvpath = '/home/pramuk/IISC/sn-mnn-auv-nav/inputs/target_var.csv'

if os.path.isfile(inputcsvpath) and os.path.isfile(targetcsvpath):
    input_data_df = pd.read_csv(inputcsvpath, index_col=0)
    input_data = input_data_df.to_numpy()
    target_var_df = pd.read_csv(targetcsvpath, index_col=0)
    Z = target_var_df.to_numpy()

else:
    print('File not present')
    exit()

error_sum = 0.0
error_list = np.zeros(epochs)
rmse_list = np.zeros(epochs)
unstable_flag = False
rmse_sum = 0.0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
                epoch, (i/len(input_data)) * 100, mnn.squared_error, mnn.rmse, mnn.rmse_x, mnn.rmse_y, mnn.rmse_z), end="\r")
        print(pred_list)

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
