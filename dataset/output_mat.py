import numpy as np
from scipy.io import savemat

# Load the .npy file
data = np.load('IMU_in.npy')
data1 = np.load('V.npy')

# Save it as a .mat file
savemat('IMU.mat', {'data': data})
savemat('V.mat', {'data': data})

# Load the .npy file
data = np.load('IMU_in.npy')
data1 = np.load('V.npy')

# Save it as a .mat file
savemat('IMU.mat', {'data': data})
savemat('V.mat', {'data': data})