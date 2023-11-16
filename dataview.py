import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dvl=pd.read_excel('/home/airl/auvnav/sn-mnn-auv-nav/AUV- Snapir/Tedelyne DVL/teledyne_navigator_measurements.xlsx')
dvl.info()
x=dvl['latitude']
y=dvl['longitute']
plt.plot(x,y)