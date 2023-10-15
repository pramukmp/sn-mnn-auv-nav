import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dvl=pd.read_excel('/home/pramuk/IISC/AUV- Snapir/Tedelyne DVL/teledyne_navigator_measurements.xlsx')
dvl.info()
x=dvl['latitude']
y=dvl['longitude']
plt.plot(x,y)