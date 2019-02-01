from linear1 import *
import numpy as np
import json
# import pandas as pd
# solve crashing issue https://github.com/MTG/sms-tools/issues/36
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

x = np.array([.86, .09, -.85, .87, -.44, -.43, -1.1, .40, -.96, .17])
y = np.array([2.49, .83, -.25, 3.1, .87, .02, -.12, 1.81, -.83, .43])

plt.plot(x, y, 'o')

lr = [Decay(),LearningRate(),Momentum()]
for lr in lr:
    print(lr)
    regressor = LinearRegressionGD(lr)
    regressor.fit(x, y)
    print(regressor.w)
