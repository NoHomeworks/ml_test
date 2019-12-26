import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import fully_connected
from sklearn.datasets import make_blobs

data01,data02,data03 = make_blobs(n_samples=100,n_features=3,centers=2,random_state=101)

# plt.scatter(data)
# plt.show()