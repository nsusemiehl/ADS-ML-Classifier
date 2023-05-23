import numpy as np

x = np.arange(10)
print(x)
np.save("x.npy", x)
y = np.load("x.npy")
print(y)