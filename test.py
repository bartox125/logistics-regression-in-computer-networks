import numpy as np
from log_reg import train

X = np.array([
    [0.5, -1.2, 0.3],
    [2.1,  1.5, -0.4],
    [-0.8, 0.1,  1.7],
    [1.4, -0.6,  0.2]
], dtype=float)

print(train(X))

