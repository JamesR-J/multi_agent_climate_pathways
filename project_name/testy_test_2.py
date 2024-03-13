import numpy as np
import matplotlib.pyplot as plt

def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


matrix_h = make_HiPPO(8)
print(matrix_h)
plt.imshow(matrix_h)
# for row in matrix_h:
#     plt.imshow(range(len(row)), row)
plt.show()
