import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


# matrix_h = make_HiPPO(8)
# print(matrix_h)
# plt.imshow(matrix_h)
# # for row in matrix_h:
# #     plt.imshow(range(len(row)), row)
# plt.show()


print(jnp.arange(11))
print(5 * 2 ** jnp.arange(11))

remaining_reward = 5 * 2 ** jnp.arange(11)
print(remaining_reward)
remaining_reward = remaining_reward.at[-4:].set(0)
print(remaining_reward)
