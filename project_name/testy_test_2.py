import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


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

#
# print(jnp.arange(11))
# print(5 * 2 ** jnp.arange(11))
#
# remaining_reward = 5 * 2 ** jnp.arange(11)
# print(remaining_reward)
# remaining_reward = remaining_reward.at[-4:].set(0)
# print(remaining_reward)
#
#
# # plt.rcParams['text.usetex'] = True
#
# plt.plot([0,1,2,3,4], [0,1,2,3,4])
# plt.xlabel("$\sigma 123^213$")
# plt.show()

x1 = 1
x2 = 8
x3 = 0.5

# first_term = np.sin((x1 + x2) / 2)
# second_term = np.sin((x1 - x2) / 2)
#
# print(first_term * second_term)
# print(first_term)
# print(second_term)
#
#
# print(f"This must be leq than 1 : {np.sin(x1 + x2)}")

# print(np.sin(x1) * np.sin(x2))
# print(np.sin(x2))

print(np.sin(x1) * np.sin(x2) * np.sin(x3))
print(np.sin(x1) * np.sin(x2))
