import operator

import torch
import sys
from datetime import datetime
import numpy as np

# print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))


def get_global_temperature(phi_t, temperature, b_t, f_2x, m_at, m_at_1750, exogenous_emissions):
    """Get the temperature levels."""
    return np.dot(phi_t, temperature) + np.dot(
        b_t, f_2x * np.log(m_at / m_at_1750) / np.log(2) + exogenous_emissions
    )


b_t = [0.1005, 0]
f_2x = 3.6813
m_at_1750 = 588
# above are constants that can't change

m_at = 1.0  # 851.0
exogenous_emissions = 10.0  # 0.5

result = np.dot(b_t, f_2x * np.log(m_at / m_at_1750) / np.log(2) + exogenous_emissions)  # this must be negative somehow
# print(result)

q_values = [0.16186343, 0.22504172, 0.2389216, 0.13161036]
max_ind = np.argmax(q_values)
q_dict = dict(enumerate(q_values))
q_dict.pop(max_ind)
# print(q_values)
# print(max_ind)
# print(q_dict)
# print(list(q_dict.keys())[list(q_dict.values()).index(max(list(q_dict.values())))])
# print(np.random.choice(list(q_dict.keys())))


# rewards = [tensor([[ 32.1177],
#         [582.5378]]), tensor([[11.7752],
#         [64.3992]]), tensor([[11.8183],
#         [67.9852]])]
rewards = [torch.tensor([32.1177, 582.5378]), torch.tensor([11.7752, 64.3992]), torch.tensor([11.8183, 67.9852])]
concatenated = torch.stack(rewards[-50:])
mean = torch.mean(concatenated, dim=0)

# print(rewards)
# print(concatenated)
# print(mean)

X_MID = [240, 7e13, 501.5198]
def compactification(x, x_mid):
    if x == 0:
        return 0.
    if x == np.infty:
        return 1.
    return x / (x + x_mid)

def inv_compactification(y, x_mid):
    if y == 0:
        return 0.
    if np.allclose(y, 1):
        return np.infty
    return x_mid * y / (1 - y)

print(compactification(7e13, X_MID[1]))
print(inv_compactification(0.35, X_MID[2]))



