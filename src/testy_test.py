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
print(result)

