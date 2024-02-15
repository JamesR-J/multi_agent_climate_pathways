"""
adapted from https://github.com/timkittel/ays-model/
"""
from __future__ import division, print_function

import numpy as np
import warnings as warn
import sys
import torch
import jax
import jax.numpy as jnp
import jax.random as jrandom


# # long name (command option line style) : short name (lower case)
# DEFAULT_NAME = "default"
# MANAGEMENTS = {
#     "degrowth": "dg",
#     "solar-radiation": "srm",
#     "energy-transformation": "et",
#     "carbon-capture-storage": "ccs",
# }
#
AYS_parameters = {}
AYS_parameters["A_offset"] = 600  # pre-industrial level corresponds to A=0
AYS_parameters["beta"] = 0.03  # 1/yr
AYS_parameters["beta_DG"] = AYS_parameters["beta"] / 2
AYS_parameters["epsilon"] = 147.  # USD/GJ
AYS_parameters["rho"] = 2.  # 1
AYS_parameters["phi"] = 47.e9  # GJ/GtC
AYS_parameters["phi_CCS"] = AYS_parameters["phi"] * 4 / 3 # 25% carbon taken away in form oc co2 from the system
AYS_parameters["sigma"] = 4.e12  # GJ
AYS_parameters["sigma_ET"] = AYS_parameters["sigma"] * .5 ** (1 / AYS_parameters["rho"])
AYS_parameters["tau_A"] = 50.  # yr
AYS_parameters["tau_S"] = 50.  # yr
AYS_parameters["theta"] = AYS_parameters["beta"] / (950 - AYS_parameters["A_offset"])  # 1/(yr GJ)
AYS_parameters["theta_SRM"] = 0.5 * AYS_parameters["theta"]

boundary_parameters = {}
boundary_parameters["A_PB"] = 945 - AYS_parameters["A_offset"]  # 450ppm
# boundary_parameters["A_PB_350"] = 735 - AYS_parameters["A_offset"]
boundary_parameters["W_SF"] = 4e13  # USD, year 2000 GWP

grid_parameters = {}

current_state = [240, 7e13, 5e11]

# rescaling parameters
grid_parameters["A_mid"] = current_state[0]
grid_parameters["W_mid"] = current_state[1]
grid_parameters["S_mid"] = current_state[2]

grid_parameters["n0"] = 40
grid_parameters["grid_type"] = "orthogonal"
border_epsilon = 1e-3

grid_parameters["boundaries"] = np.array([[0, 1],  # a: rescaled A
                [0, 1],  # w: resclaed W
                [0, 1]  # s: rescaled S
                ], dtype=float)

# use the full stuff in the S direction
grid_parameters["boundaries"][:2, 0] = grid_parameters["boundaries"][:2, 0] + border_epsilon
grid_parameters["boundaries"][:2, 1] = grid_parameters["boundaries"][:2, 1] - border_epsilon


def globalize_dictionary(dictionary, module="__main__"):
    if isinstance(module, str):
        module = sys.modules[module]

    for key, val in dictionary.items():
        if hasattr(module, key):
            warn.warn("overwriting global value / attribute '{}' of '{}'".format(key, module.__name__))
        setattr(module, key, val)
#
#
# # JH: maybe transform the whole to log variables since W,S can go to infinity...
# def _AYS_rhs(AYS, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
#     A, W, S = AYS
#     print(AYS)
#     U = W / epsilon
#     F = U / (1 + (S/sigma)**rho)
#     R = U - F
#     E = F / phi
#     Adot = E - A / tau_A
#     Wdot = (beta - theta * A) * W
#     Sdot = R - S / tau_S
#
#     print(Adot)
#     sys.exit()
#
#     return Adot, Wdot, Sdot
#
# #@jit(nopython=NB_USING_NOPYTHON)
# def AYS_rescaled_rhs(ays, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None):
#     a, y, s = ays
#     # print(ays)
#     # A, y, s = Ays
#
#     s_inv = 1 - s
#     s_inv_rho = s_inv ** rho
#     K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho)
#
#     a_inv = 1 - a
#     w_inv = 1 - y
#     Y = W_mid * y / w_inv
#     A = A_mid * a / a_inv
#     adot = K / (phi * epsilon * A_mid) * a_inv * a_inv * Y - a * a_inv / tau_A
#     ydot = y * w_inv * ( beta - theta * A )
#     sdot = (1 - K) * s_inv * s_inv * Y / (epsilon * S_mid) - s * s_inv / tau_S
#
#     return adot, ydot, sdot

def AYS_rescaled_rhs_marl2(ayse, t, args):
    """
    beta    = 0.03/0.015 = args[0]
    epsilon = 147        = args[1]
    phi     = 4.7e10     = args[2]
    rho     = 2.0        = args[3]
    sigma   = 4e12/sigma * 0.5 ** (1 / rho) = args[4]
    tau_A   = 50         = args[5]
    tau_S   = 50         = args[6]
    theta   = beta / (950 - A_offset) = args[7]
    # trade = args[8]
    """
    A_mid = 250  # TODO should these change with the different starting points though? idk yikes !!!I think so!!!
    Y_mid = 7e13
    S_mid = 5e11
    E_mid = 0  # TODO needs to add as an argument basically unless it is first step
    # TODO when first run it can calc the intro emissions that be used for E_mid
    # TODO how will this work with multi-agent though

    print(ayse)
    print(args)


    ays_inv_matrix = 1 - ayse
    # inv_s_rho = ays_inv_matrix.copy()
    inv_s_rho = ays_inv_matrix.at[:, 2].power(args[:, 3])[:, 2]
    # A_matrix = (A_mid * ays_matrix[:, 0] / ays_inv_matrix[:, 0]).view(2, 1)

    # Normalise
    A_matrix = A_mid * (ayse[:, 0] / ays_inv_matrix[:, 0])
    Y_matrix = Y_mid * (ayse[:, 1] / ays_inv_matrix[:, 1])
    G_matrix = inv_s_rho / (inv_s_rho + (S_mid * ayse[:, 2] / args[:, 4]) ** args[:, 3])
    E_matrix = G_matrix / args[:, 2] * Y_matrix
    # E_tot = jnp.tile((jnp.sum(E_matrix) / num_agents), (num_agents, 1))  # TODO check this as number seems tinY
    E_tot = jnp.sum(E_matrix)

    adot = (E_tot - (A_matrix / args[:, 5])) * (ays_inv_matrix[:, 0] * ays_inv_matrix[:, 0] / A_mid)  # TODO check this maths
    # adot = G_matrix / (phi * epsilon * A_mid) * (ays_inv_matrix[:, 0] * ays_inv_matrix[:, 0]).reshape(-1, 1) * Y_matrix - (ays_matrix[:, 0] * ays_inv_matrix[:, 0]).reshape(-1, 1) / tau_A  # TODO this won't work in current stage cus of combined A
    ydot = ayse[:, 1] * ays_inv_matrix[:, 1] * (args[:, 0] - args[:, 7] * A_matrix)
    sdot = (1 - G_matrix) * ays_inv_matrix[:, 2] * ays_inv_matrix[:, 2] * Y_matrix / (args[:, 1] * S_mid) - ayse[:, 2] * ays_inv_matrix[:, 2] / args[:, 6]

    # E_output = E_matrix / (E_matrix + E_mid)  # TODO sort out for later - maybe add a flag so it uses the made E_matrix as E_mid if flag True and then otherwise uses the fed in argument
    E_output = E_matrix

    return jnp.concatenate((adot[:, jnp.newaxis], ydot[:, jnp.newaxis], sdot[:, jnp.newaxis], E_output[:, jnp.newaxis]), axis=1)









