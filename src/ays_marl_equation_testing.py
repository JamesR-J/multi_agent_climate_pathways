from scipy.integrate import odeint


epsilon = 147
A = 0.50488132
Y = 0.52151895
S = 0.5
sigma = 4e12
rho = 2.0
phi = 4.7e10
tau_A = 50


U = Y / epsilon
F = U / (1 + (S/sigma)**rho)
E = F / phi
Adot = E - A / tau_A

# print(E)
# print(Adot)

a = 0.50488132
y = 0.52151895
s = 0.5
A_mid = 240
W_mid = 7e13
S_mid = 5e11

s_inv = 1 - s
s_inv_rho = s_inv ** rho
a_inv = 1 - a
w_inv = 1 - y
A = A_mid * a / a_inv
Y = W_mid * y / w_inv
K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho)

adot = K / (phi * epsilon * A_mid) * a_inv * a_inv * Y - a * a_inv / tau_A
adot_2 = (K / (phi * epsilon) * Y - A / tau_A) * (a_inv * a_inv / A_mid)
E = K / (phi * epsilon) * Y
adot_3 = (E - A / tau_A) * (a_inv * a_inv / A_mid)

# print(adot)
# print(adot_2)
# print(adot_3)

def AYS_rescaled_rhs_marl(ays, t=0, beta=None, epsilon=None, phi=None, rho=None, sigma=None, tau_A=None, tau_S=None, theta=None, E=None):
    a, y, s = ays
    print(a, y ,s)
    # A, y, s = Ays

    s_inv = 1 - s
    s_inv_rho = s_inv ** rho
    K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho)

    a_inv = 1 - a
    w_inv = 1 - y
    Y = W_mid * y / w_inv
    A = A_mid * a / a_inv
    E = K / (phi * epsilon) * Y  # TODO is it accurate to assume with pre calcs or shuld it be done better cus ays change within the odeint thingo
    # print(E)
    adot = (E - A / tau_A) * (a_inv * a_inv / A_mid)
    ydot = y * w_inv * ( beta - theta * A )
    sdot = (1 - K) * s_inv * s_inv * Y / (epsilon * S_mid) - s * s_inv / tau_S

    return adot, ydot, sdot



state = [0.5048813,  0.52151895,  0.5       ]

y = state[1]
s = state[2]
w_inv = 1 - y
s_inv = 1 - s
s_inv_rho = s_inv ** rho
Y = W_mid * y / w_inv
K = s_inv_rho / (s_inv_rho + (S_mid * s / sigma) ** rho)
E = K / (phi * epsilon) * Y
# print(E)

row = (3.0000e-02, 1.4700e+02, 4.7000e+10, 2.0000e+00, 4.0000e+12, 5.0000e+01,
        5.0000e+01, 8.5714e-05, E)

result = odeint(AYS_rescaled_rhs_marl, state, [0, 1], args=row, mxstep=50000)
print(result)

#### answer is [[0.50488132 0.52151895 0.5       ]
 # [0.5109028  0.52370622 0.49900241]]

# [0.50488132 0.52151895 0.5       ]  # from the sim normal
# -0.010097626447602127

# -0.010097626399924514  # from this normal


# [0.50488132 0.52151895 0.5       ]  # from the sim scaled
# 0.006106592420469616

# 0.006106592357041313  # from this scaled
