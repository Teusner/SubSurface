import numpy as np
import matplotlib.pyplot as plt


g = 9.81
rho = 1000
l = 1  # length of the cylinder
r = 0.055  # radius of the cylinder
m = 6

# Fin parameters
Cd = 0.8
S = 0.1

# Drag parameters
fd = 100
fa = 1.5


def v(z):
    if z <= r:
        return 0
    elif -r < z < r:
        return l * (r**2 * np.arccos(z/r) - z * np.sqrt(r**2 - z**2))
    else:
        return np.pi * r**2 * l

def f(x, u=0):
    """System dynamics."""
    dxdt = np.zeros(4)
    dxdt[0] = x[2]
    dxdt[1] = x[3]
    dxdt[2] = (u - fa * x[2]**2) / m
    dxdt[3] = - rho * v(x[1]) * g / m + g - fd * x[3] / m + (0.5 * rho * Cd * S * x[2]**2) / m
    return dxdt

def u_model(t):
    if t < 6:
        return 5
    else:
        return 2*fa*g/rho/S/Cd*(rho * np.pi * r**2 * l - m)
    
def u_smc(X, zd, dzd=0):
    de = dzd - X[-1][3]
    e = zd - X[-1][1]
    s = de + e
    return 30 * np.tanh(s)


if __name__=="__main__":
    X = [np.array([0, 0, 0, 0])]
    zd = []
    u = []

    h = 0.05
    time = np.arange(0, 150, h)
    for t in time:
        zd.append(10)
        u.append(u_smc(X, 10))
        # u.append(u_model(t))
        X.append(X[-1] + h * f(X[-1], u[-1]))

    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(time, np.array(zd), color="crimson", label='target')
    ax[0].plot(time, np.array(X)[:-1, 1], label='z Position')
    ax[0].legend()
    ax[0].set_ylabel('Z Position (m)')
    ax[0].set_xlabel('Time (s)')

    ax[1].plot(time, np.asarray(u), color="purple", label='u')
    ax[1].set_ylabel('Control Input (N)')
    ax[1].set_xlabel('Time (s)')

    print(X[-1])

    plt.savefig("smc_control_setpoint.pdf", bbox_inches='tight')
    plt.show()