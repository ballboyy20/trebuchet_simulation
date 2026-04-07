# This script is to be used to test your function to simulate the motion
# of a trebuchet. You are welcome to change model parameters, initial
# conditions, etc., but your function MUST work with an unmodified version
# of this script in order to be considered fully correct. Pay special
# attention to initial conditions and units.

import numpy as np
import matplotlib.pyplot as plt

# Change this line to match your filename
from jake_sutton import sutton

# ---------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------

# Parameters that are sent to your function
mp = 30             # Projectile mass (kg)
mw = 3000           # Counterweight mass (kg)
dw = 1.5            # Distance to counterweight (m)

# Parameters used in determining initial conditions
dr = 8              # Distance to tip of arm (m)
l = 6               # Rope length (m)
h = 6               # Height of pivot (m)

# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

# Simulation time -- Your function must work with whatever time vector I
# send to it
t_span = (0,12)
t = np.arange(*t_span, 0.001)

# Initial conditions -- Note that specifying the initial angle of the arm
# uniquely defines the initial x
theta0 = -45
phi0 = 0
x0 = dr*np.cos(np.deg2rad(theta0)) - np.sqrt(l**2 - (dr*np.sin(np.deg2rad(theta0)) + h)**2)
y0 = -h

# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------

# Call your function -- Remember that your function and its file must be
# named with your last name, all lowercase. Everything you provide should
# be contained in a single file. There can be multiple functions within
# that file, but this driver function will only call the function called
# your_last_name().

x, y, theta, phi, N, T = sutton(x0, y0, theta0, phi0, t, mp, mw, dw)

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

# Motion vs. Time
fig, axs = plt.subplots(4,1)
axs[0].plot(t, x)
axs[0].set_ylabel("x (m)")
axs[1].plot(t, y)
axs[1].set_ylabel("y (m)")
axs[2].plot(t, theta)
axs[2].set_ylabel(r"$\theta$ (deg)")
axs[3].plot(t, phi)
axs[3].set_ylabel(r"$\phi$ (deg)")
axs[3].set_xlabel("t (s)")

# Constraint Forces vs. Time
fig, axs = plt.subplots(2,1)
axs[0].plot(t, N)
axs[0].set_ylabel("N (N)")
axs[1].plot(t, T)
axs[1].set_ylabel("T (N)")
axs[1].set_xlabel("t (s)")

# Projectile Motion (y vs. x)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_aspect("equal")

plt.show()