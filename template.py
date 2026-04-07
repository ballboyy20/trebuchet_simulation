#%%
# Home work 4 problem 4
import sympy as sp
from sympy.physics.vector import dynamicsymbols, vlatex
from IPython.display import Math, display
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



#------------------------------------------------
# Function called by solve_ivp()
#------------------------------------------------

def func(t, y, mass1, mass2, k, c, force, g, lambda1s, ts):
    
    # Extract states (generalized coordinates and generalized
    # velocities) from the state vector y
    y_position = y[0]
    theta      = y[1]
    R          = y[2]   # will stay ~0.75 due to constraint
    y_dot      = y[3]
    theta_dot  = y[4]
    R_dot      = y[5]   # will stay ~0 due to constraint


    # M Matrix
    M = np.array([[(mass1+mass2), -mass2*R*np.sin(theta), mass2*np.cos(theta)],
                  [-R*np.sin(theta), mass2*R**2, 0],
                  [mass2*np.cos(theta), 0, mass2]])

    # F Vector
    F = np.array([
    [2*mass2*R_dot*theta_dot + mass2*R*theta_dot**2*np.cos(theta) - k*y_position + (mass1+mass2)*g + force + c*y_dot],
    [-2*mass2*R*R_dot*theta_dot - mass2*g*R*np.sin(theta)],
    [mass2*R*theta_dot**2 + mass2*g*np.cos(theta)]])

    # Constraint matrix
    a = np.array([[0, 0, 1]])   # shape (1,3)

    # Constraint derivative matrix
    dadt = np.array([[0,0,0]])   # shape (1,3)

    # Vector of generalized velocities
    qd = np.array([[y_dot],[theta_dot],[R_dot]])

    # Array of zeros for use in matrix equation
    zeros = np.zeros((1,1))

    # Calculate phi
    LeftHandSide = np.block([[M,-a.T],[-a, zeros]])
    RightHandSide2 = dadt@qd
    RightHandSide = np.block([[F],[RightHandSide2]])
    phi = np.linalg.solve(LeftHandSide, RightHandSide)
 
    # Extract second derivative values from phi to return
    q1dd = phi[0,0] # ÿ
    q2dd = phi[1,0] # θ̈
    q3dd  = phi[2, 0]  # R̈
    lamb1 = phi[3, 0]  # λ 

    # Extract lambda1
    lamb1 = phi[3,0]

    # Store lamb1 at each time step to a growing list
    lambda1s.append(lamb1)

    # Store the current time to a growing list
    ts.append(t)

    return np.array([y_dot, theta_dot, R_dot, q1dd, q2dd, q3dd])

#------------------------------------------------
# Setup
#------------------------------------------------

# Parameters
mass1 = 2 # kg
mass2 = 1 # kg
R = 0.75 # m
k = 1200 # N/m
c = 0
force = 0
g = 9.81

# Initial conditions
theta0 = np.deg2rad(30)
y_0 = 0.25 # m
y_dot_0 = 0
theta_dot0 = 0
R_dot = 0
y0 = [y_0,theta0, R, y_dot_0, theta_dot0, R_dot]

# Lists to store results within function
lambda1s = []
ts = []

#------------------------------------------------
# Solve
#------------------------------------------------

# Time vector
t_span = (0, 3)
t_eval = np.linspace(0, 3, 500)

# Call solve_ivp()
sol = solve_ivp(func, t_span, y0, t_eval=t_eval, args=(mass1, mass2, k, c, force, g, lambda1s,ts), rtol=1e-6, atol=1e-9)

# After sol = solve_ivp(...)
Y_pos     = sol.y[0]   # y position
Theta     = sol.y[1]   # theta
R_history = sol.y[2]   # R (should stay ~0.75)

#------------------------------------------------
# Plot Results
#------------------------------------------------

# Motion vs. Time
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t_eval,Y_pos)
plt.ylabel('y (m)')
plt.grid(True)
plt.tight_layout()
plt.subplot(2,1,2)
plt.plot(t_eval,Theta)
plt.ylabel('θ (rad)')
plt.xlabel('t (s)')
plt.grid(True)
plt.tight_layout()
plt.show()


#------------------------------------------------
# Constraint forces
#------------------------------------------------

# Get unique times and their indices (in case there are repeats)
ts_raw, indices = np.unique(ts, return_index=True)

# Get values of lambda1 at those unique time values
lambda1s_raw = np.array(lambda1s)[indices]

# Interpolate to get lambda1 values only at t_eval times
lambda1_interp = np.interp(t_eval, ts_raw, lambda1s_raw)

# Calculate generalized constraint forces
C = lambda1_interp

# Plot R vs time
plt.figure(3)
plt.plot(t_eval, R_history)
plt.ylabel('R (m)')
plt.xlabel('t (s)')
plt.title('Pendulum Length R vs Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# Constraint force plot (your existing figure 4, no changes needed)
plt.figure(4)
plt.plot(t_eval, C)

plt.xlabel('t (s)')
plt.ylabel('P (N)')
plt.title('Constraint Force vs Time')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# %%