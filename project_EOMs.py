#%%
# Symbolic Derviation of EOMs for a trebuchet
# Jake Sutton
# Dynamics Project 2026

import sympy as sp
from sympy import Rational, trigsimp, simplify
import numpy as np
from sympy.physics.vector import dynamicsymbols, vlatex
from IPython.display import Math, display
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Initialize printing for Jupyter environments
sp.init_printing(use_latex='mathjax')

# 1. Define variables
# phi, theta, x, y = dynamicsymbols('phi theta x y')
phi = sp.symbols('ϕ', real=True, cls=dynamicsymbols)
phi_dot = sp.diff(phi, 't')
phi_dd = sp.diff(phi, 't', 2)
theta = sp.symbols('θ', real=True, cls=dynamicsymbols)
theta_dot = sp.diff(theta, 't')
theta_dd = sp.diff(theta, 't', 2)
x = sp.symbols('x', real=True, cls=dynamicsymbols)
x_dot = sp.diff(x, 't')
x_dd = sp.diff(x, 't', 2)
y = sp.symbols('y', real=True, cls=dynamicsymbols)
y_dot = sp.diff(y, 't')
y_dd = sp.diff(y, 't', 2)
d_W, d_A,d_G, d_R = sp.symbols('d_W d_A d_G d_R',real=True)
m_p, m_A, m_W, l, g, h, mu = sp.symbols('m_p m_A m_W l g h μ', real=True)
normal_force = sp.symbols('force_N', real=True)
I_AO, I_WG = sp.symbols('I_AO I_WG',real=True)


# Position vectors
r_weight = sp.Matrix([-d_W *sp.cos(theta) + d_G * sp.sin(phi), 
                 -d_W * sp.sin(theta) - d_G * sp.cos(phi), 
                 0])

r_Projectile = sp.Matrix([x, 
                 y, 
                 0])

# Velocity vectors
r_weight_dot = sp.diff(r_weight, 't')
r_Projectile_dot = sp.diff(r_Projectile, 't')

# Kinetic Energy
## Arm
T_arm = Rational(1, 2)*I_AO*theta_dot**2
## Counterweight
T_weight = Rational(1, 2)*I_WG*phi_dot**2 + Rational(1, 2)*m_W*(r_weight_dot.dot(r_weight_dot))
## Projectile
T_projectile = Rational(1, 2)*m_p*(r_Projectile_dot.dot(r_Projectile_dot))

# Potential Energy
## Arm
V_arm = m_A * g * (h + d_A*sp.sin(theta))
## Counterweight
V_weight = m_W * g * (h - r_weight[1]) # THIS MAY NEED A SIGN FLIPPED, MIGHT BE - R_WEIGHT[1]
## Projectile
V_projectile = m_p * g * (r_Projectile[1]) # MAKE SURE THIS IS CORRECT ALSO...

# Lagrangian
L = T_arm + T_weight + T_projectile - (V_arm + V_weight + V_projectile)

# Deritives for EOMs
## For x
dL_dx_dot = sp.diff(L, x_dot)
d_dt_dL_dx_dot = sp.diff(dL_dx_dot, 't')
dL_dx = sp.diff(L, x)
## For y
dL_dy_dot = sp.diff(L, y_dot)
d_dt_dL_dy_dot = sp.diff(dL_dy_dot, 't')
dL_dy = sp.diff(L, y)
## For theta
dL_dtheta_dot = sp.diff(L, theta_dot)
d_dt_dL_dtheta_dot = sp.diff(dL_dtheta_dot, 't')
dL_dtheta = sp.diff(L, theta)
## For phi
dL_dphi_dot = sp.diff(L, phi_dot)
d_dt_dL_dphi_dot = sp.diff(dL_dphi_dot, 't')
dL_dphi = sp.diff(L, phi)


# EOMs
EOM_x = sp.trigsimp(sp.Eq(d_dt_dL_dx_dot - dL_dx,-normal_force*mu))
EOM_y = sp.trigsimp(sp.Eq(d_dt_dL_dy_dot - dL_dy,0))
EOM_theta = sp.trigsimp(sp.Eq(d_dt_dL_dtheta_dot - dL_dtheta,0))
EOM_phi = sp.trigsimp(sp.Eq(d_dt_dL_dphi_dot - dL_dphi,0))

print("EOM for x:")
display(Math(vlatex(sp.simplify(EOM_x))))
print("EOM for y:")
display(Math(vlatex(sp.simplify(EOM_y))))
print("EOM for theta:")
display(Math(vlatex(sp.simplify(EOM_theta))))
print("EOM for phi:")
display(Math(vlatex(sp.simplify(EOM_phi))))


# Every term with a second derivative belongs in the [M] matrix, 
# and every term without a second derivative belongs in the [F] vector. 
# The EOMs can be rearranged to be in the form M*q_dd = F, 
# where q_dd is the vector of second derivatives (phi_dd, theta_dd, x_dd, y_dd).
# It goes x, y, theta, phi 
# for the order of the second derivatives in the [M] matrix
mass_matrix = sp.Matrix([[m_p, 0, 0, 0],
                         [0, m_p, 0, 0],
                         [0, 0, I_AO + d_W**2*m_W, d_G*d_W*m_W*sp.sin(theta-phi)],
                         [0, 0, d_G*d_W*m_W*sp.sin(theta-phi), I_WG+d_G**2*m_W]])


force_vector = sp.Matrix([-normal_force*mu,
                        -m_p * g,
                        (d_W*m_W-d_A*m_A)*g*sp.cos(theta) + d_G*d_W*m_W*sp.cos(theta-phi)*phi_dot**2,
                        d_G*m_W*(-d_W*sp.cos(theta-phi)*theta_dot**2 -g*sp.sin(phi))])

print("Mass Matrix [M]:     (Note: the order of the second derivatives is x, y, θ, ϕ)")
display(Math(vlatex(sp.simplify(mass_matrix))))
print("Force Vector {F}:")
display(Math(vlatex(sp.simplify(force_vector))))

### Auto solve for [M] and {F} from the EOMs
eom_equations = [EOM_x.lhs - EOM_x.rhs, 
                 EOM_y.lhs - EOM_y.rhs, 
                 EOM_theta.lhs - EOM_theta.rhs, 
                 EOM_phi.lhs - EOM_phi.rhs]


# This dictates the order of columns in [M] and rows in {F}
generalized_coordinates_dd = [x_dd, y_dd, theta_dd, phi_dd]


# It assumes the form: [M] * q_dd - {F} = 0
M_auto, F_auto = sp.linear_eq_to_matrix(eom_equations, generalized_coordinates_dd)

M_auto = sp.simplify(M_auto)
F_auto = sp.simplify(F_auto)

# print("Auto-generated Mass Matrix [M]:")
# display(Math(vlatex(M_auto)))

# print("Auto-generated Force Vector {F}: (should have g*cos(theta) factored out of two terms in the theta row)")
# display(Math(vlatex(sp.trigsimp(F_auto))))
#%%
print("Constraints:")

sliding_constraint_1 = sp.Eq((-l**2 + (x-((d_R)*sp.cos(theta)))**2 + (y-((d_R)*sp.sin(theta)))**2 ), 0)
sliding_constraint_2 = sp.Eq(y+h, 0)

sliding_constraint_1_velocity = sp.diff(sliding_constraint_1.lhs, 't')
sliding_constraint_2_velocity = sp.diff(sliding_constraint_2.lhs, 't')

print("Sliding constraints: \nthe distance from the projectile to the end of the arm must be equal to the length of the sling\nthe projectile must remain in contact with the ground")
display(Math(vlatex(sliding_constraint_1)))
display(Math(vlatex(sliding_constraint_2)))

print("Velocity forms of sliding constraints:")
display(Math(vlatex((sp.Eq(sliding_constraint_1_velocity, 0)))))
display(Math(vlatex(sp.simplify(sp.Eq(sliding_constraint_2_velocity, 0)))))

# make a list of the terms I want to isolate
generalized_coordinates_dot = [x_dot, y_dot, theta_dot, phi_dot]

# extract the terms from the velocity equations
a_terms, remainder_vel = sp.linear_eq_to_matrix(
    [sliding_constraint_1_velocity, sliding_constraint_2_velocity], 
    generalized_coordinates_dot
)

da_dt_terms = sp.diff(a_terms, 't')

print("Coefficients of the first derivatives [a]:")
display(Math(vlatex(sp.simplify(a_terms))))

print("Coefficients of the second derivatives [da/dt]:")
display(Math(vlatex(sp.simplify(da_dt_terms))))




#%%

