from scipy.integrate import solve_ivp
import numpy as np


#==============
# 1. The Model
# ======================

def compute_accelerations_forces_N_T(time, y, hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state):

    # Establish hardcoded values
    m_A = hard_coded_values["m_A"]
    I_AO = hard_coded_values["I_AO"]
    I_WG = hard_coded_values["I_WG"]
    d_R = hard_coded_values["d_R"]
    d_A = hard_coded_values["d_A"]
    d_G = hard_coded_values["d_G"]
    h = hard_coded_values["h"]
    mu = hard_coded_values["mu"]
    g = hard_coded_values["g"]

    # Set initial conditions
    x_pos = y[0]
    y_pos = y[1]
    theta = y[2]
    phi = y[3]

    x_dot = y[4]
    y_dot = y[5]
    theta_dot = y[6]
    phi_dot = y[7]

    if time == 0 or len(normal_force_values) == 0:
         N = 0
    else:
         N = normal_force_values[-1]

    # M Matrix
    M = np.array([[(mp), 0 , 0, 0],
                    [0, (mp), 0 ,0],
                    [0 , 0, (I_AO+dw**2*mw), (d_G*dw*mw*np.sin(theta-phi))],
                    [0, 0, (d_G*dw*mw*np.sin(theta-phi)), (I_WG+d_G**2*mw)]])


    # F Vector
    F = np.array([[-N*mu],
                    [-g*mp],
                    [d_G*dw*mw*np.cos(theta-phi)*phi_dot**2 - g*np.cos(theta)*(d_A*m_A - dw*mw)],
                    [-d_G*mw*(dw*np.cos(theta-phi)*theta_dot**2 + g*np.sin(phi))]])

    
    #constraint matrices FOR THE FIRST STAGE
    if state == "sliding":
        a = np.array([[0,1,0,0],
                    [x_pos - (d_R*np.cos(theta)), y_pos - (d_R*np.sin(theta)), d_R*((x_pos*np.sin(theta))-(y_pos*np.cos(theta))),0]])
        
        da_dt = np.array([[0,0,0,0],
            [(d_R*np.sin(theta)*theta_dot) + x_dot, -d_R*np.cos(theta) * theta_dot + y_dot, d_R*( (x_pos*np.cos(theta)*theta_dot) + (y_pos*theta_dot*np.sin(theta)) + (np.sin(theta)* x_dot) - (np.cos(theta)*y_dot)),0]])

        qd = np.array([[x_dot], [y_dot], [theta_dot], [phi_dot]])
        zeros = np.zeros((a.shape[0], a.shape[0]))

        #calculate PHI
        LeftHandSide = np.block([[M,-a.T],[-a, zeros]])
        RightHandSide2 = da_dt@qd
        RightHandSide = np.block([[F],[RightHandSide2]])
        phi_augmented = np.linalg.solve(LeftHandSide, RightHandSide)

        # Extract second derivative values from phi to return
        x_dd = phi_augmented[0,0] # ẍ
        y_dd = phi_augmented[1,0] # ÿ
        theta_dd  = phi_augmented[2, 0]  # θ̈
        phi_dd = phi_augmented[3, 0]  # ϕ̈
        lambda1 = phi_augmented[4, 0]  # λ1
        lambda2 = phi_augmented[5, 0]  # λ2

        # Store lambda1 at each time step to a growing list
        normal_force = lambda1

        # calculate T 
        tension = np.sqrt((lambda2 * a[1,0])**2 + (lambda2 * a[1,1])**2)
        
    
    # constraint matrices FOR THE SECOND STAGE
    if state == "swinging":
        a = np.array([[x_pos - (d_R*np.cos(theta)), y_pos - (d_R*np.sin(theta)), d_R*((x_pos*np.sin(theta))-(y_pos*np.cos(theta))),0]])
        
        da_dt = np.array([[(d_R*np.sin(theta)*theta_dot) + x_dot, -d_R*np.cos(theta) * theta_dot + y_dot, d_R*( (x_pos*np.cos(theta)*theta_dot) + (y_pos*theta_dot*np.sin(theta)) + (np.sin(theta)* x_dot) - (np.cos(theta)*y_dot)),0]])

        qd = np.array([[x_dot], [y_dot], [theta_dot], [phi_dot]])
        zeros = np.zeros((a.shape[0], a.shape[0]))

        #calculate PHI
        LeftHandSide = np.block([[M,-a.T],[-a, zeros]])
        RightHandSide2 = da_dt@qd
        RightHandSide = np.block([[F],[RightHandSide2]])
        phi_augmented = np.linalg.solve(LeftHandSide, RightHandSide)

        # Extract second derivative values from phi to return
        x_dd = phi_augmented[0,0] # ẍ
        y_dd = phi_augmented[1,0] # ÿ
        theta_dd  = phi_augmented[2, 0]  # θ̈
        phi_dd = phi_augmented[3, 0]  # ϕ̈
        lambda2 = phi_augmented[4, 0]  # λ2

        normal_force = 0 # once swinging, the normal force should be zero because the arm is no longer pushing on the projectile

        # calculate T 
        tension = np.sqrt((lambda2 * a[0,0])**2 + (lambda2 * a[0,1])**2)

    # constraint matrices FOR THE THIRD STAGE
    if state == "flying":
            q_dd = np.linalg.solve(M, F)
            x_dd = q_dd[0,0]
            y_dd = q_dd[1,0]
            theta_dd = q_dd[2,0]
            phi_dd = q_dd[3,0]

            normal_force = 0
            tension = 0

    return x_dd, y_dd, theta_dd, phi_dd, normal_force, tension

def func(time, y, hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state):
    x_dot,y_dot, theta_dot, phi_dot = y[4], y[5], y[6], y[7]
    x_dd, y_dd, theta_dd, phi_dd, N, T = compute_accelerations_forces_N_T(time, y, hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state)

    normal_force_values.append(N)
    tension_values.append(T)
    time_step.append(time)

    return np.array([x_dot, y_dot, theta_dot, phi_dot, x_dd, y_dd, theta_dd, phi_dd])

# ==============
# The event triggers
# =============

def sliding_to_swinging(time, y, hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state):
    _, _, _, _, current_N, _ = compute_accelerations_forces_N_T(time, y, hard_coded_values, mp, mw, dw, [],[], [], state)
    # this is the normal force at the moment of transition. I want to trigger when this hits zero, which means the arm is no longer pushing on the projectile and it starts swinging freely.
    return current_N
sliding_to_swinging.terminal = True

def swinging_to_flying(time, y, hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state):
    d_R = hard_coded_values["d_R"]
    theta = y[2]
    x_tip = d_R*np.cos(theta)
    y_tip = d_R*np.sin(theta)
    x_projectile = y[0]
    y_projectile = y[1]

    if abs(x_tip) < 1e-10 or abs(x_projectile - x_tip) < 1e-10:
        return 1.0
    
    k_arm = (y_tip)/(x_tip)
    k_rope = (y_projectile - y_tip)/(x_projectile - x_tip)

    slope_ratio = (abs(k_arm-k_rope) / abs(k_arm))
    slope_ratio_threshold = 0.7

    return slope_ratio - slope_ratio_threshold # this is the threshold. Does it need to be less than? or is equal too okay???
swinging_to_flying.terminal = True
swinging_to_flying.direction = -1 # only trigger when crossing downward, when the slope ratio drops below the threshold



def sutton(x0, y0, theta0, phi0, time, mp, mw, dw):

    theta0 = np.deg2rad(theta0)
    phi0 = np.deg2rad(phi0)

    # Establish hardcoded values
    m_A = 150 # kg
    I_AO = 1466 # kg m ^2
    I_WG = 6750 #kg m^2
    l = 6 # meters
    d_R = 8 # meters
    d_A = 1.5 # meters
    d_G = 1.5 # meters
    h = 6 # meters
    mu = 2
    g = 9.81 # m/s^2

    hard_coded_values = {
        "m_A": m_A,
        "I_AO": I_AO,
        "I_WG": I_WG,
        "d_R": d_R,
        "d_A": d_A,
        "d_G": d_G,
        "h": h,
        "mu": mu,
        "g": g
    }

    t_start = time[0]
    t_end = time[-1]

    normal_force_values = []
    tension_values = []
    time_step = []

    x = []
    y = []
    theta = []
    phi = []

    x_dot = 0
    y_dot = 0
    theta_dot = 0
    phi_dot = 0

    y0 = [x0, y0, theta0, phi0, x_dot, y_dot, theta_dot, phi_dot]
    state = "sliding"
    solution_sliding = solve_ivp(func,
                            (t_start, t_end), 
                            y0,
                            events=[sliding_to_swinging],
                            args=(hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state),
                            rtol=1e-12, atol=1e-15)
    
    t_start = solution_sliding.t[-1]
    y0 = solution_sliding.y[:, -1]
    state = "swinging"
    solution_swinging = solve_ivp(func,
                            (t_start, t_end),
                            y0,
                            events=[swinging_to_flying],
                            args=(hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state),
                            rtol=1e-12, atol=1e-15)
    
    t_start = solution_swinging.t[-1]
    y0 = solution_swinging.y[:, -1]
    state = "flying"
    solution_flying = solve_ivp(func,
                            (t_start, t_end),
                            y0,
                            args=(hard_coded_values, mp, mw, dw, time_step, normal_force_values, tension_values, state),
                            rtol=1e-12, atol=1e-15)

    # concatenate all three phases
    solutions_full = np.concatenate([solution_sliding.y, solution_swinging.y, solution_flying.y], axis=1)
    time_full = np.concatenate([solution_sliding.t, solution_swinging.t, solution_flying.t])

    x     = np.interp(time, time_full, solutions_full[0])
    y     = np.interp(time, time_full, solutions_full[1])
    theta = np.interp(time, time_full, solutions_full[2])
    phi   = np.interp(time, time_full, solutions_full[3])

    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)

    ts_raw, indices = np.unique(time_step, return_index=True)
    normal_forces_raw = np.array(normal_force_values)[indices]
    tension_values_raw = np.array(tension_values)[indices]

    N = np.interp(time,ts_raw, normal_forces_raw)
    T = np.interp(time,ts_raw, tension_values_raw)

    

    ### RETURNS THETA AND PHI IN DEGREES NOT RADIANS
    return x, y, theta, phi, N, T