from scipy.integrate import solve_ivp
import numpy as np

#==============
# 1. The Model
# ======================

def func(time, y, hard_coded_values, mp, mw, dw, N_current, time_step, lambda1s, T_values_internal, state):

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

    N = N_current

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
            [(d_R*np.sin(theta)*theta_dot) + x_dot, -d_R*np.cos(theta) * theta_dot, d_R*( (x_pos*np.cos(theta)*theta_dot) + (y_pos*theta_dot*np.sin(theta)) + (np.sin(theta)* x_dot) - (np.cos(theta)*y_dot)),0]])

    
    # constraint matrices FOR THE SECOND STAGE
    if state == "swinging":
        a = np.array([[0,1,0,0],
                    [x_pos - (d_R*np.cos(theta)), y_pos - (d_R*np.sin(theta)), d_R*((x_pos*np.sin(theta))-(y_pos*np.cos(theta))),0]])
        
        da_dt = np.array([[0,0,0,0],
            [(d_R*np.sin(theta)*theta_dot) + x_dot, -d_R*np.cos(theta) * theta_dot, d_R*( (x_pos*np.cos(theta)*theta_dot) + (y_pos*theta_dot*np.sin(theta)) + (np.sin(theta)* x_dot) - (np.cos(theta)*y_dot)),0]])

    # constraint matrices FOR THE THIRD STAGE
    if state == "flying":
            q_dd = np.linalg.solve(M, F)
            x_dd = q_dd[0,0]
            y_dd = q_dd[1,0]
            theta_dd = q_dd[2,0]
            phi_dd = q_dd[3,0]

            lambda1s.append(0)
            T_values_internal.append(0)
            time_step.append(time)
            return np.array([x_dot, y_dot, theta_dot, phi_dot, x_dd, y_dd, theta_dd, phi_dd])
    
    
    qd = np.array([[x_dot], [y_dot], [theta_dot], [phi_dot]])
    zeros = np.zeros((a.shape[0], a.shape[0]))

    #calculate PHI
    LeftHandSide = np.block([[M,-a.T],[-a, zeros]])
    RightHandSide2 = da_dt@qd
    RightHandSide = np.block([[F],[RightHandSide2]])
    phi = np.linalg.solve(LeftHandSide, RightHandSide)

    # Extract second derivative values from phi to return
    x_dd = phi[0,0] # ẍ
    y_dd = phi[1,0] # ÿ
    theta_dd  = phi[2, 0]  # θ̈
    phi_dd = phi[3, 0]  # ϕ̈
    lambda1 = phi[4, 0]  # λ1
    lambda2 = phi[5, 0]  # λ2

    # Store lambda1 at each time step to a growing list
    lambda1s.append(lambda1)

    # calculate T 
    T = np.sqrt((lambda2 * a[1,0])**2 + (lambda2 * a[1,2])**2 + (lambda2 * a[1,3])**2)
    T_values_internal.append(T)

    # Store the current time to a growing list
    time_step.append(time)

    return np.array([x_dot, y_dot, theta_dot, phi_dot, x_dd, y_dd, theta_dd, phi_dd])

# ==============
# helper functions: swinging to flying, and calculating current Normal force
# ==============
def helper_slope_ratio(x_arm, y_arm, x_projectile, y_projectile):
     k_arm = (y_arm - 0)/(x_arm - 0)
     k_rope = (y_projectile - y_arm)/(x_projectile - x_arm)

     return (abs(k_arm-k_rope) / abs(k_arm))

def calculate_current_lambda1(y, hard_coded_values, mp, mw, dw, N_current):
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
    N = N_current

    qd = np.array([[x_dot], [y_dot], [theta_dot], [phi_dot]])
    

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
     
     
    a = np.array([[0,1,0,0],
                    [x_pos - (d_R*np.cos(theta)), y_pos - (d_R*np.sin(theta)), d_R*((x_pos*np.sin(theta))-(y_pos*np.cos(theta))),0]])
        
    da_dt = np.array([[0,0,0,0],
            [(d_R*np.sin(theta)*theta_dot) + x_dot, -d_R*np.cos(theta) * theta_dot, d_R*( (x_pos*np.cos(theta)*theta_dot) + (y_pos*theta_dot*np.sin(theta)) + (np.sin(theta)* x_dot) - (np.cos(theta)*y_dot)),0]])
    
    zeros = np.zeros((a.shape[0], a.shape[0]))

    # calculate PHI
    LeftHandSide = np.block([[M,-a.T],[-a, zeros]])
    RightHandSide2 = da_dt@qd
    RightHandSide = np.block([[F],[RightHandSide2]])
    phi = np.linalg.solve(LeftHandSide, RightHandSide)
    return phi[4, 0]  # λ1

# ==============
# The event triggers
# =============

def sliding_to_swinging(time, y, hard_coded_values, mp, mw, dw, N_current, time_step, lambda1s, T_values_internal, state):
    target_N = 0
    lambda1 = calculate_current_lambda1(y, hard_coded_values, mp, mw, dw, N_current)
    N = lambda1 # this is the normal force at the moment of transition. I want to trigger when this hits zero, which means the arm is no longer pushing on the projectile and it starts swinging freely.
    return N - target_N
sliding_to_swinging.terminal = True

def swinging_to_flying(time, y, hard_coded_values, mp, mw, dw, N_current, time_step, lambda1s, T_values_internal, state):
    d_R = hard_coded_values["d_R"]
    theta = y[2]
    x_arm = y[0] - d_R*np.cos(theta)
    y_arm = y[1] - d_R*np.sin(theta)
    x_projectile = y[0]
    y_projectile = y[1]

    return helper_slope_ratio(x_arm, y_arm, x_projectile, y_projectile) - 0.7 # this is the threshold. Does it need to be less than? or is equal too okay???
swinging_to_flying.terminal = True
swinging_to_flying.direction = -1 # only trigger when crossing downward, when the slope ratio drops below the threshold



def jake_sutton(x0, y0, theta0, phi0, time, mp, mw, dw):

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


    lambda1s = []
    T_values_internal = []
    time_step = []

    x = []
    y = []
    theta = []
    phi = []

    raw_solutions = []

    x_dot = 0
    y_dot = 0
    theta_dot = 0
    phi_dot = 0

    N_current = 0
    

    y0 = [x0, y0, theta0, phi0, x_dot, y_dot, theta_dot, phi_dot]
    state = "sliding"


    while t_start < t_end:
        if state == "sliding":
            active_events = [sliding_to_swinging]
        elif state == "swinging":
            active_events = [swinging_to_flying]
        else:
            active_events = []

        solution = solve_ivp(func, 
                                (t_start, t_end), 
                                y0,
                                events=active_events,
                                args=(hard_coded_values, mp, mw, dw, N_current, time_step, lambda1s, T_values_internal, state),
                                rtol=1e-12, atol=1e-15)

        # x.append(solution.y[0])
        # y.append(solution.y[1])
        # theta.append(solution.y[2])
        # phi.append(solution.y[3])
        raw_solutions.append(solution)
        

        if solution.status == 1: # an event was triggered
            if state == "sliding":
                print(f"Transition from sliding to swinging at time {solution.t_events[0][0]:.4f} seconds.")
                state = "swinging"
            elif state == "swinging":
                print(f"Transition from swinging to flying at time {solution.t_events[0][0]:.4f} seconds.")
                state = "flying"
            else:
                print("Simulation reached t_end.")
                break

            # update initial conditions for the next phase
            N_current = lambda1s[-1]
            t_start = solution.t[-1]
            y0 = solution.y[:, -1]
            current_lambda1 = calculate_current_lambda1(y0, hard_coded_values, mp, mw, dw, N_current)

        elif solution.status == 0: # this should kill the loop...
            print("Simulation donzo")
            break
            
        
    solutions_full = np.concatenate([s.y for s in raw_solutions], axis=1)
    time_full = np.concatenate([s.t for s in raw_solutions])

    x = np.interp(time, time_full, solutions_full[0])
    y = np.interp(time, time_full, solutions_full[1])
    theta = np.interp(time, time_full, solutions_full[2])
    phi = np.interp(time, time_full, solutions_full[3])

    ts_raw, indices = np.unique(time_step, return_index=True)
    lambda1s_raw = np.array(lambda1s)[indices]
    T_values_raw = np.array(T_values_internal)[indices]

    N = np.interp(time,ts_raw,lambda1s_raw)
    T = np.interp(time,ts_raw, T_values_raw)


    return x, y, theta, phi, N, T