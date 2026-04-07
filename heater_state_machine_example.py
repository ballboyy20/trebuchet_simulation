import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ==========================================
# 1. The Continuous Model (ODE)
# ==========================================
def temperature_model(t, y, heater_on):
    T = y[0]
    T_ext = 10.0 # Outside temperature
    k = 0.1      # Cooling rate
    Q = 2.0      # Heater power
    
    # dT/dt = cooling + heating
    dTdt = -k * (T - T_ext) + (Q if heater_on else 0)
    return [dTdt]


# ==========================================
# 2. The Discrete Events (Triggers)
# ==========================================
def hit_lower_bound(t, y, heater_on):
    return y[0] - 18.0
hit_lower_bound.terminal = True
hit_lower_bound.direction = -1 # Only trigger crossing downward

def hit_upper_bound(t, y, heater_on):
    return y[0] - 22.0
hit_upper_bound.terminal = True
hit_upper_bound.direction = 1  # Only trigger crossing upward


# ==========================================
# 3. The Hybrid Simulation Loop
# ==========================================
def run_hybrid_simulation():
    t_start = 0.0
    t_end = 100.0
    
    # Initial conditions
    t_current = t_start
    y_current = [15.0]  # Starts at 15 degrees
    heater_state = True # State machine: Start with heater ON
    
    # Arrays to store the combined results
    t_results = []
    y_results = []
    heater_results = [] # Tracking this so we can plot it!
    
    while t_current < t_end:
        # 1. Decide which event to look for based on current state
        if heater_state:
            active_events = [hit_upper_bound]
        else:
            active_events = [hit_lower_bound]
            
        # 2. Run the solver
        sol = solve_ivp(
            fun=temperature_model,
            t_span=(t_current, t_end),
            y0=y_current,
            events=active_events,
            args=(heater_state,), 
            max_step=0.5 # Keeps the resolution smooth for plotting
        )
        
        # 3. Store results
        t_results.append(sol.t)
        y_results.append(sol.y[0])
        heater_results.append(np.full_like(sol.t, heater_state, dtype=float))
        
        # 4. Update for next iteration
        t_current = sol.t[-1]
        y_current = sol.y[:, -1]
        
        # 5. Execute state machine logic if an event stopped the solver
        if sol.status == 1: # A termination event occurred
            if heater_state:
                print(f"Time {t_current:05.1f}s: Hit 22°C. Turning heater OFF.")
                heater_state = False
            else:
                print(f"Time {t_current:05.1f}s: Hit 18°C. Turning heater ON.")
                heater_state = True
        elif sol.status == 0:
            print("Simulation reached t_end.")
            break
            
    # Concatenate the lists of arrays into single flat arrays for plotting
    return np.concatenate(t_results), np.concatenate(y_results), np.concatenate(heater_results)


# ==========================================
# 4. Execution and Plotting
# ==========================================
if __name__ == "__main__":
    # Run the simulation
    t_sim, y_sim, h_sim = run_hybrid_simulation()

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Temperature on the left Y-axis
    color = 'tab:red'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Temperature (°C)', color=color)
    ax1.plot(t_sim, y_sim, color=color, linewidth=2, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add target threshold lines
    ax1.axhline(22, linestyle='--', color='gray', alpha=0.5)
    ax1.axhline(18, linestyle='--', color='gray', alpha=0.5)

    # Plot Heater State on the right Y-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Heater State (1=ON, 0=OFF)', color=color)  
    ax2.plot(t_sim, h_sim, color=color, linestyle=':', linewidth=2, label='Heater State')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0, 1])

    fig.tight_layout()  
    plt.title('Hybrid Simulation: Thermostat with Hysteresis')
    plt.grid(True, alpha=0.3)
    plt.show()
    