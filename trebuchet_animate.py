"""
trebuchet_animate.py
--------------------
Animates the trebuchet simulation output from jake_sutton().

Geometry matches the assignment diagram:
  - Pivot is at the world origin (0, 0)
  - theta: arm angle measured from +x axis (tip side)
  - phi:   counterweight pendulum angle (from vertical)
  - h:     height of pivot above ground (ground is at y = -h)
  - d_R:   distance from pivot to arm tip
  - d_W:   distance from pivot to arm butt (counterweight side)
  - d_G:   length of counterweight pendulum
  - l:     rope length from arm tip to projectile
  - Projectile (x, y) is in the pivot frame

Run: python trebuchet_animate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from jake_sutton import sutton

# ─────────────────────────────────────────────
# 1.  Simulation parameters (must match driver)
# ─────────────────────────────────────────────
mp  = 30
mw  = 3000
dw  = 1.5       # d_W: pivot-to-butt length
dr  = 8         # d_R: pivot-to-tip length
d_G = 1.5       # counterweight pendulum length
l   = 6         # rope length
h   = 6         # pivot height above ground

theta0 = -45    # degrees
phi0   = 0      # degrees
x0 = dr*np.cos(np.deg2rad(theta0)) \
     - np.sqrt(l**2 - (dr*np.sin(np.deg2rad(theta0)) + h)**2)
y0 = -h

t = np.arange(0, 12, 0.01)   # 1200 frames at 10 ms
x, y, theta_deg, phi_deg, N, T = sutton(x0, y0, theta0, phi0, t, mp, mw, dw)



theta_rad = np.deg2rad(theta_deg)
phi_rad   = np.deg2rad(phi_deg)


slide_end_i = next((i for i in range(1, len(t)) if N[i] <= 0.5), len(t) - 1)
swing_end_i = next((i for i in range(slide_end_i + 1, len(t)) if T[i] <= 0.5), len(t) - 1)
# Add this right after your simulation call
print(f"N max: {N.max():.2f}, N min: {N.min():.2f}")
print(f"T max: {T.max():.2f}, T min: {T.min():.2f}")
print(f"N is all zero: {np.all(N == 0)}")
print(f"T is all zero: {np.all(T == 0)}")
# Check rope length at phase transition points
theta_rad_full = np.deg2rad(theta_deg)

rope_length = np.sqrt((x - dr*np.cos(theta_rad_full))**2 + 
                      (y - dr*np.sin(theta_rad_full))**2)

print(f"Rope length at t=0:              {rope_length[0]:.4f} m (should be {l})")
print(f"Rope length at slide→swing:      {rope_length[slide_end_i]:.4f} m")
print(f"Rope length at swing→fly:        {rope_length[swing_end_i]:.4f} m")

plt.plot(t, rope_length)
plt.axhline(l, color='r', linestyle='--', label=f'l = {l} m')
plt.ylabel("Rope length (m)")
plt.xlabel("t (s)")
plt.legend()
plt.show()

# ─────────────────────────────────────────────
# 2.  Phase boundaries (for label colour)
# ─────────────────────────────────────────────

def get_phase(i):
    if i <= slide_end_i:
        return "● SLIDING",  "#ff9a3c"
    elif i <= swing_end_i:
        return "● SWINGING", "#4a9eff"
    else:
        return "● FLYING",   "#7ee8a2"

# ─────────────────────────────────────────────
# 3.  Figure / axes
#     World frame: pivot at (0,0), ground at y = -h
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#16213e")

# x range: projectile launches left (−x), so show −220 to +20
# y range: ground is at −h = −6, peak is ~150 m above pivot
ax.set_xlim(-220, 20)
ax.set_ylim(-h - 2, 160)
ax.set_aspect("equal")
ax.set_xlabel("x (m)", color="#e0e0e0", fontsize=11)
ax.set_ylabel("y (m)", color="#e0e0e0", fontsize=11)
ax.tick_params(colors="#888888")
for spine in ax.spines.values():
    spine.set_edgecolor("#333355")
ax.set_title("Trebuchet Simulation", color="#c9d1d9",
             fontsize=16, fontweight="bold", pad=12)

# Ground at y = -h
ground_y = -h
ax.axhline(ground_y, color="#4a9eff", linewidth=1.0, linestyle="--", alpha=0.35)
ax.fill_between([-220, 20], [ground_y - 2]*2, [ground_y]*2,
                color="#0d1117", alpha=0.9)
ax.fill_between([-220, 20], [ground_y]*2, [ground_y + 1.2]*2,
                color="#2d6a2d", alpha=0.6)

# ── Static structure ─────────────────────────
# Vertical support from ground to pivot
tower_w = 0.8
ax.add_patch(patches.Rectangle(
    (-tower_w / 2, ground_y), tower_w, h,
    linewidth=1, edgecolor="#8b7355", facecolor="#5c4a2a", zorder=3
))

# Pivot circle
ax.add_patch(plt.Circle((0, 0), 0.4, color="#e0a020", zorder=6))

# ─────────────────────────────────────────────
# 4.  Dynamic artists
# ─────────────────────────────────────────────

# Arm (tip side + butt side drawn as one line through pivot)
arm_line,  = ax.plot([], [], color="#c8a97e", linewidth=4.0,
                     solid_capstyle="round", zorder=4)

# Rope from arm tip to projectile
rope_line, = ax.plot([], [], color="#d4b896", linewidth=1.5, zorder=4)

# Counterweight pendulum rope (butt → CW centre)
cw_rope,   = ax.plot([], [], color="#8899aa", linewidth=1.5, zorder=4)

# Counterweight block
CW_W, CW_H = 2.0, 2.0
cw_patch = patches.Rectangle(
    (0, 0), CW_W, CW_H,
    linewidth=1.5, edgecolor="#4a9eff", facecolor="#1a3a5c", zorder=5
)
ax.add_patch(cw_patch)

# Projectile
proj_circle = plt.Circle((0, 0), 0.8, color="#ff6b35", zorder=7)
ax.add_patch(proj_circle)

# Projectile trail
TRAIL_LEN = 80
trail_x, trail_y = [], []
trail_line, = ax.plot([], [], color="#ff6b35", linewidth=1.0,
                      alpha=0.35, zorder=6)

# Phase + time labels
phase_text = ax.text(-215, 148, "", color="#ff9a3c",
                     fontsize=11, fontweight="bold", zorder=10)
time_text  = ax.text(-215, 140, "", color="#c9d1d9",
                     fontsize=10, zorder=10)

# ─────────────────────────────────────────────
# 5.  Geometry helpers
# ─────────────────────────────────────────────
def arm_tip(th):
    """Arm tip position (pivot frame)."""
    return dr * np.cos(th), dr * np.sin(th)

def arm_butt(th):
    """Arm butt position (pivot frame) — opposite side of pivot."""
    return -dw * np.cos(th), -dw * np.sin(th)

def cw_centre(butt_x, butt_y, ph):
    """
    Counterweight centre hangs from the butt on a pendulum of length d_G.
    phi is measured from the downward vertical (diagram convention).
    """
    cw_x = butt_x + d_G * np.sin(ph)
    cw_y = butt_y - d_G * np.cos(ph)
    return cw_x, cw_y

# ─────────────────────────────────────────────
# 6.  Init / update
# ─────────────────────────────────────────────
def init():
    arm_line.set_data([], [])
    rope_line.set_data([], [])
    cw_rope.set_data([], [])
    trail_line.set_data([], [])
    cw_patch.set_xy((-9999, -9999))
    proj_circle.center = (x[0], y[0])
    phase_text.set_text("")
    time_text.set_text("")
    return (arm_line, rope_line, cw_rope, cw_patch,
            trail_line, proj_circle, phase_text, time_text)


def update(i):
    th = theta_rad[i]
    ph = phi_rad[i]

    # Arm endpoints
    tx, ty = arm_tip(th)
    bx, by = arm_butt(th)

    arm_line.set_data([bx, 0, tx], [by, 0, ty])

    # Projectile position (simulation gives pivot-frame coords)
    px, py = x[i], y[i]

    # Rope (only while attached)
    if T[i] > 0.5:
        rope_line.set_data([tx, px], [ty, py])
    else:
        rope_line.set_data([], [])

    # Counterweight
    cx, cy = cw_centre(bx, by, ph)
    cw_rope.set_data([bx, cx], [by, cy])
    cw_patch.set_xy((cx - CW_W / 2, cy - CW_H / 2))

    # Projectile dot
    proj_circle.center = (px, py)

    # Trail
    trail_x.append(px)
    trail_y.append(py)
    if len(trail_x) > TRAIL_LEN:
        trail_x.pop(0)
        trail_y.pop(0)
    trail_line.set_data(trail_x, trail_y)

    # Labels
    label, colour = get_phase(i)
    phase_text.set_text(label)
    phase_text.set_color(colour)
    time_text.set_text(f"t = {t[i]:.2f} s")

    return (arm_line, rope_line, cw_rope, cw_patch,
            trail_line, proj_circle, phase_text, time_text)


# ─────────────────────────────────────────────
# 7.  Run
# ─────────────────────────────────────────────
SKIP   = 3          # animate every Nth frame (increase to go faster)
frames = range(0, len(t), SKIP)

ani = animation.FuncAnimation(
    fig, update, frames=frames,
    init_func=init, interval=20, blit=True
)

plt.tight_layout()
plt.show()
