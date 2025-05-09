import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons

# Function to calculate particle trajectory in magnetic field
def calculate_trajectory(q, m, v0, B, t_max, dt):
    """
    Calculate particle trajectory in a uniform magnetic field
    
    Parameters:
    - q: charge of the particle (Coulombs)
    - m: mass of the particle (kg)
    - v0: initial velocity vector [vx, vy, vz] (m/s)
    - B: magnetic field vector [Bx, By, Bz] (Tesla)
    - t_max: maximum time to simulate (seconds)
    - dt: time step (seconds)
    
    Returns:
    - positions: array of particle positions [x, y, z] over time
    - velocities: array of particle velocities [vx, vy, vz] over time
    """
    # Initialize arrays
    steps = int(t_max / dt)
    positions = np.zeros((steps, 3))
    velocities = np.zeros((steps, 3))
    
    # Set initial conditions
    positions[0] = [0, 0, 0]  # Start at origin
    velocities[0] = v0
    
    # Calculate trajectory using Lorentz force
    for i in range(1, steps):
        # F = q(v × B)
        v = velocities[i-1]
        F = q * np.cross(v, B)
        
        # a = F/m
        a = F / m
        
        # Update velocity: v = v + a*dt
        velocities[i] = velocities[i-1] + a * dt
        
        # Update position: r = r + v*dt
        positions[i] = positions[i-1] + velocities[i] * dt
        
    return positions, velocities

# Create figure and 3D axis
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Adjust subplot position to make room for sliders
plt.subplots_adjust(left=0.1, bottom=0.35)

# Initial parameters
B_strength = 1.0  # Tesla
q = 1.602e-19  # electron charge in Coulombs
m = 9.109e-31  # electron mass in kg
v_magnitude = 1e6  # initial velocity magnitude in m/s
v0 = [v_magnitude, 0, 0]  # initial velocity in m/s
t_max = 1e-8  # maximum simulation time in seconds
dt = 1e-10  # time step in seconds

# Define the magnetic field (initially downward)
B = [0, 0, -B_strength]

# Calculate initial trajectory
positions, velocities = calculate_trajectory(q, m, v0, B, t_max, dt)

# Create the grid for the magnetic field vectors
x = np.linspace(-5e-6, 5e-6, 5)
y = np.linspace(-5e-6, 5e-6, 5)
z = np.linspace(-5e-6, 5e-6, 5)
X, Y, Z = np.meshgrid(x, y, z)

# Define uniform magnetic field in the negative z-direction (downward)
Bx = np.zeros_like(X)
By = np.zeros_like(Y)
Bz = -np.ones_like(Z) * B_strength

# Initialize plot objects
field_arrows = ax.quiver(X, Y, Z, Bx, By, Bz, length=1e-6, normalize=True, color='b')
trajectory_line, = ax.plot([], [], [], 'r-', linewidth=2, label='Particle Trajectory')
particle_point, = ax.plot([], [], [], 'ro', markersize=8)

# Add a title and labels
title = ax.set_title('Charged Particle in Uniform Magnetic Field (1T Downward)', fontsize=14)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)

# Set limits based on trajectory
max_pos = np.max(np.abs(positions))
ax.set_xlim(-max_pos*1.2, max_pos*1.2)
ax.set_ylim(-max_pos*1.2, max_pos*1.2)
ax.set_zlim(-max_pos*1.2, max_pos*1.2)

# Set equal aspect ratio
ax.set_box_aspect([1, 1, 1])

# Add legend
ax.legend()

# Define animation function
def animate(i):
    i = i % len(positions)  # Loop the animation
    
    # Update trajectory up to current point
    trajectory_line.set_data(positions[:i+1, 0], positions[:i+1, 1])
    trajectory_line.set_3d_properties(positions[:i+1, 2])
    
    # Update particle position
    particle_point.set_data([positions[i, 0]], [positions[i, 1]])
    particle_point.set_3d_properties([positions[i, 2]])
    
    return trajectory_line, particle_point

# Create animation
ani = FuncAnimation(fig, animate, frames=min(100, len(positions)), 
                    interval=50, blit=True)

# Add sliders for interactive parameters
ax_B = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_v = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_charge = plt.axes([0.25, 0.1, 0.65, 0.03])

s_B = Slider(ax_B, 'B Field (T)', 0.1, 3.0, valinit=B_strength)
s_v = Slider(ax_v, 'Initial v (m/s)', 1e5, 1e7, valinit=v_magnitude)
s_charge = Slider(ax_charge, 'Charge (q/e)', -3, 3, valinit=1)

# Add reset button
ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset')

# Add radio buttons for field direction
ax_radio = plt.axes([0.025, 0.1, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('Down (-z)', 'Up (+z)', 'Right (+x)', 'Forward (+y)'))

# Define update functions
def update(val):
    # Get current values from sliders
    B_strength = s_B.val
    v_magnitude = s_v.val
    charge_factor = s_charge.val
    
    # Get current field direction
    direction = radio.value_selected
    
    if direction == 'Down (-z)':
        B = [0, 0, -B_strength]
    elif direction == 'Up (+z)':
        B = [0, 0, B_strength]
    elif direction == 'Right (+x)':
        B = [B_strength, 0, 0]
    elif direction == 'Forward (+y)':
        B = [0, B_strength, 0]
    
    # Update magnetic field vectors
    if direction == 'Down (-z)':
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        Bz = -np.ones_like(Z) * B_strength
    elif direction == 'Up (+z)':
        Bx = np.zeros_like(X)
        By = np.zeros_like(Y)
        Bz = np.ones_like(Z) * B_strength
    elif direction == 'Right (+x)':
        Bx = np.ones_like(X) * B_strength
        By = np.zeros_like(Y)
        Bz = np.zeros_like(Z)
    elif direction == 'Forward (+y)':
        Bx = np.zeros_like(X)
        By = np.ones_like(Y) * B_strength
        Bz = np.zeros_like(Z)
        
    field_arrows.remove()
    ax.quiver(X, Y, Z, Bx, By, Bz, length=1e-6, normalize=True, color='b')
    
    # Initial velocity direction is always perpendicular to B
    # This ensures we see circular motion
    if direction == 'Down (-z)' or direction == 'Up (+z)':
        v0 = [v_magnitude, 0, 0]  # Moving in x-direction when B is along z
    elif direction == 'Right (+x)':
        v0 = [0, v_magnitude, 0]  # Moving in y-direction when B is along x
    else:  # 'Forward (+y)'
        v0 = [v_magnitude, 0, 0]  # Moving in x-direction when B is along y
    
    # Update particle charge
    q_current = q * charge_factor
    
    # Recalculate trajectory
    positions, velocities = calculate_trajectory(q_current, m, v0, B, t_max, dt)
    
    # Update plot limits
    max_pos = np.max(np.abs(positions))
    ax.set_xlim(-max_pos*1, max_pos*1)
    ax.set_ylim(-max_pos*1, max_pos*1)
    ax.set_zlim(-max_pos*1, max_pos*1)
    
    # Update title
    title.set_text(f'Charged Particle in {direction} Magnetic Field ({B_strength:.1f}T)')
    
    # Reset animation
    ani.frame_seq = ani.new_frame_seq()
    
    fig.canvas.draw_idle()

def reset(event):
    s_B.reset()
    s_v.reset()
    s_charge.reset()
    radio.set_active(0)  # Reset to "Down (-z)"
    update(None)

# Connect the update functions to the widgets
s_B.on_changed(update)
s_v.on_changed(update)
s_charge.on_changed(update)
radio.on_clicked(update)
button_reset.on_clicked(reset)

# Show plot
plt.tight_layout()
plt.show()