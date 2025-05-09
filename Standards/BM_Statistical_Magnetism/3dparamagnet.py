import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Adjust subplot position to make room for slider
plt.subplots_adjust(bottom=0.25)

# Initial field strength value
initial_B = 1.0  # Tesla

# Function to create streamlines for magnetic field
def plot_field_lines(B_strength):
    # Clear previous plot elements
    ax.clear()
    
    # Set up starting points for field lines
    # For a uniform field, we'll create a grid of starting points in the xy-plane
    x_start = np.linspace(-5, 5, 8)
    y_start = np.linspace(-5, 5, 8)
    X_start, Y_start = np.meshgrid(x_start, y_start)
    Z_start = np.ones_like(X_start) * 5  # Start at the top
    
    # Flatten arrays for starting points
    x_start = X_start.flatten()
    y_start = Y_start.flatten()
    z_start = Z_start.flatten()
    
    # Create field lines
    for i in range(len(x_start)):
        # For each starting point, create a line going downward
        # For a uniform field in -z direction, the lines are straight
        z_line = np.linspace(5, -5, 100)
        x_line = np.ones_like(z_line) * x_start[i]
        y_line = np.ones_like(z_line) * y_start[i]
        
        # Plot the field line
        ax.plot(x_line, y_line, z_line, 'b-', linewidth=1.5, alpha=0.8)
    
    # Add small dots at regular intervals along field lines to indicate direction
    # and magnitude (more dots = stronger field)
    # num_dots = int(10 * B_strength)  # Number of dots scales with field strength
    # z_dots = np.linspace(5, -5, num_dots)
    
    # for i in range(len(x_start)):
    #     x_dots = np.ones_like(z_dots) * x_start[i]
    #     y_dots = np.ones_like(z_dots) * y_start[i]
    #     ax.plot(x_dots, y_dots, z_dots, 'bo', markersize=4)
    
    # Add a title and labels
    ax.set_title(f'Uniform Magnetic Field ({B_strength:.1f} Tesla Downward)', fontsize=14)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    # Add a text annotation to indicate the field strength
    ax.text(5, 5, 5, f'B = {B_strength:.1f}T (downward)', fontsize=10)
    
    # Set limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add reference arrow for field direction
    ax.quiver(4, 4, 0, 0, 0, -2, color='r', linewidth=2, 
              label=f'B = {B_strength:.1f}T', arrow_length_ratio=0.3)
    
    # Add legend
    ax.legend()

# Initial plot
plot_field_lines(initial_B)

# Create slider axis
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
b_slider = Slider(
    ax=ax_slider,
    label='Field Strength (Tesla)',
    valmin=0.1,
    valmax=3.0,
    valinit=initial_B,
)

# Update function for slider
def update(val):
    B_strength = b_slider.val
    plot_field_lines(B_strength)
    fig.canvas.draw_idle()

# Connect the update function to the slider
b_slider.on_changed(update)

# Show the plot
plt.tight_layout()
plt.show()