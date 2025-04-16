import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

# Constants
e0 = const.epsilon_0
e = const.elementary_charge
k = 1 / (4 * np.pi * e0)

# Dipole Parameters
d = 2e-2  # Distance between dipole charges (2cm)
q = e  # Elementary charge
field_range = 6e-2  # Field display range (in meters)

# Define charge positions
pos_charge = np.array([d/2, 0])  # Positive charge at (0, d/2)
neg_charge = np.array([-d/2, 0])  # Negative charge at (0, -d/2)

def mag_hat(vector):
    """Calculate magnitude and unit vector."""
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return magnitude, np.array([0, 0])  # Avoid division by zero
    hat = vector / magnitude
    return magnitude, hat

def e_calc(charge, rmag, rhat):
    """Calculate the electric field using Coulomb's Law."""
    if rmag == 0:
        return np.array([0, 0])  # Avoid singularities
    return (k * charge / rmag**3) * rhat

def calculate_electric_field(point):
    """Computes the electric field at a specific point."""
    # Field from positive charge
    r_p = point - pos_charge
    rmag_p, rhat_p = mag_hat(r_p)
    E_p = e_calc(q, rmag_p, rhat_p)

    # Field from negative charge
    r_n = point - neg_charge
    rmag_n, rhat_n = mag_hat(r_n)
    E_n = e_calc(-q, rmag_n, rhat_n)

    # Superposition: E_total = E_pos + E_neg
    return E_p + E_n

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 7))

# Plot charge locations
ax.scatter(*pos_charge, color="red", s=100, label="Positive Charge (+q)")
ax.scatter(*neg_charge, color="black", s=100, label="Negative Charge (-q)")
# Convert axis limits to cm
ax.set_xlim(-field_range * 1e2, field_range * 1e2)
ax.set_ylim(-field_range* 1e2, field_range * 1e2)

# Update labels to show cm instead of meters
# ax.set_xlabel("x (cm)", fontsize=14)
# ax.set_ylabel("y (cm)", fontsize=14)
ax.set_title("Electric Field Near a Dipole (Mouse Interactive, cm Scale)")


# Labels and display settings
#ax.set_xlabel("x (m)")
#ax.set_ylabel("y (m)")
# ax.set_title("Electric Field Near a Dipole (Mouse Interactive)")
ax.legend()
ax.set_xlim(-field_range, field_range)
ax.set_ylim(-field_range, field_range)
ax.grid(True)

# Initialize an empty quiver arrow (will update dynamically)
quiver = ax.quiver([], [], [], [], color="blue", angles="xy", scale_units="xy", scale = 0.2)

def update_quiver(event):
    """Update the quiver arrow and legend dynamically with the electric field vector, ensuring correct units."""
    global quiver, legend  

    if event.xdata is None or event.ydata is None:
        return  

    # Convert mouse position from cm to meters for calculations
    point_meters = np.array([event.xdata, event.ydata])

    # Compute electric field at the mouse position (still in SI units)
    E_field = calculate_electric_field(point_meters)

    # Determine a suitable scale for the vector
    E_mag, E_hat = mag_hat(E_field)
    max_field_display = 5e2  # Reference max field for scaling

    if E_mag > 0:
        E_display = (E_mag * E_hat)   # Scale for visualization
    else:
        E_display = np.array([0, 0])

    # Convert electric field vector to cm for plotting
    E_display_cm = E_display * 1e2  

    # Remove previous quiver and draw new one (plotted in cm)
    quiver.remove()
    quiver = ax.quiver(event.xdata, event.ydata, E_display_cm[0], E_display_cm[1], 
                       color="blue", angles="xy", scale_units="xy", scale=1)

    # Update legend with the new E-field vector
    legend_text = f"Electric Field: ({E_field[0]:.2e}, {E_field[1]:.2e}) V/m"
    legend.get_texts()[2].set_text(legend_text)  

    # Redraw plot
    fig.canvas.draw_idle()


# Connect mouse motion event to update function
fig.canvas.mpl_connect("motion_notify_event", update_quiver)
legend = ax.legend(["Positive Charge (+q)", "Negative Charge (-q)", "Electric Field: (0, 0) V/m"])


# Define a grid of test points (in meters)
x_vals = np.linspace(-field_range, field_range, 10)
y_vals = np.linspace(-field_range, field_range, 10)
X, Y = np.meshgrid(x_vals, y_vals)

# Prepare arrays to hold the field components
Ex = np.zeros_like(X)
Ey = np.zeros_like(Y)

# Compute electric field at each grid point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        E = calculate_electric_field(point)
        Ex[i, j], Ey[i, j] = E

# Calculate magnitudes
E_mags = np.sqrt(Ex**2 + Ey**2)

# Logarithmic scale for better visibility
E_scaled = np.log1p(E_mags)  # log1p(x) = log(1 + x), avoids log(0)

# Normalize and scale
Ex_scaled = (Ex / E_mags) * E_scaled
Ey_scaled = (Ey / E_mags) * E_scaled

# Plot fixed-point field vectors (scaled for clarity)
ax.quiver(X, Y, Ex_scaled, Ey_scaled, color="gray", alpha=0.6, angles="xy", scale_units="xy", scale="log")

# Show the interactive plot
plt.show()
