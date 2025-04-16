import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

def lorentz_force(charge, velocity, magnetic_field):
    """Calculate the Lorentz force on a charged particle in a magnetic field."""
    return charge * np.cross(velocity, magnetic_field)

def simulate_motion(velocity, position, magnetic_field, n, dt=0.001):
    """
    Simulates the proton's motion using a loop instead of recursion.

    Parameters:
    - velocity: Initial velocity vector (m/s).
    - position: Initial position vector (m).
    - magnetic_field: Magnetic field vector (T).
    - n: Number of time steps.
    - dt: Time step (s).

    Returns:
    - List of positions for plotting.
    """
    positions = [position]  # Store position history
    velocities = [velocity]  # Store velocity history

    with open("proton_motion.txt", "w") as file:
        file.write("Proton Motion Simulation Log\n")
        file.write("=" * 50 + "\n")

        for step in range(n):
            force = lorentz_force(proton_charge, velocity, magnetic_field)
            acceleration = force / proton_mass
            velocity = velocity + acceleration * dt
            position = position + velocity * dt

            positions.append(position)
            velocities.append(velocity)

            # Write to file
            file.write(f"Step {step+1}:\n")
            file.write(f"Velocity (m/s): {velocity}\n")
            file.write(f"Position (m): {position}\n")
            file.write(f"Force (N): {force}\n")
            file.write(f"Acceleration (m/s²): {acceleration}\n")
            file.write("-" * 50 + "\n")

    return np.array(positions)

def plot_trajectory(positions):
    """
    Plots the trajectory of the proton in 2D space (XY plane).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(positions[:, 0], positions[:, 1], marker="o", linestyle="-", markersize=2, label="Proton Path")
    
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Proton Trajectory in Magnetic Field")
    plt.legend()
    plt.grid(True)
    plt.show()

# Constants
proton_charge = const.elementary_charge  # Charge of a proton in coulombs
proton_mass = const.proton_mass  # Mass of a proton in kg

# Initial conditions
initial_velocity = np.array([0.001, 9.57e-5, 0])  # Example velocity (m/s)
initial_position = np.array([0, 0, 0])  # Proton starts at the origin
magnetic_field = np.array([0, 0, -1e-6])  # Example magnetic field (T)

# Number of time steps to simulate
num_steps = 1000  # Increase for a longer simulation

# Run simulation
position_history = simulate_motion(initial_velocity, initial_position, magnetic_field, num_steps)

# Plot the results
plot_trajectory(position_history)
