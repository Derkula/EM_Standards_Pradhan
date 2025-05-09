import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations, product
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """Custom 3D arrow for better visualization"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def calculate_fields_at_boundary(B_ext, chi, face, amplify_factor=1000):
    """
    Calculate the magnetic fields and currents at a specific face of the aluminum cube.
    
    Parameters:
    - B_ext: External magnetic field vector [Bx, By, Bz] in Tesla
    - chi: Magnetic susceptibility of the material (dimensionless)
    - face: Name of the face ('top', 'bottom', 'left', 'right', 'front', 'back')
    - amplify_factor: Factor to amplify the small effects for visualization
    
    Returns:
    - Dictionary containing all relevant field vectors and values
    """
    mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability
    
    # Define outward normals for each face
    normals = {
        'top': np.array([0, 0, 1]),     # z+ face
        'bottom': np.array([0, 0, -1]), # z- face
        'left': np.array([-1, 0, 0]),   # x- face
        'right': np.array([1, 0, 0]),   # x+ face
        'front': np.array([0, 1, 0]),   # y+ face
        'back': np.array([0, -1, 0])    # y- face
    }
    
    n = normals[face]  # Normal vector for the current face
    
    # Calculate H field from B field (H = B/μ₀)
    H_ext = B_ext / mu_0
    
    # Calculate magnetization (M = χH for a linear paramagnetic material)
    M = chi * H_ext
    
    # Amplify M for visualization purposes (since χ is very small for aluminum)
    M_amplified = M * amplify_factor
    
    # Calculate effective surface current (K_m = M × n)
    K_m = np.cross(M, n)
    K_m_amplified = np.cross(M_amplified, n)
    
    # Inside the material B_in = μ₀(H_ext + M)
    B_in = B_ext + mu_0 * M
    
    # Outside the material - just B_ext
    B_out = B_ext.copy()
    
    # The tangential component of B has a discontinuity given by our boundary condition
    # B_above^‖ - B_below^‖ = μ₀(K × n)
    # Calculate this discontinuity
    tangential_jump = mu_0 * np.cross(K_m, n)
    tangential_jump_amplified = mu_0 * np.cross(K_m_amplified, n)
    
    # Extract normal and tangential components
    B_normal_component = np.dot(B_ext, n) * n  # Normal component is continuous
    
    # Extract tangential components
    B_in_tangential = B_in - np.dot(B_in, n) * n
    B_out_tangential = B_out - np.dot(B_out, n) * n
    
    # The actual discontinuity in tangential components should match our boundary condition
    actual_jump = B_out_tangential - B_in_tangential
    
    return {
        'face': face,
        'normal': n,
        'B_ext': B_ext,
        'M': M,
        'M_amplified': M_amplified,
        'K_m': K_m,
        'K_m_amplified': K_m_amplified,
        'B_in': B_in,
        'B_out': B_out,
        'B_normal_component': B_normal_component,
        'B_in_tangential': B_in_tangential,
        'B_out_tangential': B_out_tangential,
        'tangential_jump': tangential_jump,
        'tangential_jump_amplified': tangential_jump_amplified,
        'actual_jump': actual_jump
    }

def visualize_boundary_condition(B_ext, chi, amplify_factor=1000):
    """
    Create a comprehensive visualization of the boundary conditions on all faces
    of the aluminum cube, with a focus on the specified boundary condition equation.
    
    Parameters:
    - B_ext: External magnetic field vector [Bx, By, Bz] in Tesla
    - chi: Magnetic susceptibility of the material (dimensionless)
    - amplify_factor: Factor to amplify the small effects for visualization
    """
    # Face names and their center coordinates on the unit cube
    faces = {
        'top': {'center': np.array([0, 0, 0.5]), 'color': 'lightblue'},
        'bottom': {'center': np.array([0, 0, -0.5]), 'color': 'lightgreen'},
        'left': {'center': np.array([-0.5, 0, 0]), 'color': 'lightcoral'},
        'right': {'center': np.array([0.5, 0, 0]), 'color': 'lightsalmon'},
        'front': {'center': np.array([0, 0.5, 0]), 'color': 'lightpink'},
        'back': {'center': np.array([0, -0.5, 0]), 'color': 'wheat'}
    }
    
    # Calculate fields for each face
    results = {}
    for face_name in faces.keys():
        results[face_name] = calculate_fields_at_boundary(
            B_ext, chi, face_name, amplify_factor)
    
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(15, 12))
    
    # Create 3D axis for the cube visualization
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Draw the aluminum cube (as a wireframe)
    r = [-0.5, 0.5]
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax1.plot3D(*zip(s, e), color="gray", alpha=0.7, linewidth=2)
    
    # Draw faces with colors
    for face_name, face_info in faces.items():
        if face_name == 'top':
            x, y = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
            z = np.ones_like(x) * 0.5
            ax1.plot_surface(x, y, z, alpha=0.3, color=face_info['color'])
        elif face_name == 'bottom':
            x, y = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
            z = np.ones_like(x) * (-0.5)
            ax1.plot_surface(x, y, z, alpha=0.3, color=face_info['color'])
        elif face_name == 'left':
            y, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
            x = np.ones_like(y) * (-0.5)
            ax1.plot_surface(x, y, z, alpha=0.3, color=face_info['color'])
        elif face_name == 'right':
            y, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
            x = np.ones_like(y) * 0.5
            ax1.plot_surface(x, y, z, alpha=0.3, color=face_info['color'])
        elif face_name == 'front':
            x, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
            y = np.ones_like(x) * 0.5
            ax1.plot_surface(x, y, z, alpha=0.3, color=face_info['color'])
        elif face_name == 'back':
            x, z = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
            y = np.ones_like(x) * (-0.5)
            ax1.plot_surface(x, y, z, alpha=0.3, color=face_info['color'])
    
    # Add external field vector
    origin = np.array([-1, 0, 0])  # Start external field outside the cube
    ax1.quiver(origin[0], origin[1], origin[2], 
              B_ext[0], B_ext[1], B_ext[2], 
              color='blue', label='External B-field', 
              length=0.5, normalize=True, arrow_length_ratio=0.25)
    
    # Calculate surface current loops based on the magnetization and face normals
    # For a magnetic field in z-direction and paramagnetic material, 
    # the magnetization will be in z-direction, creating surface currents circulating around z-axis
    
    # Draw the surface current loops as continuous lines wrapping around the cube
    if np.allclose(B_ext, [0, 0, 1]):  # If B-field is along z-axis
        # Surface currents will circulate in x-y plane around the z-axis
        # Top face (z=0.5) - clockwise circulation looking from +z
        theta = np.linspace(0, 2*np.pi, 100)
        x_loop = 0.4 * np.cos(theta)
        y_loop = 0.4 * np.sin(theta)
        z_loop = np.ones_like(theta) * 0.5
        ax1.plot(x_loop, y_loop, z_loop, 'g-', linewidth=2, label='Surface Current (K_m)')
        
        # Add arrows to indicate direction
        for i in range(0, 100, 25):
            # Tangent vector
            tangent = np.array([-np.sin(theta[i]), np.cos(theta[i]), 0])
            # Add arrow
            ax1.quiver(x_loop[i], y_loop[i], z_loop[i], 
                      tangent[0], tangent[1], tangent[2], 
                      color='green', length=0.1, normalize=True, arrow_length_ratio=0.3)
        
        # Bottom face (z=-0.5) - counterclockwise circulation looking from +z
        z_loop = np.ones_like(theta) * (-0.5)
        ax1.plot(x_loop, y_loop, z_loop, 'g-', linewidth=2)
        
        # Add arrows to indicate direction (opposite to top face)
        for i in range(0, 100, 25):
            # Tangent vector (opposite direction)
            tangent = np.array([np.sin(theta[i]), -np.cos(theta[i]), 0])
            # Add arrow
            ax1.quiver(x_loop[i], y_loop[i], z_loop[i], 
                      tangent[0], tangent[1], tangent[2], 
                      color='green', length=0.1, normalize=True, arrow_length_ratio=0.3)
        
        # Connecting lines on the sides
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x_line = 0.4 * np.cos(angle)
            y_line = 0.4 * np.sin(angle)
            z_line = np.linspace(-0.5, 0.5, 10)
            ax1.plot([x_line]*10, [y_line]*10, z_line, 'g-', linewidth=2)
    
    # Add field vectors at the right face (for comparison with the detailed view)
    chosen_face = 'right'
    face_center = faces[chosen_face]['center']
    result = results[chosen_face]
    
    # Add B-field inside (magenta)
    ax1.quiver(face_center[0]-0.1, face_center[1], face_center[2],
              result['B_in'][0], result['B_in'][1], result['B_in'][2],
              color='magenta', label='B-field Inside', 
              length=0.2, arrow_length_ratio=0.25)
    
    # Add B-field outside (cyan)
    ax1.quiver(face_center[0]+0.1, face_center[1], face_center[2],
              result['B_out'][0], result['B_out'][1], result['B_out'][2],
              color='cyan', label='B-field Outside', 
              length=0.2, arrow_length_ratio=0.25)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Magnetic Field and Surface Currents at Aluminum Cube Boundary')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Create a detailed diagram of the boundary condition for the right face
    ax2 = fig.add_subplot(222, projection='3d')
    
    # Focus on the right face
    face_result = results['right']
    face_center = faces['right']['center']
    
    # Create a zoomed view of the right face
    # Draw the face
    y, z = np.meshgrid([-0.3, 0.3], [-0.3, 0.3])
    x = np.ones_like(y) * 0.5
    ax2.plot_surface(x, y, z, alpha=0.4, color='lightsalmon')
    
    # Offset points for inside and outside
    inside_point = face_center - np.array([0.1, 0, 0])
    outside_point = face_center + np.array([0.1, 0, 0])
    
    # Draw normal vector
    ax2.quiver(face_center[0], face_center[1], face_center[2],
               face_result['normal'][0], face_result['normal'][1], face_result['normal'][2],
               color='black', label='Normal (n)', length=0.2, arrow_length_ratio=0.25)
    
    # Add vectors at inside point
    ax2.quiver(inside_point[0], inside_point[1], inside_point[2],
              face_result['B_in'][0], face_result['B_in'][1], face_result['B_in'][2],
              color='magenta', label='B Inside (total)', length=0.2, arrow_length_ratio=0.25)
    
    ax2.quiver(inside_point[0], inside_point[1], inside_point[2],
              face_result['B_in_tangential'][0], face_result['B_in_tangential'][1], face_result['B_in_tangential'][2],
              color='darkmagenta', label='B Inside (tangential)', length=0.2, arrow_length_ratio=0.25)
    
    # Add vectors at outside point
    ax2.quiver(outside_point[0], outside_point[1], outside_point[2],
              face_result['B_out'][0], face_result['B_out'][1], face_result['B_out'][2],
              color='cyan', label='B Outside (total)', length=0.2, arrow_length_ratio=0.25)
    
    ax2.quiver(outside_point[0], outside_point[1], outside_point[2],
              face_result['B_out_tangential'][0], face_result['B_out_tangential'][1], face_result['B_out_tangential'][2],
              color='blue', label='B Outside (tangential)', length=0.2, arrow_length_ratio=0.25)
    
    # Draw surface current as a line segment with arrow (green)
    # The current on the right face for a z-directed field will be in the y-direction
    y_line = np.linspace(-0.2, 0.2, 10)
    z_start = 0
    x_face = 0.5
    ax2.plot([x_face]*10, y_line, [z_start]*10, 'g-', linewidth=2, label='Surface Current (K_m)')
    
    # Add arrow to the current line
    ax2.quiver(x_face, 0.15, z_start, 0, 0.1, 0, color='green', length=0.1, arrow_length_ratio=0.5)
    
    # Add amplified surface current cross normal vector (K_m × n)
    K_cross_n = np.cross(face_result['K_m_amplified'], face_result['normal'])
    if np.linalg.norm(K_cross_n) > 1e-10:
        ax2.quiver(face_center[0], face_center[1], face_center[2],
                  K_cross_n[0], K_cross_n[1], K_cross_n[2],
                  color='orange', label='K_m × n (amplified)', length=0.2, arrow_length_ratio=0.25)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Detailed View of Boundary Condition at Right Face')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Create a 2D diagram illustrating the boundary condition
    ax3 = fig.add_subplot(223)
    
    # Draw the boundary line
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    
    # Label the regions
    ax3.text(-0.4, 0.9, "Inside Aluminum\n(Below)", fontsize=12)
    ax3.text(0.1, 0.9, "Outside\n(Above)", fontsize=12)
    
    # Set up coordinates
    arrow_length = 0.3
    y_positions = [0.7, 0.5, 0.3]
    
    # Draw B inside tangential
    ax3.arrow(-0.2, y_positions[0], arrow_length, 0, 
              head_width=0.03, head_length=0.05, fc='darkmagenta', ec='darkmagenta',
              label='B_below (tangential)')
    
    # Draw B outside tangential
    ax3.arrow(0.2, y_positions[0], arrow_length, 0, 
              head_width=0.03, head_length=0.05, fc='blue', ec='blue',
              label='B_above (tangential)')
    
    # Draw K_m as a line with arrows (into page)
    # Draw a circle for 'into the page' current
    current_radius = 0.02
    for y_pos in np.linspace(y_positions[1] - 0.15, y_positions[1] + 0.15, 5):
        circle = plt.Circle((0, y_pos), current_radius, color='green', fill=False)
        ax3.add_patch(circle)
        ax3.plot(0, y_pos, 'gx')  # 'x' symbol for 'into the page'
    
    # Add arrow showing direction of current flow
    ax3.annotate('', xy=(0, y_positions[1] - 0.2), xytext=(0, y_positions[1] + 0.2),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    
    # Draw difference vector
    ax3.arrow(0.2, y_positions[2], arrow_length, 0, 
              head_width=0.03, head_length=0.05, fc='red', ec='red',
              label='B_above - B_below')
    
    # Draw n vector (out of page)
    circle = plt.Circle((0, 0.15), 0.03, color='black', fill=False)
    ax3.add_patch(circle)
    ax3.plot(0, 0.15, 'k.')  # '.' symbol for 'out of the page'
    
    # Add text labels
    ax3.text(-0.4, y_positions[0], "B_below", color='darkmagenta', fontsize=10)
    ax3.text(0.5, y_positions[0], "B_above", color='blue', fontsize=10)
    ax3.text(0.05, y_positions[1], "K_m", color='green', fontsize=10)
    ax3.text(0.5, y_positions[2], "B_above - B_below", color='red', fontsize=10)
    ax3.text(0.05, 0.15, "n", color='black', fontsize=10)
    
    # Add the boundary condition equation
    ax3.text(0.15, 0.05, r"$\vec{B}_{above}^{\parallel} - \vec{B}_{below}^{\parallel} = \mu_0 (\vec{K} \times \hat{n})$", 
             fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_xlim(-0.5, 0.8)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Illustration of Boundary Condition', fontsize=14)
    

    # Create a 2D cross-section showing the physical picture
    ax3 = fig.add_subplot(224)
    
    # Draw the boundary as a vertical line
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    
    # Label the regions
    ax3.text(-0.7, 0.9, "Inside Aluminum", fontsize=12)
    ax3.text(0.1, 0.9, "Outside (Vacuum)", fontsize=12)
    
    # Draw the B-field lines (continuous across boundary but with different density)
    # For paramagnetic material, field lines are slightly more concentrated inside
    x_ext = np.linspace(-1, 1, 100)
    
    # Field lines are slightly bent toward vertical inside the material
    # (paramagnetic effect enhances the field inside)
    for y0 in np.linspace(0.1, 0.9, 5):
        # Outside the material (uniform field)
        x_outside = np.linspace(0, 1, 50)
        y_outside = np.ones_like(x_outside) * y0
        
        # Inside the material (slightly concentrated)
        x_inside = np.linspace(-1, 0, 50)
        y_inside = y0 + 0.05 * np.sin(np.pi * x_inside)  # Slight bending
        
        # Plot field lines (blue)
        ax3.plot(x_outside, y_outside, 'b-', linewidth=1.5)
        ax3.plot(x_inside, y_inside, 'b-', linewidth=1.5)
        
        # Add arrowheads to indicate field direction
        ax3.arrow(0.5, y0, 0.1, 0, head_width=0.02, head_length=0.05, 
                 fc='blue', ec='blue')
        ax3.arrow(-0.5, y_inside[25], 0.1, 0, head_width=0.02, head_length=0.05, 
                 fc='blue', ec='blue')
    
    # Draw the surface current loops (green circles with X's and dots)
    # Surface currents flow in circles on the boundary, perpendicular to both
    # the magnetization and the surface normal
    
    # Since the B-field is in the y-direction, and the boundary normal is in the x-direction,
    # the surface currents flow in the z-direction (into and out of the page)
    
    # Represent currents flowing into page (X) and out of page (⋅)
    for y_pos in np.linspace(0.2, 0.8, 4):
        # Current into page (left side of boundary, top half)
        if y_pos > 0.5:
            circle = plt.Circle((-0.05, y_pos), 0.03, color='green', fill=False)
            ax3.add_patch(circle)
            ax3.plot(-0.05, y_pos, 'gx')  # 'x' for into page
        # Current out of page (left side of boundary, bottom half)
        else:
            circle = plt.Circle((-0.05, y_pos), 0.03, color='green', fill=False)
            ax3.add_patch(circle)
            ax3.plot(-0.05, y_pos, 'g.')  # '.' for out of page
    
    # Add arrows showing the circulation of current
    arrow_height = 0.3
    ax3.arrow(-0.05, 0.3, 0, arrow_height, head_width=0.02, head_length=0.05, 
             fc='green', ec='green')
    ax3.arrow(-0.05, 0.7, 0, -arrow_height, head_width=0.02, head_length=0.05, 
             fc='green', ec='green')
    
    # Add labels
    ax3.text(-0.15, 0.5, "K", color='green', fontsize=12, ha='center')
    ax3.text(-0.5, 0.05, "B-field", color='blue', fontsize=12)
    
    # Add explanation of the surface current effect
    explanation = (
        "Surface currents appear at the boundary of\n"
        "the magnetized material. They cause a\n"
        "discontinuity in the tangential component\n"
        "of the B-field according to:\n"
        r"$\vec{B}_{above}^{\parallel} - \vec{B}_{below}^{\parallel} = \mu_0 (\vec{K} \times \hat{n})$"
    )
    ax3.text(0.2, 0.3, explanation, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.set_title('Physical Picture of Boundary Conditions', fontsize=14)
    ax3.axis('off')
    
    # Create a numerical results table
    ax4 = fig.add_subplot(223)
    face_result = results['right']  # Use right face for the table
    
    # Format vectors for display
    def format_vector(v):
        return f"[{v[0]:.2e}, {v[1]:.2e}, {v[2]:.2e}]"
    
    # Data for the table
    data = [
        ['External B-field', format_vector(face_result['B_ext']), 'Tesla (T)'],
        ['Magnetization (M)', format_vector(face_result['M']), 'A/m'],
        ['Magnetization (amplified)', format_vector(face_result['M_amplified']), 'A/m'],
        ['Surface Current (K_m)', format_vector(face_result['K_m']), 'A/m'],
        ['Surface Current (amplified)', format_vector(face_result['K_m_amplified']), 'A/m'],
        ['B Inside Material', format_vector(face_result['B_in']), 'Tesla (T)'],
        ['B Outside Material', format_vector(face_result['B_out']), 'Tesla (T)'],
        ['B Inside (tangential)', format_vector(face_result['B_in_tangential']), 'Tesla (T)'],
        ['B Outside (tangential)', format_vector(face_result['B_out_tangential']), 'Tesla (T)'],
        ['Tangential Jump (B₂-B₁)ₜ', format_vector(face_result['tangential_jump']), 'Tesla (T)'],
        ['μ₀(K_m × n)', format_vector(face_result['tangential_jump']), 'Tesla (T)']
    ]
    
    # Create table
    table = ax4.table(cellText=data, colLabels=['Quantity', 'Value', 'Units'],
                      loc='center', cellLoc='center')
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax4.axis('off')
    ax4.set_title('Numerical Values for Right Face', fontsize=14)
    
    # Add a main title
    plt.suptitle('Magnetic Boundary Conditions for Aluminum in External Field', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, results
    
    # Create a 2D diagram illustrating the boundary condition
    ax3 = fig.add_subplot(223)
    
    # Draw the boundary line
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=2)
    
    # Label the regions
    ax3.text(-0.4, 0.9, "Inside Aluminum\n(Below)", fontsize=12)
    ax3.text(0.1, 0.9, "Outside\n(Above)", fontsize=12)
    
    # Set up coordinates
    arrow_length = 0.3
    y_positions = [0.7, 0.5, 0.3]
    
    # Draw B inside tangential
    ax3.arrow(-0.2, y_positions[0], arrow_length, 0, 
              head_width=0.03, head_length=0.05, fc='darkmagenta', ec='darkmagenta',
              label='B_below (tangential)')
    
    # Draw B outside tangential
    ax3.arrow(0.2, y_positions[0], arrow_length, 0, 
              head_width=0.03, head_length=0.05, fc='blue', ec='blue',
              label='B_above (tangential)')
    
    # Draw K_m
    ax3.arrow(0, y_positions[1], 0, arrow_length, 
              head_width=0.03, head_length=0.05, fc='green', ec='green',
              label='K_m (into page)')
    
    # Draw difference vector
    ax3.arrow(0.2, y_positions[2], arrow_length, 0, 
              head_width=0.03, head_length=0.05, fc='red', ec='red',
              label='B_above - B_below')
    
    # Draw n vector (out of page)
    circle = plt.Circle((0, 0.15), 0.03, color='black', fill=False)
    ax3.add_patch(circle)
    ax3.plot(0, 0.15, 'k+')
    
    # Add text labels
    ax3.text(-0.4, y_positions[0], "B_below", color='darkmagenta', fontsize=10)
    ax3.text(0.5, y_positions[0], "B_above", color='blue', fontsize=10)
    ax3.text(0.05, y_positions[1] + arrow_length, "K_m", color='green', fontsize=10)
    ax3.text(0.5, y_positions[2], "B_above - B_below", color='red', fontsize=10)
    ax3.text(0.05, 0.15, "n", color='black', fontsize=10)
    
    # Add the boundary condition equation
    ax3.text(0.15, 0.05, r"$\vec{B}_{above}^{\parallel} - \vec{B}_{below}^{\parallel} = \mu_0 (\vec{K} \times \hat{n})$", 
             fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    ax3.set_xlim(-0.5, 0.8)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Illustration of Boundary Condition', fontsize=14)
    
    # Create a table showing the numerical values
    ax4 = fig.add_subplot(224)
    face_result = results['right']  # Use right face for the table
    
    # Format vectors for display
    def format_vector(v):
        return f"[{v[0]:.2e}, {v[1]:.2e}, {v[2]:.2e}]"
    
    # Data for the table
    data = [
        ['External B-field', format_vector(face_result['B_ext']), 'Tesla (T)'],
        ['Magnetization (M)', format_vector(face_result['M']), 'A/m'],
        ['Magnetization (amplified)', format_vector(face_result['M_amplified']), 'A/m'],
        ['Surface Current (K_m)', format_vector(face_result['K_m']), 'A/m'],
        ['Surface Current (amplified)', format_vector(face_result['K_m_amplified']), 'A/m'],
        ['B Inside Material', format_vector(face_result['B_in']), 'Tesla (T)'],
        ['B Outside Material', format_vector(face_result['B_out']), 'Tesla (T)'],
        ['B Inside (tangential)', format_vector(face_result['B_in_tangential']), 'Tesla (T)'],
        ['B Outside (tangential)', format_vector(face_result['B_out_tangential']), 'Tesla (T)'],
        ['Tangential Jump (B₂-B₁)ₜ', format_vector(face_result['tangential_jump']), 'Tesla (T)'],
        ['μ₀(K_m × n)', format_vector(face_result['tangential_jump']), 'Tesla (T)']
    ]
    
    # Create table
    table = ax4.table(cellText=data, colLabels=['Quantity', 'Value', 'Units'],
                      loc='center', cellLoc='center')
    
    # Set table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    ax4.axis('off')
    ax4.set_title('Numerical Values for Right Face', fontsize=14)
    
    # Add a main title
    plt.suptitle('Magnetic Boundary Conditions for Aluminum in External Field', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig, results

def main():
    """Main function to run the simulation and visualization"""
    
    # Set parameters
    B_ext = np.array([0, 0, 1.0])  # External B-field in z-direction (1 Tesla)
    chi_Al = 2.2e-5  # Magnetic susceptibility of aluminum
    
    # Run the visualization with amplified effects for clarity
    amplify_factor = 1000  # Make the small effects visible
    fig, results = visualize_boundary_condition(B_ext, chi_Al, amplify_factor)
    
    # Display numerical verification of the boundary condition
    print("Verification of Boundary Condition: B_above^‖ - B_below^‖ = μ₀(K × n)")
    print("======================================================================")
    
    # Check for all faces
    mu_0 = 4 * np.pi * 1e-7
    for face_name, result in results.items():
        # Tangential component difference (directly calculated)
        actual_jump = result['B_out_tangential'] - result['B_in_tangential']
        
        # Expected jump from boundary condition
        expected_jump = mu_0 * np.cross(result['K_m'], result['normal'])
        
        # Calculate the error
        error_vector = actual_jump - expected_jump
        error_magnitude = np.linalg.norm(error_vector)
        
        print(f"\nFace: {face_name}")
        print(f"Actual jump: {actual_jump}")
        print(f"Expected jump: {expected_jump}")
        print(f"Error magnitude: {error_magnitude}")
        
        # Check if they match
        if error_magnitude < 1e-12:
            print(f"✓ Boundary condition verified for {face_name} face")
        else:
            print(f"✗ Discrepancy detected for {face_name} face")
    
    # Save the figure
    plt.savefig('aluminum_boundary_conditions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()