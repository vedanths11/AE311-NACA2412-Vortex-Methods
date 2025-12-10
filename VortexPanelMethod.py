import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def camberline(x):
    # make a function to return the y value for a given x value along the camberline
    # NACA 2 4 1 2
    m = 0.02  # maximum camber (first no. / 100)
    p = 0.4   # location of maximum camber (second no. / 10)
    
    if x < p:
        yc = m / (p**2) * (2*p*x - x**2)
    else:
        yc = m / ((1-p)**2) * ((1 - 2*p) + 2*p*x - x**2)
    
    return yc

def diff(x):
    # return the difference between neighboring points in an array
    return x[1:] - x[:-1]

alpha_vals = np.linspace(-10, 10, 21, dtype=np.intc)
Cl_vals = np.ones(alpha_vals.shape[0])
Cm_vals = np.ones(alpha_vals.shape[0])

num_panels = 20

# the coordinates of the endpoints of each panel
# everytime you see x, think x/c
# we use the normalized coordinate to simplify some equations
x = np.linspace(0, 1, num_panels + 1)

# iterate over angles of attack
for i in range(alpha_vals.shape[0]):
    alpha = alpha_vals[i]
    alpha_rad = alpha / 180 * np.pi
    
    y = np.zeros(num_panels + 1)
    for j in range(num_panels + 1):
        y[j] = camberline(x[j])
    
    # calculate the coordinates of the vortex on each panel
    # vortex is at 1/4 chord of each panel
    xv = x[:-1] + 0.25 * (x[1:] - x[:-1])
    yv = y[:-1] + 0.25 * (y[1:] - y[:-1])
    
    # calculate the coordinates of the control point on each panel
    # control point is at 3/4 chord of each panel
    xc = x[:-1] + 0.75 * (x[1:] - x[:-1])
    yc = y[:-1] + 0.75 * (y[1:] - y[:-1])
    
    # length of each panel
    dx = diff(x)
    dy = diff(y)
    L = np.sqrt(dx**2 + dy**2)
    
    # Initialize the A and B matrices
    A = np.ones((num_panels, num_panels))
    B = np.ones((num_panels, 1))
    
    # iterate over all panels
    for p in range(num_panels):
        for q in range(num_panels): 
            # distance between vortex q and control point p
            dx_pq = xc[p] - xv[q]
            dy_pq = yc[p] - yv[q]
            R = np.sqrt(dx_pq**2 + dy_pq**2)
            
            # calculate the different trig terms necessary
            # the angles are defined in the attached notes
            
            # angle from vortex q to control point p
            d2pq = np.arctan2(dy_pq, dx_pq)
            
            # angle of panel p (from x-axis)
            theta_p = np.arctan2(dy[p], dx[p])
            
            cos_d2pq = np.cos(d2pq)
            cos_thetap = np.cos(theta_p)
            sin_d2pq = np.sin(d2pq)
            sin_thetap = np.sin(theta_p)
            
            numerator = cos_d2pq * cos_thetap + sin_d2pq * sin_thetap
            denominator = 2 * np.pi * R
            A[p, q] = numerator / denominator
        
        # Calculate the angle of the panel p
        theta_p = np.arctan2(dy[p], dx[p])
        B[p, 0] = np.sin(alpha_rad - theta_p)
    
    # solve the linear equation for the Gamma vector
    Gamma = la.inv(A) @ B
    
    Cl = 2 * np.sum(Gamma)
    Cl_vals[i] = Cl

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(alpha_vals, Cl_vals, 'bo-', linewidth=2, markersize=6, label='Vortex Panel Method')

# Theoretical comparison (thin airfoil theory)
# Cl = 2*pi*(alpha + alpha_L0), where alpha_L0 ~ -2.06° for NACA 2412
alpha_L0 = -2.077
Cl_theory = 2 * np.pi * (alpha_vals - alpha_L0) / 180 * np.pi
plt.plot(alpha_vals, Cl_theory, 'r--', linewidth=2, label='Thin Airfoil Theory')

plt.xlabel('Angle of Attack (degrees)', fontsize=12)
plt.ylabel('Lift Coefficient (Cl)', fontsize=12)
plt.title('NACA 2412 - Lift Coefficient vs Angle of Attack', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.show()

print(f"Number of panels: {num_panels}")
print(f"\nSample results:")
for i in range(0, len(alpha_vals), 5):
    print(f"α = {alpha_vals[i]:6.2f}°  →  Cl = {Cl_vals[i]:7.4f}")