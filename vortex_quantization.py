import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# Grid
N = 256
x = np.linspace(-15, 15, N)
y = np.linspace(-15, 15, N)
X, Y = np.meshgrid(x, y)

# Initial phase: two vortices with opposite circulation
theta = np.arctan2(Y, X-5) - np.arctan2(Y, X+5)  # winding +1 at (5,0), -1 at (-5,0)
theta += 0.1 * np.random.randn(N, N)  # small noise

# Initial density: Gaussian envelope
I = np.exp(-(X**2 + Y**2)/50)

# Relaxation via diffusion (simulating dissipation)
for step in range(400):
    # Compute Laplacian of phase (simplified, ignoring branch cuts)
    lap_theta = (np.roll(theta,1,0) + np.roll(theta,-1,0) +
                 np.roll(theta,1,1) + np.roll(theta,-1,1) - 4*theta)

    # Diffusion with small step
    theta += 0.02 * lap_theta

    # Keep phase modulo 2π (unwrap carefully)
    theta = np.mod(theta + np.pi, 2*np.pi) - np.pi

    # Occasional smoothing to remove noise
    if step % 50 == 0:
        theta = gaussian_filter(theta, sigma=0.5, mode='wrap')

# Compute velocity field from phase gradient
ux = np.gradient(theta, axis=1)
uy = np.gradient(theta, axis=0)

# Compute vorticity ω = ∂x uy - ∂y ux
vort = np.gradient(uy, axis=1) - np.gradient(ux, axis=0)

# Function to compute circulation accurately using interpolation
def circulation_accurate(xc, yc, radius=2.0, n_points=200):
    """Compute ∮ u·dl around a circle using griddata interpolation."""
    # Points on circle
    angles = np.linspace(0, 2*np.pi, n_points)
    circle_x = xc + radius * np.cos(angles)
    circle_y = yc + radius * np.sin(angles)

    # Create grid of original points
    points = np.array([X.ravel(), Y.ravel()]).T
    ux_flat = ux.ravel()
    uy_flat = uy.ravel()

    # Interpolate velocity at circle points
    ux_circ = griddata(points, ux_flat, (circle_x, circle_y), method='linear', fill_value=0)
    uy_circ = griddata(points, uy_flat, (circle_x, circle_y), method='linear', fill_value=0)

    # Tangent vectors
    tx = -np.sin(angles)
    ty = np.cos(angles)

    # Line integral (trapezoidal rule)
    ds = radius * (angles[1:] - angles[:-1])
    circ = np.sum((ux_circ[:-1]*tx[:-1] + uy_circ[:-1]*ty[:-1]) * ds)

    return circ

# Compute circulation around each vortex
circ1 = circulation_accurate(5.0, 0.0, radius=2.0)
circ2 = circulation_accurate(-5.0, 0.0, radius=2.0)

print(f"Circulation around (5,0): {circ1:.3f} (expected ~ {2*np.pi:.3f})")
print(f"Circulation around (-5,0): {circ2:.3f} (expected ~ {-2*np.pi:.3f})")
print(f"Quantized in units of 2π: {circ1/(2*np.pi):.3f}, {circ2/(2*np.pi):.3f}")

# Detect branch cuts (where phase jumps by 2π)
phase_jump_x = np.abs(np.diff(theta, axis=1)) > np.pi
phase_jump_y = np.abs(np.diff(theta, axis=0)) > np.pi

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Phase
im1 = axes[0,0].imshow(theta, extent=[-15,15,-15,15], cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[0,0].set_title('Phase θ')
axes[0,0].set_xlabel('x'); axes[0,0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0,0])

# Vorticity
im2 = axes[0,1].imshow(vort, extent=[-15,15,-15,15], cmap='RdBu', vmin=-1, vmax=1)
axes[0,1].set_title('Vorticity ω')
axes[0,1].set_xlabel('x'); axes[0,1].set_ylabel('y')
plt.colorbar(im2, ax=axes[0,1])

# Branch cuts (vortex cores)
axes[1,0].imshow(theta, extent=[-15,15,-15,15], cmap='gray', alpha=0.5)
axes[1,0].scatter(X[:-1,:-1][phase_jump_x], Y[:-1,:-1][phase_jump_x],
                 c='red', s=1, label='Phase jumps')
axes[1,0].scatter(X[:-1,:-1][phase_jump_y], Y[:-1,:-1][phase_jump_y],
                 c='blue', s=1, label='Phase jumps')
axes[1,0].set_title('Vortex Cores (Phase Discontinuities)')
axes[1,0].set_xlabel('x'); axes[1,0].set_ylabel('y')
axes[1,0].set_xlim([-15,15]); axes[1,0].set_ylim([-15,15])

# Velocity field (quiver, subsampled)
skip = 4
axes[1,1].quiver(X[::skip,::skip], Y[::skip,::skip],
                 ux[::skip,::skip], uy[::skip,::skip], scale=50)
axes[1,1].set_title('Velocity Field')
axes[1,1].set_xlabel('x'); axes[1,1].set_ylabel('y')
axes[1,1].set_xlim([-15,15]); axes[1,1].set_ylim([-15,15])

plt.tight_layout()
plt.show()