import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants (natural units: 8πG = 1, so H² = ρ/3)
H0 = 1.0  # normalize to 1 today
Omega_m0 = 0.3
Omega_L0 = 0.7
rho_crit0 = 3 * H0**2

# Initial conditions at high redshift (a = 0.01)
a_start = 0.01
z_start = 1/a_start - 1

# Scalar field parameters
m = 0.5  # mass (in units of H0)
zeta0 = 0.01  # viscosity coefficient

# Set initial field values (small, slow-roll)
phi_start = 0.1
dphi_start = 0.0

# Initial matter density
rho_m_start = Omega_m0 * rho_crit0 / a_start**3

# Initial Hubble parameter (matter-dominated approx)
H_start = np.sqrt(rho_m_start / 3)

def flrw_viscous(t, y):
    """FLRW + scalar field with bulk viscosity.
    y = [a, phi, dphi]
    """
    a, phi, dphi = y

    # Potential (quadratic DM-like + constant DE)
    U = 0.5 * m**2 * phi**2 + Omega_L0 * rho_crit0
    dU_dphi = m**2 * phi

    # Scalar field energy and pressure
    rho_phi = 0.5 * dphi**2 + U
    P_phi = 0.5 * dphi**2 - U

    # Matter density (dust)
    rho_m = Omega_m0 * rho_crit0 / a**3
    P_m = 0

    # Total
    rho_tot = rho_phi + rho_m
    P_tot = P_phi + P_m

    # Hubble parameter from Friedmann
    H = np.sqrt(rho_tot / 3)

    # Bulk viscosity (ζ = zeta0 * H * rho_tot, example form)
    zeta = zeta0 * H * rho_tot
    P_eff = P_tot - 3 * zeta * H

    # Friedmann acceleration
    dH_dt = - (rho_tot + 3*P_eff) / 2

    # Klein-Gordon
    ddphi = -3 * H * dphi - dU_dphi

    # da/dt
    da_dt = a * H

    return [da_dt, dphi, ddphi]

# Time array (logarithmic spacing from early to now)
t_start = 0
t_end = 20  # enough to reach a=1
t_eval = np.logspace(-2, np.log10(t_end), 1000)

# Solve
sol = solve_ivp(flrw_viscous, [t_start, t_end],
                [a_start, phi_start, dphi_start],
                method='RK45', t_eval=t_eval)

a = sol.y[0]
phi = sol.y[1]
dphi = sol.y[2]

# Compute derived quantities
rho_phi = 0.5 * dphi**2 + (0.5 * m**2 * phi**2 + Omega_L0 * rho_crit0)
rho_m = Omega_m0 * rho_crit0 / a**3
H = np.sqrt((rho_phi + rho_m) / 3)

# Redshift
z = 1/a - 1

# ΛCDM for comparison
z_plot = np.linspace(0, 10, 100)
Hz_LCDM = H0 * np.sqrt(Omega_m0*(1+z_plot)**3 + Omega_L0)

# Interpolate VES results to same z range for comparison
H_interp = interp1d(z, H, bounds_error=False, fill_value='extrapolate')
Hz_VES = H_interp(z_plot)

# Plot H(z)
plt.figure(figsize=(10, 6))
plt.plot(z_plot, Hz_VES / H0, 'b-', label='VES (viscous)')
plt.plot(z_plot, Hz_LCDM / H0, 'r--', label='ΛCDM')
plt.xlabel('z')
plt.ylabel('H(z) / H0')
plt.legend()
plt.title('Hubble Parameter Evolution')
plt.grid(True)
plt.xlim(0, 10)
plt.show()

# Growth suppression estimate (simplified)
# Growth factor D(a) ~ a * exp(-∫ ζ da) approximation
def growth_factor(a, zeta0):
    # Simple exponential suppression from viscosity
    return a * np.exp(-zeta0 * np.log(1/a))

a_plot = np.linspace(0.1, 1, 100)
D_ves = growth_factor(a_plot, zeta0)
D_lcdm = a_plot  # linear growth in matter era

plt.figure(figsize=(10, 6))
plt.plot(a_plot, D_ves / D_ves[-1], 'b-', label='VES (suppressed)')
plt.plot(a_plot, D_lcdm / D_lcdm[-1], 'r--', label='ΛCDM (linear)')
plt.xlabel('a')
plt.ylabel('D(a) / D(1)')
plt.legend()
plt.title('Growth Factor Suppression from Viscosity')
plt.grid(True)
plt.show()

# fσ8 suppression (approximate)
sigma8_0 = 0.8  # typical ΛCDM value
sigma8_ves = sigma8_0 * (D_ves / a_plot)  # rough scaling

plt.figure(figsize=(10, 6))
plt.plot(z_plot, sigma8_ves * np.ones_like(z_plot), 'b-', label='VES (suppressed)')
plt.axhline(y=sigma8_0, color='r', linestyle='--', label='ΛCDM')
plt.xlabel('z')
plt.ylabel('σ8(z)')
plt.legend()
plt.title('σ8 Suppression at Low z')
plt.grid(True)
plt.show()