import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class FDTD2D:
    def __init__(self, nx=400, ny=300, pml_thickness=30, direction_degrees=30):
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.pml = pml_thickness
        
        # Physical constants
        self.c0 = 1.0
        self.epsilon0 = 1.0
        self.mu0 = 1.0
        
        # Grid parameters
        self.dx = 1.0
        self.dy = 1.0
        self.dt = 0.5 * min(self.dx, self.dy) / self.c0
        
        # Initialize fields
        self.Ez = np.zeros((nx, ny))
        self.Hx = np.zeros((nx, ny))
        self.Hy = np.zeros((nx, ny))
        
        # Source parameters
        self.source_x = 40
        self.source_y = ny // 2
        self.omega0 = 0.2  # Base frequency ω0
        self.beam_width = 20
        
        # Medium parameters based on equation (6) from the paper
        self.medium_start = 100
        self.medium_width = 20
        self.c_mu = 2.0  # Base permeability constant
        self.phi_dc = 0.1  # DC phase component
        self.phi_rf = 0.5  # RF phase modulation depth
        self.beta_m = 0.8  # Spatial modulation frequency
        self.omega_m = 0.3  # Temporal modulation frequency (ωm > ω0)
        self.phi = 0.0     # Phase offset
        
        # Enhanced PML parameters
        self.setup_enhanced_pml()
        
        # Initialize space-time varying permeability
        self.mu_medium = np.ones((nx, ny)) * self.mu0
        
    def setup_enhanced_pml(self):
        self.sigma_x = np.zeros((self.nx, self.ny))
        self.sigma_y = np.zeros((self.nx, self.ny))
        
        pml_profile = lambda x: (x ** 3)
        
        for i in range(self.pml):
            x = i / self.pml
            sigma = pml_profile(1 - x)
            
            self.sigma_x[i, :] = sigma
            self.sigma_x[-(i+1), :] = sigma
            self.sigma_y[:, i] = sigma
            self.sigma_y[:, -(i+1)] = sigma
        
        self.sigma = np.sqrt(self.sigma_x**2 + self.sigma_y**2)
    
    def update_medium(self, t):
        # Update the permeability according to equation (6)
        x_coords = np.arange(self.medium_start, self.medium_start + self.medium_width)
        y_coords = np.arange(self.ny)
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        phase_term = self.phi_dc + self.phi_rf * np.sin(
            self.beta_m * Y - self.omega_m * t + self.phi
        )
        
        self.mu_medium[self.medium_start:self.medium_start + self.medium_width, :] = (
            self.mu0 * self.c_mu * (1 / np.cos(phase_term))
        )
    
    def gaussian_pulse(self, t, y):
        envelope = np.exp(-(y - self.source_y)**2 / (2 * self.beam_width**2))
        oscillation = np.sin(2 * np.pi * self.omega0 * t)
        temporal_envelope = np.exp(-(t - 30)**2 / 100)
        return envelope * oscillation * temporal_envelope
    
    def update_fields(self, t):
        self.update_medium(t)
        
        # Update H-fields with improved PML
        self.Hx[:, :-1] = (self.Hx[:, :-1] * (1 - self.sigma_x[:, :-1] * self.dt) - 
            self.dt / (self.mu_medium[:, :-1]) * (self.Ez[:, 1:] - self.Ez[:, :-1]) / self.dy)
        
        self.Hy[:-1, :] = (self.Hy[:-1, :] * (1 - self.sigma_y[:-1, :] * self.dt) + 
            self.dt / (self.mu_medium[:-1, :]) * (self.Ez[1:, :] - self.Ez[:-1, :]) / self.dx)
        
        # Update E-field with improved PML
        self.Ez[1:-1, 1:-1] = (self.Ez[1:-1, 1:-1] * (1 - self.sigma[1:-1, 1:-1] * self.dt) + 
            self.dt / self.epsilon0 * (
                (self.Hy[1:-1, 1:-1] - self.Hy[:-2, 1:-1]) / self.dx -
                (self.Hx[1:-1, 1:-1] - self.Hx[1:-1, :-2]) / self.dy
            ))
        
        # Add source
        self.Ez[self.source_x, :] += self.gaussian_pulse(t, np.arange(self.ny))

# Create simulation
fdtd = FDTD2D(nx=400, ny=300, pml_thickness=30)

# Set up figure
fig, ax = plt.subplots(figsize=(15, 10))

# Initialize plot
im = ax.imshow(fdtd.Ez.T, 
               cmap='RdYlBu_r',
               aspect='equal',
               vmin=-0.5, vmax=0.5,
               origin='lower')

# Add colorbar
plt.colorbar(im, ax=ax)

# Add medium boundary lines with enhanced visibility
ax.axvline(x=fdtd.medium_start, color='black', linewidth=2, label='Medium boundaries')
ax.axvline(x=fdtd.medium_start + fdtd.medium_width, color='black', linewidth=2)

# Add shaded region to highlight medium
ax.axvspan(fdtd.medium_start, fdtd.medium_start + fdtd.medium_width, 
           color='gray', alpha=0.2, label='Medium region')

# Add labels and title
ax.set_xlabel('x(λ₀)')
ax.set_ylabel('y(λ₀)')
ax.set_title('TE Wave Beam-Splitting Frequency Generation (ωm > ω0)')
ax.legend()

def init():
    im.set_array(fdtd.Ez.T)
    return im,

def update(frame):
    # Multiple updates per frame for smoother wave propagation
    for _ in range(3):
        fdtd.update_fields(frame * fdtd.dt)
    
    im.set_array(fdtd.Ez.T)
    return im,

anim = FuncAnimation(
    fig, 
    update,
    init_func=init,
    frames=400,
    interval=20,
    blit=False
)

plt.tight_layout()
plt.show()