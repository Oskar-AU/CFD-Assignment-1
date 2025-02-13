from Meshes import create_lattice_mesh
from Solvers import get_transport_equation_coefficients
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt 

def exact_solution(x: npt.NDArray, Pe: float) -> npt.NDArray:
    return (np.exp(Pe * x) - 1) / (np.exp(Pe) - 1)

N = (100, 20, 20)
Pe = (100, 40, -20)
L = 1

fig, axs = plt.subplots(1, 3)
fig.set_size_inches((12, 5))
for i in range(3):
    
    # Creates the mesh.
    mesh = create_lattice_mesh(
        dimensions=(N[i],), 
        grid_size=L / N[i],
    )

    x_exact = np.linspace(0, 1, N[i], endpoint=True)
    axs[i].plot(x_exact, exact_solution(x_exact, Pe[i]), 'k-', label='Exact', linewidth=2)
    
    for j, convection_scheme in enumerate(('central_difference', 'upwind')):
        
        # Calculate the matrix and vectors.
        matrix, vector = get_transport_equation_coefficients(
            convection_scheme=convection_scheme, 
            diffusion_scheme='central_difference', 
            density=1, 
            diffusion=1/Pe[i], 
            velocity_vector=(1,), 
            mesh=mesh,
            boundary_conditions={'type': ('constant', 'constant'), 'value': (0, 1)}
        )

        axs[i].plot(np.linspace(L / N[i] / 2, L-L/N[i]/2, N[i]), np.linalg.inv(matrix) @ vector, label=convection_scheme, linestyle='--')
    axs[i].legend()
    axs[i].grid()
    axs[i].set_title(f"N = {N[i]}, Pe = {Pe[i]}")
    axs[i].set_xlabel('x [m]')
    axs[i].set_ylabel('phi')
fig.tight_layout()
plt.show()

Pe = 20
N = np.logspace(1, 3, 50, dtype=int)
grid_size = L / N
fig, ax = plt.subplots()
errors = np.empty(len(N))

for i, convection_scheme in enumerate(('central_difference', 'upwind')):
    for j in range(len(N)):
        # Creates the mesh.
        mesh = create_lattice_mesh(
            dimensions=(N[j],), 
            grid_size=grid_size[j],
        )

        # Calculate the matrix and vectors.
        matrix, vector = get_transport_equation_coefficients(
            convection_scheme=convection_scheme, 
            diffusion_scheme='central_difference', 
            density=1, 
            diffusion=1/Pe, 
            velocity_vector=(1,), 
            mesh=mesh,
            boundary_conditions={'type': ('constant', 'constant'), 'value': (0, 1)}
        )

        x = np.linspace(grid_size[j] / 2, L-grid_size[j]/2, N[j])
        errors[j] = np.sum(np.abs(exact_solution(x, Pe) - np.linalg.inv(matrix) @ vector)) / N[j]
    ax.set_yscale('log')
    ax.set_xscale('log')
    order_of_discretion_error = np.polyfit(np.log10(grid_size), np.log10(errors), 1)
    ax.plot((grid_size[0], grid_size[-1]), 10**np.dot([[np.log10(grid_size[0]), 1], [np.log10(grid_size[-1]), 1]], order_of_discretion_error), 
            label=f'Fit line ({order_of_discretion_error[0]:.2f})')
    ax.scatter(L/N, errors, label=convection_scheme)
ax.grid()
ax.legend()
ax.set_xlabel('Grid size [m]')
ax.set_ylabel('Truncation error [-]')
plt.show()