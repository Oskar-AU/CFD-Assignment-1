import igraph as ig
from typing import Literal
from Boundary_Conditions import Boundary_Condition
import numpy as np
import numpy.typing as npt

def get_transport_equation_coefficients(convection_scheme: Literal['central_difference', 'upwind'], 
                                        diffusion_scheme: Literal['central_difference'],
                                        density: float,
                                        diffusion: float,
                                        velocity_vector: npt.ArrayLike, 
                                        mesh: ig.Graph,
                                        boundary_conditions: Boundary_Condition) -> npt.NDArray:
    
    matrix = np.zeros((mesh.vcount(),)*2)
    vector = np.zeros(mesh.vcount())

    # Looping over all interal surfaces.
    for source_cell in mesh.vs:
        for target_cell in source_cell.neighbors():
            surface = mesh.get_eid(source_cell, target_cell)
            # Calculates the surface normal vector assuming that it is parallel with the vector between the verticies.
            surface_normal_vector = target_cell['spatial_coordinates'] - source_cell['spatial_coordinates']
            # Calculates distance between the two verticies.
            distance_between_cells = np.linalg.norm(surface_normal_vector)
            # Calculates the surface normal velocity assuming constant velocity across the mesh.
            surface_normal_velocity = np.dot(velocity_vector, surface_normal_vector / distance_between_cells)
            # Calculates the mass flow normal to the surface.
            mass_flow = density * surface_normal_velocity * mesh.es[surface]['surface_area']
            # Looping over all convection schemes.
            match convection_scheme:
                case 'central_difference':
                    # Equation (4.19) - assumes faces centered between nodes (see Figure 4.1).
                    linear_interpolation_coefficient = 1/2
                    # Equation (4.48).
                    convection_matrix_coefficient = linear_interpolation_coefficient * mass_flow
                    matrix[source_cell.index, target_cell.index] += convection_matrix_coefficient
                    matrix[source_cell.index, source_cell.index] -= convection_matrix_coefficient - mass_flow
                case 'upwind':
                    # Equation (4.47).
                    matrix[source_cell.index, target_cell.index] += min(mass_flow, 0.0)
                    matrix[source_cell.index, source_cell.index] += max(mass_flow, 0.0)
                case _:
                    raise ValueError(f"Scheme '{convection_scheme}' isn't available for the convective term.")
            # Looping over all diffusion schemes.
            match diffusion_scheme:
                case 'central_difference':
                    # Equation (4.50).
                    diffusion_matrix_coefficient = diffusion * mesh.es[surface]['surface_area'] / distance_between_cells
                    matrix[source_cell.index, target_cell.index] -= diffusion_matrix_coefficient
                    matrix[source_cell.index, source_cell.index] += diffusion_matrix_coefficient
                case _:
                    raise ValueError(f"Scheme '{diffusion_scheme}' isn't available for the diffusion term.")

    # Looping over all boundaries.
    for i, boundary_indices in enumerate(mesh['boundaries_indices']):
        # Looping over all surfaces in each boundary.
        for j, boundary_index in enumerate(boundary_indices):
            surface_normal_vector = mesh['boundaries_spatial_coordinates'][i][j] - mesh.vs['spatial_coordinates'][boundary_index]
            distance_between_cell_and_boundary = np.linalg.norm(surface_normal_vector)
            surface_normal_velocity = np.dot(velocity_vector, surface_normal_vector / distance_between_cell_and_boundary)
            # Calculates the mass flow normal to the surface.
            mass_flow = density * surface_normal_velocity * mesh['boundary_surface_area'][i][j]
            match boundary_conditions['type'][i]:
                case 'constant':
                    match convection_scheme:
                        case 'central_difference':
                            vector[boundary_index] -= mass_flow * boundary_conditions['value'][i]
                        case 'upwind':
                            vector[boundary_index] -= min(mass_flow, 0.0) * boundary_conditions['value'][i]
                            matrix[boundary_index, boundary_index] += max(mass_flow, 0.0)
                        case _:
                            pass
                    match diffusion_scheme:
                        case 'central_difference':
                            diffusion_coefficient = diffusion * mesh['boundary_surface_area'][i][j] / distance_between_cell_and_boundary 
                            vector[boundary_index] += diffusion_coefficient * boundary_conditions['value'][i]
                            matrix[boundary_index, boundary_index] += diffusion_coefficient
                        case _:
                            pass
                case 'symmetry':
                    match convection_scheme:
                        case 'central_difference':
                            matrix[boundary_index, boundary_index] += mass_flow
                        case 'upwind':
                            matrix[boundary_index, boundary_index] += mass_flow
                        case _:
                            pass
                    match diffusion_scheme:
                        case 'central_difference':
                            pass
                        case _:
                            pass
                case _:
                    raise ValueError(f"Boundary condtion '{boundary_conditions['type'][i]}' isn't available.")

    return matrix, vector

# Examples.
if __name__ == "__main__":
    from Meshes import create_lattice_mesh
    import matplotlib.pyplot as plt
    
    """ # Creates the mesh.
    mesh = create_lattice_mesh(
        dimensions=(50, 50),
        grid_size=1
    )

    # Calculates the equation coefficients.
    matrix, vector = get_transport_equation_coefficients(
        convection_scheme='upwind',
        diffusion_scheme='central_difference',
        density=1,
        diffusion=100,
        velocity_vector=(0, 1),
        mesh=mesh,
        boundary_conditions={'type': ('constant', 'symmetry', 'symmetry', 'constant'), 'value': (0, None, None, 1)}
    )

    mesh.vs['transport_quantity'] = np.linalg.inv(matrix) @ vector

    coordinates = np.array(mesh.vs['spatial_coordinates'])
    plt.scatter(coordinates[:, 0], coordinates[:, 1], marker='s', c=mesh.vs['transport_quantity'])
    plt.colorbar()
    plt.show() """

    N = 5
    L = 1
    Pe = 100

    # Creates the mesh.
    mesh = create_lattice_mesh(
        dimensions=(N,), 
        grid_size=L / N,
    )
    
    # Calculate the A-matrix.
    matrix, vector = get_transport_equation_coefficients(
        convection_scheme='upwind', 
        diffusion_scheme='central_difference', 
        density=1, 
        diffusion=1, 
        velocity_vector=(Pe,), 
        mesh=mesh,
        boundary_conditions={'type': ('constant', 'constant'), 'value': (0, 1)}
    )

    plt.plot(range(N), (np.exp(Pe*np.linspace(0, L, N)/L)-1)/(np.exp(Pe)-1), label='Exact')
    plt.plot(range(N), np.linalg.inv(matrix) @ vector, label='CFD', linestyle='--')
    plt.legend()
    plt.show()