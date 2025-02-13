import igraph as ig
import numpy as np
import numpy.typing as npt

def create_lattice_mesh(dimensions: tuple[int, ...], grid_size: float) -> ig.Graph:
    """
    Creates an iso-lattice mesh with n dimensions.

    Parameters
    ----------
    dimensions : tuple of int
        The size of the lattice for each dimension in n.
    grid_size : float
        The distance between each cell.
    """
    # Creates the base lattice.
    mesh: ig.Graph = ig.Graph.Lattice(dim=dimensions, circular=False)
    
    # Assigns the surface area attribute to each edge.
    mesh.es['surface_area'] = (grid_size**(len(dimensions)-1),)*mesh.ecount()
    
    # Initializes the spatial coordinates matrix.
    spatial_coordinates = np.empty((mesh.vcount(), len(dimensions)))
    
    # Determines the spatial coordinates for each vertex.
    grid_dimensions_coordinates = np.meshgrid(*(np.linspace(grid_size/2, i*grid_size-grid_size/2, i) for i in dimensions))
    
    # Assigns the spatial coordinates attribute to each vertex.
    for i, grid_dimension_coordinates in enumerate(grid_dimensions_coordinates):
        spatial_coordinates[:, i] = grid_dimension_coordinates.flatten()
    mesh.vs['spatial_coordinates'] = spatial_coordinates
    
    # Determines which mesh indices to assign a boundary condition to.
    index_lattice: npt.NDArray = np.arange(mesh.vcount()).reshape(np.flip(dimensions))
    low_index_boundaries = np.full((len(dimensions),)*2, slice(0, None))
    high_index_boundaries = np.copy(low_index_boundaries)
    np.fill_diagonal(low_index_boundaries, slice(0, 1))
    np.fill_diagonal(high_index_boundaries, slice(-1, None))
    boundaries_indices_slices = np.concatenate((low_index_boundaries, high_index_boundaries))

    # Appends each boundary's indices to a global list.
    mesh['boundaries_indices'] = list()
    for i, boundary_indices_slices in enumerate(boundaries_indices_slices):
        mesh['boundaries_indices'].append(index_lattice[*boundary_indices_slices].flatten())
    
    # Calculates the normal vector for each boundary.    
    boundary_normal_vectors = np.flip(np.concatenate((-np.eye(len(dimensions), dtype=np.int8), 
                                                       np.eye(len(dimensions), dtype=np.int8))), axis=1)
    
    # Appends each boundary surface center coordinate to a global list.
    mesh['boundaries_spatial_coordinates'] = list()
    mesh['boundary_surface_area'] = list()
    for i, boundary_indices in enumerate(mesh['boundaries_indices']):
        mesh['boundaries_spatial_coordinates'].append(np.empty((len(boundary_indices), len(dimensions))))
        mesh['boundary_surface_area'].append(np.full(len(boundary_indices), grid_size**(len(dimensions)-1)))
        for j, boundary_index in enumerate(boundary_indices):
            mesh['boundaries_spatial_coordinates'][i][j] = mesh.vs['spatial_coordinates'][boundary_index] + 1/2 * grid_size * boundary_normal_vectors[i]



    return mesh

# Example.
if __name__ == "__main__":

    # Creates the mesh.
    mesh = create_lattice_mesh(
        dimensions=(4,), 
        grid_size=5,
    )

    print(mesh['boundaries_indices'])
    print(mesh['boundaries_spatial_coordinates'])
    print(mesh['boundary_surface_area'])

    
    # Plots the mesh.
    import matplotlib.pyplot as plt
    ig.config['plotting.backend'] = 'matplotlib'
    ig.plot(
        mesh, 
        vertex_label=np.arange(mesh.vcount()),
        # vertex_label=mesh.vs['spatial_coordinates'],
        edge_label=mesh.es['surface_area']
    )
    plt.show()