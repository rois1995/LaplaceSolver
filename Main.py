from unstructured_poisson_solver import UnstructuredPoissonSolver
import numpy as np

# Initialize with CGNS file
# solver = UnstructuredPoissonSolver("VeryCoarse_WithVoxel_OneBoundary.cgns")
# solver = UnstructuredPoissonSolver("Cube.cgns")
# solver = UnstructuredPoissonSolver("Cube_2.cgns")
# solver = UnstructuredPoissonSolver("Sphere_Struct.cgns")
# solver = UnstructuredPoissonSolver("Sphere_Medium.cgns")
# solver = UnstructuredPoissonSolver("Sphere.cgns")
# solver = UnstructuredPoissonSolver("SmallSphere.cgns")
# solver = UnstructuredPoissonSolver("IsolatedCube.cgns")
solver = UnstructuredPoissonSolver("IsolatedCube_Unstr.cgns")

casename="Coarse_Unstr_Cosine"

solver.set_num_threads(4)

# Read unstructured grid
# solver.read_cgns_unstructured(zone_name = 'blk-3', BCs = {'quad_Body':'quad', 'tri_Body':'tri', 'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Top':'quad', 'Bot':'quad', 'PX':'quad', 'MX':'quad', 'PY':'quad', 'MY':'quad', 'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Sphere':'tri', 'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Farfield':'quad'})
solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Farfield':'tri'})

# Set boundary conditions by name

def vol_condition(x, y, z, exactSolution=False):
    if exactSolution:
        return -12 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
    else:
        return 0

def exact_dirichlet_boundary_condition(x, y, z):
    return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)


# def vol_condition(x, y, z, exactSolution=False):
#     if exactSolution:
#         return 1
#     else:
#         return 0

# def exact_dirichlet_boundary_condition(x, y, z):
#     return (x**2+y**2+z**2)/6

solver.set_volume_condition(vol_condition, exactSolution=True)
# solver.set_boundary_condition("Farfield", "neumann", 0.0)
solver.set_boundary_condition("Farfield", "dirichlet", exact_dirichlet_boundary_condition)
# solver.set_boundary_condition("quad_Farfield", "neumann", 0.0)
# solver.set_boundary_condition("tri_Body", "neumann", 'normal')
# solver.set_boundary_condition("quad_Body", "neumann", 'normal')
# solver.set_boundary_condition("Top", "neumann", 'normal')
# solver.set_boundary_condition("Bot", "neumann", 'normal')
# solver.set_boundary_condition("PX", "neumann", 'normal')
# solver.set_boundary_condition("PY", "neumann", 'normal')
# solver.set_boundary_condition("MX", "neumann", 'normal')
# solver.set_boundary_condition("MY", "neumann", 'normal')
# solver.set_boundary_condition("Sphere", "neumann", 'normal')
# solver.set_boundary_condition("Sphere", "dirichlet", exact_dirichlet_boundary_condition)


solver.construct_secondaryMeshStructures()

# Compute geometric properties
# solver.compute_geometric_properties()

# Solve for component i
# phi_1 = solver.solve_poisson(component_idx=0)

solverName='gmres'
gmresOptions= {
                'tol':1e-6, 
                'maxiter':1000, 
                'restart':None, 
                'verbose':True, 
                'use_ilu':True
}

# solverName='fgmres'
# fgmresOptions= {
#                 'tol':1e-6, 
#                 'maxiter':1000, 
#                 'restart':20, 
#                 'verbose':True, 
#                 'use_ilu':True
# }

# solverName='bicgstab'
# bicgstabOptions= {
#                 'tol':1e-10, 
#                 'maxiter':20000, 
#                 'verbose':True, 
#                 'use_ilu':True
# }

# solverName='minres'
# minresOptions= {
#                 'tol':1e-10, 
#                 'maxiter':20000, 
#                 'restart':None, 
#                 'verbose':True, 
#                 'use_ilu':True
# }

# solverName='spsolve'

# Or solve all three components
# solutions = solver.solve_components(components=[0, 1, 2], solver=solverName, useReordering=True, solverOptions=fgmresOptions)
solutions = solver.solve_components(components=[0, 1, 2], solver=solverName, useReordering=False, solverOptions=gmresOptions)
# solutions = solver.solve_components(components=[0, 1, 2], solver=solverName, useReordering=True, solverOptions=bicgstabOptions)
# solutions = solver.solve_components(components=[0, 1, 2], solver=solverName, useReordering=True, solverOptions=minresOptions)
# solutions = solver.solve_components(components=[0], solver=solverName)


# Export to VTK for visualization in ParaView
solver.export_solution_vtk(solutions, "solution_"+solverName+"_"+casename+".vtu")