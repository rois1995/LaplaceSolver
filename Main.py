from unstructured_poisson_solver import UnstructuredPoissonSolver
from Parameters import options
import numpy as np
from Utils import setup_logging

options.setdefault("momentOrigin", [0, 0, 0])

Logger = setup_logging(options["verbose"])

# Initialize with CGNS file
# solver = UnstructuredPoissonSolver("VeryCoarse_WithVoxel_OneBoundary.cgns")
# solver = UnstructuredPoissonSolver("Cube.cgns")
# solver = UnstructuredPoissonSolver("Cube_2.cgns")
# solver = UnstructuredPoissonSolver("Sphere_Struct.cgns")
# solver = UnstructuredPoissonSolver("Sphere_Medium.cgns")
# solver = UnstructuredPoissonSolver("Sphere.cgns")
# solver = UnstructuredPoissonSolver("SmallSphere.cgns")
# solver = UnstructuredPoissonSolver("IsolatedCube.cgns")
solver = UnstructuredPoissonSolver(options, Logger)



solver.set_num_threads(4)

# Read unstructured grid
# solver.read_cgns_unstructured(zone_name = 'blk-3', BCs = {'quad_Body':'quad', 'tri_Body':'tri', 'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Top':'quad', 'Bot':'quad', 'PX':'quad', 'MX':'quad', 'PY':'quad', 'MY':'quad', 'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Sphere':'tri', 'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Farfield':'tri'})
# solver.read_cgns_unstructured(zone_name = 'blk-1', BCs = {'Farfield':'quad'})
solver.read_cgns_unstructured(options)

# Set boundary conditions by name


solver.set_VolumeAndBoundaryConditions(options)

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


# Or solve all three components
forceComponents= [0, 1]
if options["nDim"] == 3:
    forceComponents += [2]
if not (options["exactSolution"] == "None"):
    forceComponents = [0]


momentComponents= [2]
if options["nDim"] == 3:
    momentComponents = [0, 1, 2]
if not (options["exactSolution"] == "None"):
    momentComponents = []

solutions = solver.solve_components(forceComponents=forceComponents, momentComponents=momentComponents, solver=options["solverName"], useReordering=False, solverOptions=options["solverOptions"])


# Export to VTK for visualization in ParaView
solver.export_solution_vtk(solutions, "solution_"+options["solverName"]+"_"+options["caseName"]+".vtu")