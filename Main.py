from unstructured_poisson_solver import UnstructuredPoissonSolver
from Parameters import options
import numpy as np
from Utils import setup_logging



options.setdefault("momentOrigin", [0, 0, 0])

Logger = setup_logging(options["verbose"])

# Initialize with CGNS file

solver = UnstructuredPoissonSolver(options, Logger)



solver.set_num_threads(4)

# Read unstructured grid
solver.read_cgns_unstructured(options)

# Set boundary conditions by name


solver.set_VolumeAndBoundaryConditions(options)


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