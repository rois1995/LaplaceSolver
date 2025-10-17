import numpy as np

verbose=True

# GridName= "IsolatedCube_Unstr.cgns"
GridName= "Mesh.cgns"
nDim= 3
BlockName= "blk-1"
# BlockName= "dom-1"

momentOrigin=[0.5, 0.5, 0.0]

debug=1

exactSolution= "Cosine_3D"
caseName = GridName.split(".")[0]+"_"+exactSolution + "_CVSolution"

def vol_condition(x, y, z, typeOfExactSolution="None"):
    if typeOfExactSolution == "Cosine_3D":
        return -12 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
    elif typeOfExactSolution == "Parabolic_3D" or typeOfExactSolution == "Parabolic_2D":
        return 1
    elif typeOfExactSolution == "Cosine_2D":
        return -8 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    elif typeOfExactSolution == "None":
        return 0
    else:
        print("ERROR! Unknown Exact Solution!")
        exit(1)

def dirichlet_boundary_condition(x, y, z, typeOfExactSolution="None"):
    if typeOfExactSolution == "Cosine_3D":
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
    elif typeOfExactSolution == "Parabolic_3D":
        return (x**2+y**2+z**2)/6
    elif typeOfExactSolution == "Cosine_2D":
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    elif typeOfExactSolution == "Parabolic_2D":
        return (x**2+y**2)/4
    else:
        print("ERROR! Unknown Exact Solution!")
        exit(1)

def neumann_boundary_condition(x, y, z, typeOfExactSolution="None"):
    if typeOfExactSolution == "Cosine_3D":
        return np.array([-2*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z),
                         -2*np.pi*np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.cos(2*np.pi*z),
                         -2*np.pi*np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z)])
    elif typeOfExactSolution == "Parabolic_3D":
        return np.array([2*x/6,
                         2*y/6,
                         2*z/6])
    elif typeOfExactSolution == "Cosine_2D":
        return np.array([-2*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y),
                         -2*np.pi*np.cos(2*np.pi*x) * np.sin(2*np.pi*y),
                         0])
    elif typeOfExactSolution == "Parabolic_2D":
        return np.array([2*x/4,
                         2*y/4, 
                         0])
    else:
        print("ERROR! Unknown Exact Solution!")
        exit(1)


# BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}
# BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}
# BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Neumann', 'Value': 0, 'typeOfExactSolution': exactSolution },
#                       'Wall': {'Elem_type': 'line', 'BCType': 'Neumann', 'Value': 'Normal', 'typeOfExactSolution': exactSolution }}

# BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Neumann', 'Value': neumann_boundary_condition, 'typeOfExactSolution': exactSolution }}
# BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}
BoundaryConditions= { 'Farfield': {'Elem_type': 'quad', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}
# BoundaryConditions= { 'Farfield': {'Elem_type': 'tri', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}
# BoundaryConditions= { 'tri_Farfield': {'Elem_type': 'tri', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution },
#                       'quad_Farfield': {'Elem_type': 'quad', 'BCType': 'Dirichlet', 'Value': dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}

VolumeCondition= {'Value': vol_condition, 'typeOfExactSolution': exactSolution}

# solverName='gmres'
# gmresOptions= {
#                 'tol':1e-6, 
#                 'maxiter':1000, 
#                 'restart':None, 
#                 'verbose':True, 
#                 'use_ilu':True
# }

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


solverName= "bicgstab"
solverOptions= {
                'maxiter':10000,
                'fill_factor': 2
                }

options = {
            'verbose' : verbose,
            'GridName': GridName,
            'nDim'    : nDim,
            'caseName': caseName,
            'BlockName': BlockName,
            'BoundaryConditions': BoundaryConditions,
            'VolumeCondition': VolumeCondition, 
            'solverName': solverName, 
            'solverOptions': solverOptions,
            'exactSolution': exactSolution,
            'debug': debug,
            'exactSolutionFun': dirichlet_boundary_condition,
            'momentOrigin': momentOrigin
          }