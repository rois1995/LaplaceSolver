import numpy as np

verbose=False

GridName= "Mesh.cgns"
nDim= 2
BlockName= "dom-1"

debug=1

exactSolution= "Cosine_2D"
caseName = exactSolution + "_CVSolution_Neumann"

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


BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Neumann', 'Value': neumann_boundary_condition, 'typeOfExactSolution': exactSolution }}

VolumeCondition= {'Value': vol_condition, 'typeOfExactSolution': exactSolution}

solverName= "gmres"
solverOptions= {
                'maxiter':20000
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
            'exactSolutionFun': dirichlet_boundary_condition
          }