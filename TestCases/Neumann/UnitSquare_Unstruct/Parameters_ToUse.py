import numpy as np
from numba import njit

BC_TYPE_MAP = {
    'Dirichlet': 0,
    'Neumann': 1
}


EXACT_SOLUTION_MAP = {
    'Cosine_3D': 0,
    'Cosine_2D': 1,
    'Parabolic_3D': 2,
    'Parabolic_2D': 3,
    'None': -100,
    'Normal': -1,
    'Zero': -2,
}

FORCE_OR_MOMENT_MAP = {
    'Force' : 0,
    'Moment': 1
}

verbose=True

GridName= "Mesh.cgns"
nDim= 2
BlockName= "dom-1"

momentOrigin=[0.5, 0.5, 0.0]

debug=1

exactSolution= "Cosine_2D"
caseName = GridName.split(".")[0]+"_"+exactSolution + "_CVSolution"

@njit
def vol_condition(x, y, z, typeOfExactSolution=-1):
    if typeOfExactSolution == 0:
        return -12 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
    elif typeOfExactSolution == 2 or typeOfExactSolution == 3:
        return np.ones((len(x), ), dtype=float)
    elif typeOfExactSolution == 1:
        return -8 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    elif typeOfExactSolution == -100:
        return np.zeros((len(x), ), dtype=float)
    else:
        print("ERROR! Unknown Exact Solution!")

@njit
def dirichlet_boundary_condition(x, y, z, typeOfExactSolution=-1):
    if typeOfExactSolution == 0:
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
    elif typeOfExactSolution == 2:
        return (x**2+y**2+z**2)/6
    elif typeOfExactSolution == 1:
        return np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    elif typeOfExactSolution == 3:
        return (x**2+y**2)/4
    else:
        print("ERROR! Unknown Exact Solution!")

@njit
def neumann_boundary_condition(x, y, z, normal, momentOrigin, forceOrMoment=0, component_idx=0, typeOfExactSolution=-1):

    BC = np.zeros((3), dtype=np.float64)
    if typeOfExactSolution == 0:
        BC[0] = -2*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
        BC[1] = -2*np.pi*np.cos(2*np.pi*x) * np.sin(2*np.pi*y) * np.cos(2*np.pi*z)
        BC[2] = -2*np.pi*np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.sin(2*np.pi*z)
        return np.sum(BC*normal)
    elif typeOfExactSolution == 2:
        BC[0] = 2*x/6
        BC[1] = 2*y/6
        BC[2] = 2*z/6
        return np.sum(BC*normal)
    elif typeOfExactSolution == 1:
        BC[0] = -2*np.pi*np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
        BC[1] = -2*np.pi*np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
        return np.sum(BC*normal)
    elif typeOfExactSolution == 3:
        BC[0] = 2*x/4
        BC[1] = 2*y/4
        return np.sum(BC*normal)
    elif typeOfExactSolution == -1:  # normal BC, choose between Force and Moment
        if forceOrMoment == 0:
            return normal[component_idx]
        else:
            return np.cross(np.array([x, y, z], dtype=np.float64)-momentOrigin, normal)[component_idx]
    elif typeOfExactSolution == -2:  # normal BC, set as 0
        return 0.0
    else:
        print("ERROR! Unknown Exact Solution!")


    


BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Neumann', 'Value': neumann_boundary_condition, 'typeOfExactSolution': exactSolution }}

VolumeCondition= {'Value': vol_condition, 'typeOfExactSolution': exactSolution}



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