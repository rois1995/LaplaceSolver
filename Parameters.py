import numpy as np

# GridName= "IsolatedCube_Unstr.cgns"
GridName= "BiggerCircle_Unstruct.cgns"
nDim= 2

exactSolution= "Parabolic_2D"
caseName = "BiggerCircle_Unstruct_"+exactSolution + "_CVSolution"

def vol_condition(x, y, z, typeOfExactSolution="None"):
    if typeOfExactSolution == "Cosine_3D":
        return -12 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y) * np.cos(2*np.pi*z)
    elif typeOfExactSolution == "Parabolic_3D" or typeOfExactSolution == "Parabolic_2D":
        return 1
    elif typeOfExactSolution == "Cosine_2D":
        return -4 * (np.pi**2) * np.cos(2*np.pi*x) * np.cos(2*np.pi*y)
    elif typeOfExactSolution == "None":
        return 0
    else:
        print("ERROR! Unknown Exact Solution!")
        exit(1)

def exact_dirichlet_boundary_condition(x, y, z, typeOfExactSolution="None"):
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

BoundaryConditions= { 'Farfield': {'Elem_type': 'line', 'BCType': 'Dirichlet', 'Value': exact_dirichlet_boundary_condition, 'typeOfExactSolution': exactSolution }}
VolumeCondition= {'Value': vol_condition, 'typeOfExactSolution': exactSolution}
# BlockName= "blk-1"
BlockName= "dom-1"

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


solverName= "fgmres"
solverOptions= {}

UseApproximateLaplacianFormulation= False

options = {
            'GridName': GridName,
            'nDim'    : nDim,
            'caseName': caseName,
            'BlockName': BlockName,
            'BoundaryConditions': BoundaryConditions,
            'VolumeCondition': VolumeCondition, 
            'solverName': solverName, 
            'solverOptions': solverOptions,
            'UseApproximateLaplacianFormulation': UseApproximateLaplacianFormulation
          }