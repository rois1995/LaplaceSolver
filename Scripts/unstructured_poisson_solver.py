"""
Unstructured Poisson Equation Solver with CGNS Grid Support
Solves ∇²φᵢ = 0 in domain Vf with Neumann boundary conditions
Supports mixed element types: tetrahedra, hexahedra, pyramids, prisms

Compatible with NumPy 2.0+
"""
import sys
Path2Scripts="./"
sys.path.insert(1, Path2Scripts)
import numpy as np
import os
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import bicgstab, cg, gmres, minres, spilu, LinearOperator
import pyamg
from scipy.sparse.csgraph import reverse_cuthill_mckee
from collections import defaultdict
import warnings
from Mesh import MeshClass
import time
from multiprocessing import Pool
from Parameters import BC_TYPE_MAP, EXACT_SOLUTION_MAP, FORCE_OR_MOMENT_MAP
import math
import h5py

os.environ['NUMBA_NUM_THREADS'] = '48'

from numba import njit, prange
import numba



@njit(parallel=True)
def build_CV_fv_system_NumbaParallel_Ext(n_nodes, Nodes, NodesConnectedToNode, 
                                     ControlFaceDictPerEdge, ControlFaceNormalDictPerEdge,
                                     ControlVolumesPerNode, NumOfNodesConnectedToNode):
    """
    Build finite volume system for Poisson equation using node-based approach

    Parameters:
    -----------
    component_idx : int
        Component index (1, 2, or 3)

    Returns:
    --------
    A : sparse matrix
        System matrix
    b : array
        RHS vector
    """

    N = n_nodes  # Solve at nodes

    node_neighbors = NodesConnectedToNode
    ControlFaceDictPerEdge = ControlFaceDictPerEdge
    # if self.nDim == 2:
    ControlFaceNormalDictPerEdge = ControlFaceNormalDictPerEdge
    ControlVolumesPerNode = ControlVolumesPerNode

    total_entries = np.sum(NumOfNodesConnectedToNode) + n_nodes
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        offsets[i + 1] = offsets[i] + NumOfNodesConnectedToNode[i]+1

    rows = np.zeros(total_entries, dtype=np.int32)
    cols = np.zeros(total_entries, dtype=np.int32)
    data = np.zeros(total_entries, dtype=np.float64)

    # Build Laplacian matrix
    for iNode in prange(N):

        start_idx = offsets[iNode]
        
        neighbors = node_neighbors[iNode]
        nNeigh = NumOfNodesConnectedToNode[iNode]

        controlVolume = ControlVolumesPerNode[iNode]

        # Interior node: standard Laplacian
        # Approximate: ∇²φ ≈ Σ(φⱼ - φᵢ) / dᵢⱼ²
        diag_val = 0.0
        for jthNode in range(nNeigh):
            jNode = neighbors[jthNode]
            distVec = Nodes[jNode] - Nodes[iNode]
            dist = np.linalg.norm(distVec)
            controlAreaIJ = ControlFaceDictPerEdge[iNode, jthNode]
            faceNormalIJ = ControlFaceNormalDictPerEdge[iNode, jthNode]
            proj = np.sum(distVec*faceNormalIJ)/dist
            coeff = controlAreaIJ * proj/ (dist*controlVolume)

            if np.isinf(coeff):
                print(f"point {iNode} connected to point {jNode} has coeff = infs. Other values are controlVolume = {controlVolume}, dist = {dist}, controlAreaIJ = {controlAreaIJ}, faceNormalIJ = {faceNormalIJ}")
            
            rows[start_idx+jthNode] = iNode
            cols[start_idx+jthNode] = jNode
            data[start_idx+jthNode] = coeff

            diag_val -= coeff


        
        rows[start_idx+jthNode+1] = iNode
        cols[start_idx+jthNode+1] = iNode
        data[start_idx+jthNode+1] = diag_val

    return rows, cols, data
class UnstructuredPoissonSolver:
    """
    Finite volume/element Poisson solver for unstructured CGNS grids
    Supports tetrahedra, hexahedra, pyramids, and prisms
    """

    def __init__(self, options, Logger):
        """
        Initialize solver

        Parameters:
        -----------
        cgns_file : str, optional
            Path to CGNS file
        """
        self.GridFileName= options["GridName"]
        self.boundary_conditions = {}
        self.volume_condition = 0.0
        self.exactSolution = EXACT_SOLUTION_MAP.get(options["exactSolution"], -1)
        self.nDim = options["nDim"]
        self.Mesh = MeshClass(Logger, options["GridName"], options["nDim"])
        self.allNeumann = False
        self.debug = options["debug"]
        self.Logger = Logger
        self.exactSolutionFun = options["exactSolutionFun"]
        self.momentOrigin = np.array(options["momentOrigin"])

    def read_cgns_unstructured(self, options):
        """
        Read unstructured CGNS grid with mixed element types
        Compatible with NumPy 2.0
        """
        self.Logger.info(f"Reading CGNS file: {self.GridFileName}")

        startTime = time.time()

        try:
            self.Mesh._read_cgns_h5py(options["BlockName"], options["BoundaryConditions"])
        except Exception as e:
            self.Logger.error(f"h5py read failed: {e}. Failed to read CGNS file!")
            raise 

        self.Logger.info(f"Finished reading grid. Elapsed time {time.time()-startTime} s.")
        
    def construct_secondaryMeshStructures(self):
        
        self.Logger.info(f"Constructing secondary mesh structures...")

        self.Mesh.construct_secondaryMeshStructures(self.boundary_conditions)

    def set_VolumeAndBoundaryConditions(self, options):

        self.set_volume_condition(options["VolumeCondition"])
        self.set_boundary_conditions(options["BoundaryConditions"])

    


    def set_boundary_conditions(self, BC):
        """
        Set boundary condition by boundary name

        Parameters:
        -----------
        boundary_name : str
            Name of boundary region (e.g., 'farfield', 'wall', etc.)
        bc_type : str
            Type: 'neumann' or 'dirichlet'
        bc_value : float or callable
            BC value or function f(x, y, z) -> value

        Examples:
        ---------
        solver.set_boundary_condition("farfield", "neumann", 0.0)
        solver.set_boundary_condition("wall", "neumann", 1.0)
        solver.set_boundary_condition("inlet", "neumann", lambda pos: pos[0])
        """

        for bc_name in BC.keys():

            if bc_name not in self.Mesh.boundary_nodes and bc_name not in self.Mesh.boundary_faces:
                self.Logger.info(f"Warning: Boundary '{bc_name}' not found in grid")
                self.Logger.info(f"Available boundaries: {list(set(list(self.Mesh.boundary_nodes.keys()) + list(self.Mesh.boundary_faces.keys())))}")
                return

            self.boundary_conditions[bc_name] = {
                'BCType': BC_TYPE_MAP.get(BC[bc_name]['BCType'], -1),
                'Value': BC[bc_name]['Value'],
                'typeOfExactSolution': EXACT_SOLUTION_MAP.get(BC[bc_name]['typeOfExactSolution'], -1)
            }
            self.Logger.info(f"Set BC on '{bc_name}': {BC[bc_name]['BCType']} = {BC[bc_name]['Value']}, typeOfExactSolution = {BC[bc_name]['typeOfExactSolution']}")

    def set_volume_condition(self, volume_BC):
        """
        Set function for point-in-volume evaluation

        Parameters:
        -----------
        boundary_name : str
            Name of boundary region (e.g., 'farfield', 'wall', etc.)
        bc_type : str
            Type: 'neumann' or 'dirichlet'
        bc_value : float or callable
            BC value or function f(x, y, z) -> value

        Examples:
        ---------
        solver.set_boundary_condition("farfield", "neumann", 0.0)
        solver.set_boundary_condition("wall", "neumann", 1.0)
        solver.set_boundary_condition("inlet", "neumann", lambda pos: pos[0])
        """

        self.volume_condition = {
                'Value': volume_BC['Value'],
                'typeOfExactSolution': EXACT_SOLUTION_MAP.get(volume_BC['typeOfExactSolution'], -1)
            }


    def build_CV_fv_system(self, solversOptions):
        """
        Build finite volume system for Poisson equation using node-based approach

        Parameters:
        -----------
        component_idx : int
            Component index (1, 2, or 3)

        Returns:
        --------
        A : sparse matrix
            System matrix
        b : array
            RHS vector
        """
        increaseSize = self.allNeumann and not solversOptions["solveParallel"]

        # Specify the number of cores to use
        N = self.Mesh.n_nodes  # Solve at nodes
        A = lil_matrix((N+int(increaseSize), N+int(increaseSize)))
        b = np.zeros(N+int(increaseSize))

        self.Logger.info(f"Building system on interior nodes ({N} nodes)...")

        node_neighbors = self.Mesh.NodesConnectedToNode
        ControlFaceDictPerEdge = self.Mesh.ControlFaceDictPerEdge
        # if self.nDim == 2:
        ControlFaceNormalDictPerEdge = self.Mesh.ControlFaceNormalDictPerEdge
        ControlVolumesPerNode = self.Mesh.ControlVolumesPerNode
        
        exactSol = self.exactSolutionFun(self.Mesh.Nodes[:, 0], self.Mesh.Nodes[:, 1], self.Mesh.Nodes[:, 2], self.exactSolution)
        lapl_num = np.zeros(N)

        # Build Laplacian matrix
        for iNode in range(N):
            if iNode%10000 == 0:
                self.Logger.info(f"Assembling matrix at point: {iNode}")
            neighbors = node_neighbors[iNode]

            controlVolume = ControlVolumesPerNode[iNode]
            nNeigh = self.Mesh.NumOfNodesConnectedToNode[iNode]

            s = 0

            if len(neighbors) == 0:
                # Isolated node
                A[iNode, iNode] = 1.0
                b[iNode] = 0.0
                continue

            # Interior node: standard Laplacian
            # Approximate: ∇²φ ≈ Σ(φⱼ - φᵢ) / dᵢⱼ²
            diag_val = 0.0
            for jthNode in range(nNeigh):
                jNode = neighbors[jthNode]
                distVec = self.Mesh.Nodes[jNode] - self.Mesh.Nodes[iNode]
                dist = np.linalg.norm(distVec)
                controlAreaIJ = ControlFaceDictPerEdge[iNode, jthNode]
                faceNormalIJ = ControlFaceNormalDictPerEdge[iNode, jthNode]
                proj = np.sum(distVec*faceNormalIJ)/dist
                coeff = controlAreaIJ * proj/ (dist*controlVolume)
                A[iNode, jNode] = coeff
                diag_val -= coeff
                s += coeff * (exactSol[jNode] - exactSol[iNode])

                if np.isinf(coeff):
                    print(f"point {iNode} connected to point {jNode} has coeff = infs. Other values are controlVolume = {controlVolume}, dist = {dist}, controlAreaIJ = {controlAreaIJ}, faceNormalIJ = {faceNormalIJ}")       
            
            lapl_num[iNode] = s

            A[iNode, iNode] = diag_val
            

        b[:self.Mesh.n_nodes] = self.volume_condition["Value"](self.Mesh.Nodes[:, 0], self.Mesh.Nodes[:, 1], self.Mesh.Nodes[:, 2], self.volume_condition["typeOfExactSolution"])  # RHS = 0 for Laplace equation

        print("lapl_num: mean, min, max:", lapl_num.mean(), lapl_num.min(), lapl_num.max())
        print("error vs exact (1): mean, max abs:", np.mean(lapl_num - 1.0), np.max(np.abs(lapl_num - 1.0)))

        return A, b
    
    
    def build_CV_fv_system_NumbaParallel(self, solversOptions):

        N = self.Mesh.n_nodes  # Solve at nodes

        self.Logger.info(f"Building system on interior nodes ({N} nodes)...")

        # max_neighbors = max(self.Mesh.NumOfNodesConnectedToNode)

        # # Create rectangular array with padding (e.g., -1 for unused entries)
        # NodesConnectedToNode_rect = np.full((len(self.Mesh.NodesConnectedToNode), max_neighbors), -1, dtype=np.int32)
        # for i, neighbors in enumerate(self.Mesh.NodesConnectedToNode):
        #     NodesConnectedToNode_rect[i, :len(list(neighbors))] = np.array(list(neighbors))

        rows, cols, data = build_CV_fv_system_NumbaParallel_Ext(self.Mesh.n_nodes, self.Mesh.Nodes, 
                                                 self.Mesh.NodesConnectedToNode, self.Mesh.ControlFaceDictPerEdge, 
                                                 self.Mesh.ControlFaceNormalDictPerEdge, self.Mesh.ControlVolumesPerNode, 
                                                 self.Mesh.NumOfNodesConnectedToNode)
        
        increaseSize = self.allNeumann and not solversOptions["solveParallel"]
        A = coo_matrix((data, (rows, cols)), shape=(self.Mesh.n_nodes+int(increaseSize), self.Mesh.n_nodes+int(increaseSize))).tolil()

        b = np.zeros(N+int(increaseSize))
        b[:self.Mesh.n_nodes] = self.volume_condition["Value"](self.Mesh.Nodes[:, 0], self.Mesh.Nodes[:, 1], self.Mesh.Nodes[:, 2], self.volume_condition["typeOfExactSolution"])  # RHS = 0 for Laplace equation
        
        return A, b
    

    def apply_CV_BCs(self, A, b, component_idx=1, ForceOrMoment=0):
        """
        Build finite volume system for Poisson equation using node-based approach

        Parameters:
        -----------
        component_idx : int
            Component index (1, 2, or 3)

        Returns:
        --------
        A : sparse matrix
            System matrix
        b : array
            RHS vector
        """
        N = self.Mesh.n_nodes  # Solve at nodes

        self.Logger.info(f"Applying Boundary Conditions for component {component_idx} ({len(np.where(self.Mesh.isNodeOnBoundary == True)[0])} nodes)...")

        ControlVolumesPerNode = self.Mesh.ControlVolumesPerNode
        NodeOfNodes = self.Mesh.NodesConnectedToNode
        NumOfNodeOfNodes = self.Mesh.NumOfNodesConnectedToNode

        actualPoint = 0
        domain_center = np.mean(self.Mesh.Nodes, axis=0)

        extFlux= 0.0
        intFlux = np.sum(b[:N]*ControlVolumesPerNode)

        totBoundaryArea = 0.0
        totBoundaryNormal = np.zeros(3)
        totBoundaryNormalTimesArea = np.zeros(3)


        DirichletNodes = np.full((N, 1), False)

        self.Logger.info(f"N Of Points on boundary = {(np.where(self.Mesh.isNodeOnBoundary)[0]).size}")

        # Build Laplacian matrix
        for iBoundNode in np.where(self.Mesh.isNodeOnBoundary)[0]:

            if actualPoint%1000 == 0:
                self.Logger.info(f"Imposing boundary conditions at point: {iBoundNode}")

            bc_names = self.Mesh.BCsAssociatedToNode[iBoundNode]
            for iBC in range(len(bc_names)):

                bc_data = self.boundary_conditions[bc_names[iBC]]

                if bc_data['BCType'] == 1:  # Neumann BC

                    bc_nodes = self.Mesh.boundary_nodes[bc_names[iBC]]
                    bnodes = bc_nodes['connectivity']
                    CellNormals = bc_nodes['normals']
                    CVAreas = bc_nodes['BoundaryCVArea']
                    CVCentroids = bc_nodes['BoundaryCVCentroids']
                    # print(areas)
                    
                    # Find all occurrencies of this node in the connectivity
                    iElem, jPoint = np.where(bnodes == iBoundNode)

                    actualNormal = np.zeros(3, dtype=np.float64)

                    for (elem, point) in zip(iElem, jPoint):

                        BoundaryCVArea = CVAreas[elem, point]
                        Centroid = CVCentroids[elem, point]
                        bNormal = CellNormals[elem, point]

                        totBoundaryArea+=BoundaryCVArea
                        totBoundaryNormal+=bNormal
                        totBoundaryNormalTimesArea+=bNormal*BoundaryCVArea

                        bc_val = bc_data['Value'](Centroid[0], Centroid[1], Centroid[2], -bNormal, self.momentOrigin, ForceOrMoment, component_idx, bc_data["typeOfExactSolution"])
                        
                        b[iBoundNode] -= bc_val*BoundaryCVArea/ControlVolumesPerNode[iBoundNode]
                        extFlux += bc_val*BoundaryCVArea

                        # Neumann BC: ∂φ/∂n = bc_value
                        # Approximate using one-sided difference

                        

                elif bc_data['BCType'] == 0: # Dirichlet BC
                    # Dirichlet BC: φ = bc_value
                    DirichletNodes[iBoundNode] = True
                    iBC = len(bc_names)+1 # Exit from the loop since it is a Dirichlet BC and I do not care anymore
                    
            actualPoint+= 1
        
        if len(np.where(DirichletNodes)[0]) > 0:
            self.Logger.info(f"Dealing with {len(np.where(DirichletNodes)[0])} Dirichlet nodes in an efficient way")
            A_csc = A.tocsc()
            bc_val = bc_data['Value'](self.Mesh.Nodes[np.where(DirichletNodes)[0], 0], self.Mesh.Nodes[np.where(DirichletNodes)[0], 1], self.Mesh.Nodes[np.where(DirichletNodes)[0], 2], bc_data["typeOfExactSolution"])
            DiricNodsIds = np.where(DirichletNodes)[0]
            for i,uD in zip(DiricNodsIds, bc_val):
                col_slice = slice(A_csc.indptr[i], A_csc.indptr[i+1])
                b -= (A_csc[:, i].toarray().ravel()) * uD
                A_csc.data[col_slice] = 0.0

            A = A_csc.tocsr()
            for i,uD in zip(DiricNodsIds, bc_val):
                row_slice = slice(A.indptr[i], A.indptr[i+1])
                A.data[row_slice] = 0.0
                A[i,i] = 1.0
                b[i] = uD

            A = A.tolil()
        
        self.Logger.info(f"AAAAAAAAAAAAAAAAaa extFlux = {extFlux}")
        self.Logger.info(f"AAAAAAAAAAAAAAAAaa intFlux = {intFlux}")
        self.Logger.info(f"AAAAAAAAAAAAAAAAaa sumFlux = {np.sum(b[:N]*ControlVolumesPerNode)}")
        self.Logger.info(f"AAAAAAAAAAAAAAAAaa total boundary area = {totBoundaryArea}")
        self.Logger.info(f"AAAAAAAAAAAAAAAAaa total boundary normal = {totBoundaryNormal}")
        self.Logger.info(f"AAAAAAAAAAAAAAAAaa total boundary area*normal = {totBoundaryNormalTimesArea}")
        # print(A.diagonal())
        return A, b


    def solve_poisson(self, A, b, solvers, useReordering=False, solversOptions={}, component_idx=1):
        """
        Solve Poisson equation for given component

        Parameters:
        -----------
        component_idx : int
            Component index (1, 2, or 3)

        Returns:
        --------
        phi : array
            Solution field at nodes
        """


        self.Logger.info("Convert matrix to CSR format")
        # Convert to CSR format
        # np.set_printoptions(threshold=sys.maxsize)
        # print(A_csr.toarray())
        # np.savetxt("Matrix.csv", A_csr.toarray(), delimiter=",")
        # np.savetxt("b.csv", b, delimiter=",")
        # A_csr.eliminate_zeros()

        if useReordering:

            self.Logger.info("Apply Reverse Cuthill-Mckee ordering")

            if self.allNeumann and not solversOptions["solveParallel"]:
                A_csr = A[:-1, :-1].tocsr()  # Convert to CSR for efficiency
                b = b[:-1]
            else:
                A_csr = A.tocsr()  # Convert to CSR for efficiency

            # Example usage in your solver (add before calling GMRES/SPSOLVE):
            perm = reverse_cuthill_mckee(A_csr, symmetric_mode=True)  # Or A_csr/b

            # Step 4: Permute matrix and RHS
            A_rcm = A_csr[perm, :][:, perm]
            b_rcm = b[perm]

            if self.allNeumann and not solversOptions["solveParallel"]:
                # Step 4: Expand back to (N+1, N+1) by adding row and column
                # Add a column of zeros on the right
                A_expanded = hstack([A_rcm, csr_matrix((A_rcm.shape[0], 1))])
                # Add a row of zeros at the bottom
                A_rcm = vstack([A_expanded, csr_matrix((1, A_expanded.shape[1]))])

                A_rcm[-1, :] = 1
                A_rcm[-1, -1] = 0
                A_rcm[:-1, -1] = 1
                b_rcm = np.append(b, 0)
                perm = np.append(perm, len(perm))

            A_toSolve = A_rcm
            b_toSolve = b_rcm

        else:


            if self.allNeumann and not solversOptions["solveParallel"]:
                A[-1, :] = 1
                A[-1, -1] = 0
                A[:-1, -1] = 1
                b[-1] = 0
            
            A_toSolve = A.tocsr()
            b_toSolve = b
            perm = None
        
        self.Logger.info(f"Matrix: {A_toSolve.shape[0]:,} unknowns, {A_toSolve.nnz:,} nonzeros")

        if solversOptions["solveParallel"]:

            self.save_system_to_hdf5(A_toSolve, b_toSolve, perm, filenameMatrix='matrix_data.h5', filenameMesh='Mesh_data.h5')

            exit(1)
        
        self.verify_null_space_removed(A_toSolve, b_toSolve)
        # b_new = self.check_compatibility(b)

        diag = A_toSolve.diagonal()
        if np.any(diag == 0):
            self.Logger.info(f"Warning: There are {len(np.where(diag==0)[0])} zeros on the diagonal!")
            self.Logger.info(f"Indexes where this happens: {np.where(diag==0)[0][0]}")

        
        # Provide non-constant initial guess
        N = A.shape[0]
        x0 = np.random.randn(N)
        x0 = x0 - np.mean(x0)  # Remove constant component

        # Solve
        self.Logger.info(f"Solving linear system...")

        for iSolver in range(len(solvers)):
            
            solver = solvers[iSolver]
            solverOptions={}
            for key in list(solversOptions.keys()):
                if not key == "solveParallel":
                    solverOptions[key] = solversOptions[key][iSolver]

            if solver == 'spsolve':
                self.Logger.info("Using spsolve function (direct solver)")
                try:
                    phi = spsolve(A_toSolve, b_toSolve)
                except Exception as e:
                    self.Logger.error(f"Direct solve failed: {e}")
                    self.Logger.info("Trying iterative solver...")
                    phi = self.gmres_solver(A_toSolve, b_toSolve, x0, solverOptions)
            elif solver == 'gmres':
                self.Logger.info("Using GMRES (iterative solver)")
                phi = self.gmres_solver(A_toSolve, b_toSolve, x0, solverOptions)
            elif solver == 'fgmres':
                self.Logger.info("Using FGMRES (iterative solver)")
                phi = self.fgmres_solver(A_toSolve, b_toSolve, x0, solverOptions)
            elif solver == 'bicgstab':
                self.Logger.info("Using BiCGSTAB (iterative solver)")
                phi = self.bicgstab_solver(A_toSolve, b_toSolve, x0, solverOptions)
            elif solver == 'minres':
                self.Logger.info("Using MinRES (iterative solver)")
                phi = self.minres_solver(A_toSolve, b_toSolve, x0, solverOptions)
            elif solver == 'cg':
                self.Logger.info("Using CG (iterative solver)")
                phi = self.cg_solver(A_toSolve, b_toSolve, x0, solverOptions)

            self.Logger.info(f"Solution range: [{phi.min():.6e}, {phi.max():.6e}]")

            # Update first guess
            x0 = phi

        if useReordering:
            phi_reordered = np.empty_like(phi)
            phi_reordered[perm] = phi
            phi = phi_reordered
            


        return phi
    

    def save_system_to_hdf5(self, A, b, perm, filenameMatrix='matrix_data.h5', filenameMesh='Mesh_data.h5'):
        """
        Save matrix, RHS, and mesh geometry to HDF5.
        
        Parameters:
        -----------
        A    : scipy.sparse matrix
               System matrix
        b    : numpy array
               RHS vector
        perm : numpy array
               Permutation vector if CutHill-McKee
        mesh_data : dict
            Dictionary containing mesh information:
            - 'nodes': Node coordinates (N x 3)
            - 'elements': Dict of element sections
            - 'control_volumes': Control volumes per node (optional)
            - 'boundary_normals': Boundary normals (optional)
        """
        self.Logger.info(f"Saving system to {filenameMatrix} and cells tp {filenameMesh}...")
        
        # Convert matrix to CSR
        n = A.shape[0]
        nnz = A.nnz

        # self.Logger.info(f"-------------------------------------")
        # self.Logger.info(f"A.shape = {A.shape}")
        # self.Logger.info(f"A.data.shape = {A.data.shape}")
        # self.Logger.info(f"A.indices.shape = {A.indices.shape}")
        # self.Logger.info(f"A.indptr.shape = {A.indptr.shape}")
        # self.Logger.info(f"A.data = {A.data}")
        # self.Logger.info(f"A.indices = {A.indices}")
        # self.Logger.info(f"A.indptr = {A.indptr}")
        # self.Logger.info(f"A.nnz = {A.nnz}")
        # self.Logger.info(f"-------------------------------------")

        start = time.time()
        
        self.Logger.info(f"  Matrix: {n:,} x {n:,}, {nnz:,} non-zeros")
        self.Logger.info(f"  Mesh nodes: {self.Mesh.n_nodes}")
        
        # Compression for efficient storage
        comp = {'compression': 'gzip', 'compression_opts': 4}
        
        with h5py.File(filenameMatrix, 'w') as f:
            # ===== Matrix data =====
            matrix_grp = f.create_group('matrix')
            matrix_grp.attrs['n'] = n
            matrix_grp.attrs['nnz'] = nnz
            matrix_grp.create_dataset('data', data=A.data, **comp)
            matrix_grp.create_dataset('indices', data=A.indices, **comp)
            matrix_grp.create_dataset('indptr', data=A.indptr, **comp)
            
            # Preallocation helper
            nnz_per_row = np.diff(A.indptr)
            matrix_grp.create_dataset('nnz_per_row', data=nnz_per_row, **comp)
            
            # ===== RHS vector =====
            f.create_dataset('b', data=b, **comp)

            # ===== Permutation Vector if Cuthill-McKee reordering =====
            if perm is not None:
                f.create_dataset('perm', data=perm, **comp)

        self.Logger.info(f"Matrix saved to {filenameMatrix} . Elapsed time {time.time()-start} s")

        file_size = os.path.getsize(filenameMatrix)
        if file_size > (1024**3):
            file_size /= (1024**3)
            format = 'GB'
        elif file_size > (1024**2):
            file_size /= (1024**2)
            format = 'MB'
        elif file_size > (1024**1):
            file_size /= (1024**1)
            format = 'KB'
        self.Logger.info(f"  File size: {file_size:.2f} {format}")

        start = time.time()

        with h5py.File(filenameMesh, 'w') as f:
            # ===== Mesh data =====
            mesh_grp = f.create_group('mesh')
            
            # Node coordinates
            mesh_grp.create_dataset('nodes', data=self.Mesh.Nodes, **comp)
            mesh_grp.attrs['n_nodes'] = self.Mesh.n_nodes
            
            # Element sections
            elem_grp = mesh_grp.create_group('elements')
            for section_name, elem_data in self.Mesh.Elements.items():
                section_grp = elem_grp.create_group(section_name)
                section_grp.attrs['type'] = elem_data['type']
                section_grp.create_dataset('connectivity', 
                                        data=elem_data['connectivity'], 
                                        **comp)
            
            # Optional: Control volumes
            mesh_grp.create_dataset('DualControlVolume', 
                                data=self.Mesh.ControlVolumesPerNode, 
                                **comp)
            
            # Optional: Boundary normals
            # mesh_grp.create_dataset('boundary_normals', 
            #                     data=self.Mesh.boundaryNormal, 
            #                     **comp)
            
            self.Logger.info(f"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA self.exactSolution")
            if self.exactSolution >= 0:
                mesh_grp.create_dataset('ExactSolution', 
                                    data=self.exactSolutionFun(self.Mesh.Nodes[:, 0], self.Mesh.Nodes[:, 1], self.Mesh.Nodes[:, 2], self.exactSolution), 
                                    **comp)
                
        self.Logger.info(f"Cell structure saved to {filenameMesh} . Elapsed time {time.time()-start} s")
        
        file_size = os.path.getsize(filenameMesh)
        if file_size > (1024**3):
            file_size /= (1024**3)
            format = 'GB'
        elif file_size > (1024**2):
            file_size /= (1024**2)
            format = 'MB'
        elif file_size > (1024**1):
            file_size /= (1024**1)
            format = 'KB'
        self.Logger.info(f"  File size: {file_size:.2f} {format}")

        self.Logger.info(f"Done!")

    def solve_components(self, forceComponents=[0], momentComponents=[], solver=['spsolve'], useReordering=False, solverOptions={}):
        """
        Solve for all three components

        Parameters:
        -----------

        Returns:
        --------
        solutions : dict
            Solutions for each component
        """
        solutions = {}


        allNeumann = True
        for bc in self.boundary_conditions.keys():
            if not self.boundary_conditions[bc]['BCType'] == BC_TYPE_MAP.get('Neumann', 0):
                allNeumann = False
        self.allNeumann = allNeumann and not (solver[0] == "minres" or solver[0] == "cg")


        self.Mesh.build_DualControlVolumes()
        
        startTime = time.time()
        self.Logger.info(f"Initializing the A matrix and the b vector for the interior nodes...")
        # startTime = time.time()

        if self.Mesh.n_nodes > 100000:
            A, b = self.build_CV_fv_system_NumbaParallel(solverOptions)
        else:
            A, b = self.build_CV_fv_system(solverOptions)
        # self.Logger.info(f"With numba. Elapsed time {time.time()-startTime} s.")
        # startTime = time.time()
        # A, b = self.build_CV_fv_system()
        # self.Logger.info(f"Without numba. Elapsed time {time.time()-startTime} s.")

        self.Logger.info(f"Finished elaborating interior nodes. Elapsed time {time.time()-startTime} s.")


        for component_idx in forceComponents:
            self.Logger.info(f"\n{'='*60}")
            self.Logger.info(f"Solving for force component {component_idx}")
            self.Logger.info('='*60)

            startTime = time.time()
            A_WithBCs, b_withBCs = self.apply_CV_BCs(A, b, component_idx=component_idx, ForceOrMoment=FORCE_OR_MOMENT_MAP.get("Force", 0))
            self.Logger.info(f"Finished applying boundary conditions. Elapsed time {time.time()-startTime} s.")
            startTime = time.time()
            self.Logger.info(f"Starting linear system solver...")
            phi = self.solve_poisson(A_WithBCs, b_withBCs, solver, useReordering, solverOptions, component_idx=component_idx)
            self.Logger.info(f"Linear solver finished. Elapsed time {time.time()-startTime} s.")

            if self.allNeumann:
                phi = phi[:-1]
                residual = self.verify_solution(A_WithBCs[:-1, :-1], phi, b_withBCs[:-1])
            else:
                residual = self.verify_solution(A_WithBCs, phi, b_withBCs)
            
            if self.exactSolution >= 0:
                error = self.computeErrorFromExactSolution(phi)
                self.Logger.info(f"Error from exact solution = {error}")
                fid = open('Conv.log', mode='w')
                print(f"{self.Mesh.n_nodes} {error}", file=fid)
                fid.close()

            
            solutions[f'phi_{component_idx}'] = phi
            solutions[f'residual_phi_{component_idx}'] = residual

        if len(momentComponents) > 0:
            for component_idx in momentComponents:
                self.Logger.info(f"\n{'='*60}")
                self.Logger.info(f"Solving for moment component {component_idx}")
                self.Logger.info('='*60)
                A_WithBCs, b_withBCs = self.apply_CV_BCs(A, b, component_idx=component_idx, ForceOrMoment=FORCE_OR_MOMENT_MAP.get("Moment", 0))
                
                
                startTime = time.time()
                self.Logger.info(f"Starting linear system solver...")
                phi = self.solve_poisson(A_WithBCs, b_withBCs, solver, useReordering, solverOptions, component_idx=component_idx)
                self.Logger.info(f"Linear solver finished. Elapsed time {time.time()-startTime} s.")
                
                if self.allNeumann:
                    phi = phi[:-1]
                    residual = self.verify_solution(A_WithBCs[:-1, :-1], phi, b_withBCs[:-1])
                else:
                    residual = self.verify_solution(A_WithBCs, phi, b_withBCs)
                
                if self.exactSolution >= 0:
                    error = self.computeErrorFromExactSolution(phi)
                    self.Logger.info(f"Error from exact solution = {error}")
                    fid = open('Conv.log', mode='w')
                    print(f"{self.Mesh.n_nodes} {error}", file=fid)
                    fid.close()

                
                solutions[f'phi_m_{component_idx}'] = phi
                solutions[f'residual_phi_m_{component_idx}'] = residual

        return solutions
    
    def compute_residual(self, A, phi, b):
        """
        Compute residual: r = A*phi - b
        This should be ~0 if solution is correct
        """
        residual = A @ phi - b
        return residual

    def verify_solution(self, A, phi, b):
        """
        Verify solution quality
        """
        residual = self.compute_residual(A, phi, b)
        residual_norm = np.linalg.norm(residual)
        relative_residual = residual_norm / (np.linalg.norm(b) + 1e-12)
        
        self.Logger.info(f"Residual norm: {residual_norm:.6e}")
        self.Logger.info(f"Relative residual: {relative_residual:.6e}")
        self.Logger.info(f"Max residual: {np.abs(residual).max():.6e}")
        
        return residual
    
    def computeErrorFromExactSolution(self, phi):
        
        exactSolution = self.exactSolutionFun(self.Mesh.Nodes[:, 0], self.Mesh.Nodes[:, 1], self.Mesh.Nodes[:, 2], self.exactSolution)

        if self.allNeumann:
            Error = exactSolution - phi
            MeanError = np.sum(Error*self.Mesh.ControlVolumesPerNode)/np.sum(self.Mesh.ControlVolumesPerNode)
            ErrorAligned = Error-MeanError
            Error = np.sqrt(np.sum((ErrorAligned**2)*self.Mesh.ControlVolumesPerNode)/np.sum(self.Mesh.ControlVolumesPerNode))
        else:
            Error = exactSolution - phi
            Error = np.sqrt(np.sum((Error**2)*self.Mesh.ControlVolumesPerNode)/np.sum(self.Mesh.ControlVolumesPerNode))

        return Error


    def check_compatibility(self, b):
        """
        Check if RHS satisfies compatibility condition
        For pure Neumann: sum(b) should be ~0
        """
        b_sum = np.sum(b)
        self.Logger.infoint(f"RHS sum (should be ~0 for Neumann): {b_sum:.6e}")
        
        # Project RHS orthogonal to constant null space
        N = len(b)
        b_mean = np.mean(b)
        b_corrected = b - b_mean
        
        self.Logger.info(f"RHS mean removed: {b_mean:.6e}")
        return b_corrected
    
    def verify_null_space_removed(self, A, b):
        """
        Verify that constant vector is NOT in null space
        """
        N = A.shape[0]
        ones = np.ones(N)
        
        # A*ones should NOT be ~0 if null space is removed
        A_ones = A @ ones
        
        self.Logger.info(f"||A*ones|| = {np.linalg.norm(A_ones):.6e}")
        
        if np.linalg.norm(A_ones) < 1e-10:
            self.Logger.error("ERROR: Null space NOT removed! A*ones = 0")
            self.Logger.info("The pinned node constraint is not working.")
            return False
        else:
            self.Logger.info("Good: Null space appears removed")
            return True

    def export_solution_vtk(self, phi, output_file):
        """
        Export solution to VTK format

        Parameters:
        -----------
        phi : array
            Solution field
        output_file : str
            Output file path (.vtu)
        field_name : str
            Field name
        """
        try:
            import meshio

            # Prepare cells
            cells = []
            for section_name, elem_data in self.Mesh.Elements.items():
                elem_type = elem_data['type']
                connectivity = elem_data['connectivity']

                # Map CGNS to meshio cell types
                cell_type_map = {
                    10: ('tetra', 4),
                    17: ('hexahedron', 8),
                    12: ('pyramid', 5),
                    14: ('wedge', 6),
                    5: ('triangle', 3),
                    7: ('quad', 4),
                }

                if elem_type not in cell_type_map:
                    continue

                cell_type, nodes_per_elem = cell_type_map[elem_type]
                n_elems = len(connectivity) // nodes_per_elem
                cell_conn = connectivity
                cells.append((cell_type, cell_conn))

            # Append Control Volume information
            phi["DualControlVolume"] = self.Mesh.ControlVolumesPerNode
            # if self.debug:
            #     phi["Normal_X"] = self.Mesh.boundaryNormal[:, 0]
            #     phi["Normal_Y"] = self.Mesh.boundaryNormal[:, 1]
            #     phi["Normal_Z"] = self.Mesh.boundaryNormal[:, 2]
            if self.exactSolution >= 0:
                phi["ExactSolution"] = self.exactSolutionFun(self.Mesh.Nodes[:, 0], self.Mesh.Nodes[:, 1], self.Mesh.Nodes[:, 2], self.exactSolution)

            # Create mesh
            mesh = meshio.Mesh(
                points=self.Mesh.Nodes,
                cells=cells,
                point_data=phi
            )

            mesh.write(output_file)
            self.Logger.info(f"Solution exported to {output_file}")

        except ImportError:
            self.Logger.error("meshio not available. Install with: pip install meshio")
            self.Logger.info("Saving as NumPy array instead...")
            np.save(output_file.replace('.vtu', '.npy'), phi)
            self.Logger.info(f"Solution saved to {output_file.replace('.vtu', '.npy')}")
        except Exception as e:
            self.Logger.error(f"Error exporting to VTK: {e}")
            self.Logger.info("Saving as NumPy array instead...")
            np.save(output_file.replace('.vtu', '.npy'), phi)


    def set_num_threads(self, n):
        os.environ['OMP_NUM_THREADS'] = str(n)
        os.environ['OPENBLAS_NUM_THREADS'] = str(n)
        os.environ['MKL_NUM_THREADS'] = str(n)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(n)
        os.environ['NUMEXPR_NUM_THREADS'] = str(n)

    def bicgstab_solver(self, A, b, x0, solverOptions):

        tol=1e-10
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_precon=True
        if 'use_precon' in solverOptions.keys():
            use_precon = solverOptions['use_precon']

        # Try with ILU preconditioner
        M = None
        if use_precon:
            M = self.build_Preconditioner(A, solverOptions)
        

        callback = None
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)


        self.Logger.info("Starting BiCGSTAB...")
        x, info = bicgstab(A, b, 
                        x0=x0,
                        tol=tol, 
                        maxiter=maxiter,
                        M=M,
                        callback=callback)

        # Check result
        self.Logger.info(f"\nBiCGSTAB finished")
        
        if info == 0:
            self.Logger.info("✓ SUCCESS: Converged to tolerance")
            residual = np.linalg.norm(A @ x - b)
            self.Logger.info(f"Final residual: {residual:.6e}")
        elif info > 0:
            self.Logger.info(f"✗ FAILURE: Did not converge in {info} iterations")
            residual = np.linalg.norm(A @ x - b)
            self.Logger.info(f"Final residual: {residual:.6e}")
            self.Logger.info("Solution is NOT reliable - use direct solver instead")
        else:
            self.Logger.info("✗ FAILURE: Numerical breakdown")

        return x

    def minres_solver(self, A, b, x0, solverOptions):

        tol=1e-10
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_precon=True
        if 'use_precon' in solverOptions.keys():
            use_precon = solverOptions['use_precon']

        # Try with ILU preconditioner
        M = None
        if use_precon:
            M = self.build_Preconditioner(A, solverOptions)
        

        callback = None
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        self.Logger.info("Starting MinRES...")
        x, info = minres(A, b, 
                        x0=x0,
                        tol=tol, 
                        maxiter=maxiter,
                        M=M,
                        callback=callback)

        # Check result
        self.Logger.info(f"\MinRES finished")
        
        if info == 0:
            self.Logger.info("✓ SUCCESS: Converged to tolerance")
            residual = np.linalg.norm(A @ x - b)
            self.Logger.info(f"Final residual: {residual:.6e}")
        elif info > 0:
            self.Logger.info(f"✗ FAILURE: Did not converge in {info} iterations")
            residual = np.linalg.norm(A @ x - b)
            self.Logger.info(f"Final residual: {residual:.6e}")
            self.Logger.info("Solution is NOT reliable - use direct solver instead")
        else:
            self.Logger.info("✗ FAILURE: Numerical breakdown")

        return x

    def fgmres_solver(self, A, b, x0, solverOptions):

        tol=1e-10
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        restart=None
        if 'restart' in solverOptions.keys():
            restart = solverOptions['restart']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_precon=True
        if 'use_precon' in solverOptions.keys():
            use_precon = solverOptions['use_precon']

        # Try with ILU preconditioner
        M = None
        if use_precon:
            M = self.build_Preconditioner(A, solverOptions)

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Run GMRES
        self.Logger.info("Starting FGMRES solve...")
        x, info = pyamg.krylov.fgmres(A, b, tol=tol, x0=x0, M=M, maxiter=maxiter, restart=restart, callback=callback)
        if verbose:
            if info != 0:
                self.Logger.info("Warning: FGMRES did not converge (info = {info})")
            else:
                self.Logger.info(f"FGMRES completed after {callback.niter} iterations. info={info}")
        else:
            if info != 0:
                self.Logger.info("Warning: FGMRES did not converge (info = {info})")

        return x
    
    def gmres_solver(self, A, b, x0, solverOptions):

        tol=1e-10
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        restart=None
        if 'restart' in solverOptions.keys():
            restart = solverOptions['restart']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_precon=True
        if 'use_precon' in solverOptions.keys():
            use_precon = solverOptions['use_precon']

        # Try with ILU preconditioner
        M = None
        if use_precon:
            M = self.build_Preconditioner(A, solverOptions)

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Run GMRES
        print("Starting GMRES solve...")
        x, info = gmres(A, b, tol=tol, x0=x0, M=M, maxiter=maxiter, restart=restart, callback=callback, callback_type=callback_type)
        if verbose:
            if info != 0:
                self.Logger.info("Warning: GMRES did not converge (info = {info})")
            else:
                self.Logger.info(f"GMRES completed after {callback.niter} iterations. info={info}")
        else:
            if info != 0:
                self.Logger.info("Warning: GMRES did not converge (info = {info})")
        return x
    
    def cg_solver(self, A, b, x0, solverOptions):

        tol=1e-10
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_precon=True
        if 'use_precon' in solverOptions.keys():
            use_precon = solverOptions['use_precon']

        # Try with ILU preconditioner
        M = None
        if use_precon:
            M = self.build_Preconditioner(A, solverOptions)

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Run GMRES
        print("Starting CG solve...")
        x, info = cg(A, b, tol=tol, x0=x0, M=M, maxiter=maxiter, callback=callback)
        if verbose:
            if info != 0:
                self.Logger.info("Warning: GMRES did not converge (info = {info})")
            else:
                self.Logger.info(f"GMRES completed after {callback.niter} iterations. info={info}")
        else:
            if info != 0:
                self.Logger.info("Warning: GMRES did not converge (info = {info})")
        return x

    class UnifiedVerboseCallback:
        def __init__(self, A, b, Logger):
            self.niter = 0
            self.A = A
            self.b = b
            self.Logger = Logger
            
        def __call__(self, xk):
            self.niter += 1
            
            # Compute residual from solution vector
            res = self.A @ xk - self.b
            res_norm = np.linalg.norm(res)
            
            # Print every iteration (or every N iterations to reduce output)
            if self.niter % 10 == 0 or self.niter == 1:
                self.Logger.info(f"Iter {self.niter}: residual = {res_norm:.6e}")

    def build_Preconditioner(self, A, solverOptions):

        preconName = "BlockJacobi"
        if 'preconName' in solverOptions.keys():
            preconName = solverOptions['preconName']

        if preconName == "BlockJacobi":
            M = self.build_BlockJacobi(A, solverOptions)
        elif preconName == "Jacobi":
            M = self.build_Jacobi(A, solverOptions)
        elif preconName == "ILU":
            M = self.build_ILU(A, solverOptions)
        elif preconName == "AMG":
            M = self.build_AMG(A, solverOptions)
        elif preconName == "AdapSAS":
            M = self.build_AdaptiveSAS(A, solverOptions)
        else:
            self.Logger.error(f"ERROR! Preconditioner called {preconName} is not available.")
            self.Logger.error(f"Please select one of the followings: Jacobi, BlockJacobi [default], ILU, AMG.")


        return M
     
    def build_ILU(self, A, solverOptions):

        fill_factor=10
        if 'fill_factor' in solverOptions.keys():
            fill_factor = solverOptions['fill_factor']
        drop_tol=1e-4
        if 'drop_tol' in solverOptions.keys():
            drop_tol = solverOptions['drop_tol']

        M = None
        try:
            self.Logger.info("Building ILU preconditioner...")
            ilu = spilu(A.tocsc(), fill_factor=fill_factor, drop_tol=drop_tol)
            M = LinearOperator(A.shape, ilu.solve)
            self.Logger.info("ILU: SUCCESS")
        except Exception as e:
            self.Logger.error(f"ILU failed: {e}, solving without preconditioner")
            M = None
        
        return M
    
    def build_Jacobi(self, A, solverOptions):

        """Jacobi (diagonal) preconditioner as a LinearOperator."""

        try:
            self.Logger.info("Building Jacobi preconditioner...")
            D_inv = 1.0 / A.diagonal()
            def matvec(x):
                return D_inv * x
            M = LinearOperator(A.shape, matvec, dtype=A.dtype)
        except Exception as e:
            self.Logger.error(f"Jacobi failed: {e}, solving without preconditioner")
            M = None
        
        return M
    

    import numpy as np

    def build_BlockJacobi(self, A, solverOptions):
        """
        Block Jacobi preconditioner as a LinearOperator.
        Each block is inverted independently.
        """

        block_size=10
        if 'block_size' in solverOptions.keys():
            block_size = solverOptions['block_size']

        try:
            self.Logger.info("Building Block Jacobi preconditioner...")
            n = A.shape[0]
            blocks = []
            for i in range(0, n, block_size):
                block = A[i:i+block_size, i:i+block_size].toarray()
                blocks.append(np.linalg.pinv(block))
            def matvec(x):
                y = np.zeros_like(x)
                for i, inv_block in enumerate(blocks):
                    idx = slice(i*block_size, (i+1)*block_size)
                    y[idx] = inv_block @ x[idx]
                return y
            M = LinearOperator(A.shape, matvec, dtype=A.dtype)
        except Exception as e:
            self.Logger.error(f"Block Jacobi failed: {e}, solving without preconditioner")
            M = None
        

        return M
    
    def build_AMG(self, A, solverOptions):
        """
        AMG preconditioner using PyAMG's smoothed aggregation solver.
        Requires pyamg to be installed.
        """
        try:
            B = np.ones((A.shape[0], 1))
            ml = pyamg.smoothed_aggregation_solver(A, B,
                                                    symmetry='symmetric',  # or 'hermitian' 
                                                            
                                                    # Key: Use simple Jacobi smoothing (not energy!)
                                                    smooth='jacobi',  # Much less memory than energy
                                                    
                                                    # Aggressive coarsening to reduce memory
                                                    strength=('symmetric', {'theta': 0.45}),  # Higher theta = fewer connections
                                                    
                                                    # Standard aggregation (not lloyd - too expensive at this scale)
                                                    aggregate='standard',
                                                    
                                                    # Limit hierarchy depth
                                                    max_levels=3,
                                                    max_coarse=50,  # Larger coarse grid to stop earlier
                                                    
                                                    # Simple smoothers
                                                    presmoother=('gauss_seidel', {'sweep': 'forward'}),
                                                    postsmoother=('gauss_seidel', {'sweep': 'backward'}),
                                                    
                                                    keep=True  # Don't keep diagnostics
                                                    )
            M = ml.aspreconditioner(cycle='W')
        except ImportError:
            self.Logger.error("PyAMG is required for AMG preconditioning! Resorting to block jacobi")
            M = self.build_BlockJacobi(A, solverOptions)

        return M
    

    def build_AdaptiveSAS(self, A, solverOptions):
        """
        AMG preconditioner using PyAMG's smoothed aggregation solver.
        Requires pyamg to be installed.
        """
        try:
            ml, work = pyamg.aggregation.adaptive_sa_solver(A, 
                                                            symmetry='symmetric',  # or 'hermitian' 
                                                            num_candidates=2,       # Number of candidates to generate
                                                            candidate_iters=8,      # Smoothing passes per level
                                                            improvement_iters=1,      # Smoothing passes per level
                                                            epsilon=0.08,            # Target convergence factor
                                                            # Strength and smoothing:
                                                            strength=('symmetric', {'theta': 0.15}),  # Lower theta (0.1-0.2) for bad meshes
                                                            smooth=('energy', {'krylov': 'cg', 'maxiter': 6}),  # Energy minimization
                                                            
                                                            # Hierarchy control:
                                                            max_levels=15,                   # Allow more levels
                                                            max_coarse=50,                   # Smaller coarse grid for better setup
                                                            
                                                            # Smoothers:
                                                            prepostsmoother=('gauss_seidel', {'sweep': 'symmetric', 'iterations': 2}),
                                                            
                                                            keep=True                        # Keep diagnostics for bad meshes
                                                            )
            M = ml.aspreconditioner(cycle='W')
        except ImportError:
            self.Logger.error("PyAMG is required for AMG preconditioning! Resorting to block jacobi")
            M = self.build_BlockJacobi(A, solverOptions)

        return M
    

