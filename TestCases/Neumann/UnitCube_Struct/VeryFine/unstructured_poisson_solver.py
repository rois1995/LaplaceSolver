"""
Unstructured Poisson Equation Solver with CGNS Grid Support
Solves ∇²φᵢ = 0 in domain Vf with Neumann boundary conditions
Supports mixed element types: tetrahedra, hexahedra, pyramids, prisms

Compatible with NumPy 2.0+
"""
import sys
import numpy as np
import os
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import bicgstab, gmres, minres, spilu, LinearOperator
from pyamg.krylov import fgmres
from scipy.sparse.csgraph import reverse_cuthill_mckee
from collections import defaultdict
import warnings
from Mesh import MeshClass

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
        self.nodes = None
        self.elements = {}
        self.boundary_conditions = {}
        self.volume_condition = 0.0
        self.exactSolution = options["exactSolution"]
        self.boundary_faces = {}
        self.boundary_nodes = {}
        self.n_nodes = 0
        self.n_elements = 0
        self.cell_volumes = None
        self.cell_centers = None
        self.isOnBoundary = None
        self.boundaryNormal = None
        self.BCsAssociatedToNode = None
        self.node_neighbors = None
        self.nDim = options["nDim"]
        self.Mesh = MeshClass(Logger, options["GridName"], options["nDim"])
        self.allNeumann = False
        self.debug = options["debug"]
        self.Logger = Logger
        self.exactSolutionFun = options["exactSolutionFun"]

    def read_cgns_unstructured(self, options):
        """
        Read unstructured CGNS grid with mixed element types
        Compatible with NumPy 2.0
        """
        self.Logger.info(f"Reading CGNS file: {self.GridFileName}")

        try:
            self.Mesh._read_cgns_h5py(options["BlockName"], options["BoundaryConditions"])
        except Exception as e:
            self.Logger.error(f"h5py read failed: {e}. Failed to read CGNS file!")
            raise 
        
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
                self.Logger.info(f"Available boundaries: {list(set(list(self.Mesh.boundary_nodes.keys()) + list(self.boundary_faces.keys())))}")
                return

            self.boundary_conditions[bc_name] = {
                'BCType': BC[bc_name]['BCType'],
                'Value': BC[bc_name]['Value'],
                'typeOfExactSolution': BC[bc_name]['typeOfExactSolution']
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

        self.volume_condition = volume_BC


    def build_CV_fv_system(self):
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
        A = lil_matrix((N+int(self.allNeumann), N+int(self.allNeumann)))
        b = np.zeros(N+int(self.allNeumann))

        self.Logger.info(f"Building system on interior nodes ({N} nodes)...")

        node_neighbors = self.Mesh.NodesConnectedToNode
        ControlFaceDictPerEdge = self.Mesh.ControlFaceDictPerEdge
        if self.nDim == 2:
            ControlFaceNormalDictPerEdge = self.Mesh.ControlFaceNormalDictPerEdge
        ControlVolumesPerNode = self.Mesh.ControlVolumesPerNode
        # Build Laplacian matrix
        for i in range(N):
            if i%10000 == 0:
                self.Logger.info(f"Assembling matrix at point: {i}")
            neighbors = list(node_neighbors[i])

            controlVolume = ControlVolumesPerNode[i]

            if len(neighbors) == 0:
                # Isolated node
                A[i, i] = 1.0
                b[i] = 0.0
                continue

            # Interior node: standard Laplacian
            # Approximate: ∇²φ ≈ Σ(φⱼ - φᵢ) / dᵢⱼ²
            diag_val = 0.0
            for j in neighbors:
                distVec = self.Mesh.Nodes[j] - self.Mesh.Nodes[i]
                dist = np.linalg.norm(distVec)
                controlAreaIJ = ControlFaceDictPerEdge[str(i)+"-"+str(j)]
                proj=1
                if self.nDim == 2:
                    faceNormalIJ = ControlFaceNormalDictPerEdge[str(i)+"-"+str(j)]
                    proj = np.sum(distVec*faceNormalIJ)/dist
                if dist > 1e-10:
                    coeff = controlAreaIJ * proj/ (dist*controlVolume)
                    A[i, j] = coeff
                    diag_val -= coeff

            A[i, i] = diag_val
            b[i] = self.volume_condition["Value"](self.Mesh.Nodes[i, 0], self.Mesh.Nodes[i, 1], self.Mesh.Nodes[i, 2], self.volume_condition["typeOfExactSolution"])  # RHS = 0 for Laplace equation

        if self.allNeumann:
            A[-1, :] = 1
            A[-1, -1] = 0
            A[:-1, -1] = 1
            b[-1] = 0

        return A, b




    def apply_CV_BCs(self, A, b, component_idx=1):
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

        actualPoint = 0

        # Build Laplacian matrix
        for iBoundNode in np.where(self.Mesh.isNodeOnBoundary)[0]:

            if actualPoint%1000 == 0:
                self.Logger.info(f"Imposing boundary conditions at point: {iBoundNode}")

            bc_name = self.Mesh.BCsAssociatedToNode[iBoundNode][0]
            BoundaryCVArea = self.Mesh.boundaryCVArea[iBoundNode]
            bc_data = self.boundary_conditions[bc_name]
            bNormal = self.Mesh.boundaryNormal[iBoundNode]
            if bc_data['BCType'] == 'Neumann':
                # Neumann BC: ∂φ/∂n = bc_value
                # Approximate using one-sided difference
                if callable(bc_data['Value']):
                    bc_val = np.sum(bc_data['Value'](self.Mesh.Nodes[iBoundNode, 0], self.Mesh.Nodes[iBoundNode, 1], self.Mesh.Nodes[iBoundNode, 2], bc_data["typeOfExactSolution"])*bNormal)
                else:
                    if bc_data['Value'] == 0: # Farfield
                        bc_val = 0
                    elif bc_data['Value'] == 'Normal':
                        bc_val = bNormal[component_idx]
                    else:
                        self.Logger.error("ERROR! Unrecognized boundary condition value!")
                        exit(1)          

                b[iBoundNode] = -bc_val*BoundaryCVArea/ControlVolumesPerNode[iBoundNode]

            elif bc_data['BCType'] == 'Dirichlet':
                # Dirichlet BC: φ = bc_value
                bc_val = bc_data['Value']
                
                if callable(bc_val):
                    bc_val = bc_val(self.Mesh.Nodes[iBoundNode, 0], self.Mesh.Nodes[iBoundNode, 1], self.Mesh.Nodes[iBoundNode, 2], bc_data["typeOfExactSolution"])
                A[iBoundNode, :] = 0.0
                A[iBoundNode, iBoundNode] = 1.0
                b[iBoundNode] = bc_val
                
            actualPoint+= 1
                
        return A, b


    def solve_poisson(self, A, b, solver, useReordering=False, solverOptions={}, component_idx=1):
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
        A_csr = csr_matrix(A)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(A_csr.toarray())
        # np.savetxt("Matrix.csv", A_csr.toarray(), delimiter=",")
        # np.savetxt("b.csv", b, delimiter=",")
        # A_csr.eliminate_zeros()

        if useReordering:

            self.Logger.info("Apply Reverse Cuthill-Mckee ordering")
            # Example usage in your solver (add before calling GMRES/SPSOLVE):
            perm = reverse_cuthill_mckee(A_csr, symmetric_mode=True)  # Or A_csr/b

            # Step 4: Permute matrix and RHS
            A_rcm = A_csr[perm, :][:, perm]
            b_rcm = b[perm]

            A_toSolve = A_rcm
            b_toSolve = b_rcm
            permutation = perm

        else:

            A_toSolve = A_csr
            b_toSolve = b

        self.verify_null_space_removed(A_toSolve, b_toSolve)
        # b_new = self.check_compatibility(b)

        diag = A_toSolve.diagonal()
        if np.any(diag == 0):
            self.Logger.info(f"Warning: There are {len(np.where(diag==0)[0])} zeros on the diagonal!")
            self.Logger.info(f"Indexes where this happens: {np.where(diag==0)[0][0]}")

        # Solve
        self.Logger.info(f"Solving linear system...")

        if solver == 'spsolve':
            self.Logger.info("Using spsolve function (direct solver)")
            try:
                phi = spsolve(A_toSolve, b_toSolve)
            except Exception as e:
                self.Logger.error(f"Direct solve failed: {e}")
                self.Logger.info("Trying iterative solver...")
                phi = self.gmres_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'gmres':
            self.Logger.info("Using GMRES (iterative solver)")
            phi = self.gmres_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'fgmres':
            self.Logger.info("Using FGMRES (iterative solver)")
            phi = self.fgmres_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'bicgstab':
            self.Logger.info("Using BiCGSTAB (iterative solver)")
            phi = self.bicgstab_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'minres':
            self.Logger.info("Using MinRES (iterative solver)")
            phi = self.minres_solver(A_toSolve, b_toSolve, solverOptions)

        self.Logger.info(f"Solution range: [{phi.min():.6e}, {phi.max():.6e}]")

        if useReordering:
            phi_reordered = np.empty_like(phi)
            phi_reordered[perm] = phi
            phi = phi_reordered


        return phi

    def solve_components(self, components=[0], solver='spsolve', useReordering=False, solverOptions={}):
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
            if not self.boundary_conditions[bc]['BCType'] == 'Neumann':
                allNeumann = False
        self.allNeumann = allNeumann

        # Compute geometric properties if not done
        # if self.cell_volumes is None:
        #     self.compute_geometric_properties()

        
        self.Mesh.build_DualControlVolumes()
        A, b = self.build_CV_fv_system()
            # exit(1)

        for component_idx in components:
            self.Logger.info(f"\n{'='*60}")
            self.Logger.info(f"Solving component {component_idx}")
            self.Logger.info('='*60)
            A, b = self.apply_CV_BCs(A, b, component_idx=component_idx)
            phi = self.solve_poisson(A, b, solver, useReordering, solverOptions, component_idx=component_idx)
            if self.allNeumann:
                phi = phi[:-1]
                residual = self.verify_solution(A[:-1, :-1], phi, b[:-1])
            else:
                residual = self.verify_solution(A, phi, b)
            
            if not self.exactSolution == "None":
                error = self.computeErrorFromExactSolution(phi)
                self.Logger.info(f"Error from exact solution = {error}")
                fid = open('Conv.log', mode='w')
                print(f"{self.Mesh.n_nodes} {error}", file=fid)
                fid.close()

            
            solutions[f'phi_{component_idx}'] = phi
            solutions[f'residual_phi_{component_idx}'] = residual

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
            if self.debug:
                phi["Normal_X"] = self.Mesh.boundaryNormal[:, 0]
                phi["Normal_Y"] = self.Mesh.boundaryNormal[:, 1]
                phi["Normal_Z"] = self.Mesh.boundaryNormal[:, 2]
            if not self.exactSolution == "None":
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

    def bicgstab_solver(self, A, b, solverOptions):

        tol=1e-6
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_ilu=True
        if 'use_ilu' in solverOptions.keys():
            use_ilu = solverOptions['use_ilu']

        # Try with ILU preconditioner
        M = None
        if use_ilu:
            try:
                self.Logger.info("Building ILU preconditioner...")
                ilu = spilu(A.tocsc(), fill_factor=10, drop_tol=1e-4)
                M = LinearOperator(A.shape, ilu.solve)
                self.Logger.info("ILU: SUCCESS")
            except Exception as e:
                self.Logger.error(f"ILU failed: {e}, solving without preconditioner")
                M = None
        

        callback = None
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Initial guess (not zeros for Neumann problems)
        x0 = np.random.randn(len(b))
        x0 = x0 - np.mean(x0)  # Remove constant component


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
    

    def minres_solver(self, A, b, solverOptions):

        tol=1e-6
        if 'tol' in solverOptions.keys():
            tol = solverOptions['tol']
        maxiter = 1000
        if 'maxiter' in solverOptions.keys():
            maxiter = solverOptions['maxiter']
        verbose=True
        if 'verbose' in solverOptions.keys():
            verbose = solverOptions['verbose']
        use_ilu=True
        if 'use_ilu' in solverOptions.keys():
            use_ilu = solverOptions['use_ilu']

        # Try with ILU preconditioner
        M = None
        if use_ilu:
            try:
                self.Logger.info("Building ILU preconditioner...")
                ilu = spilu(A.tocsc(), fill_factor=10, drop_tol=1e-4)
                M = LinearOperator(A.shape, ilu.solve)
                self.Logger.info("ILU: SUCCESS")
            except Exception as e:
                self.Logger.error(f"ILU failed: {e}, solving without preconditioner")
                M = None
        

        callback = None
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Initial guess (not zeros for Neumann problems)
        x0 = np.random.randn(len(b))
        x0 = x0 - np.mean(x0)  # Remove constant component


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

    def fgmres_solver(self, A, b, solverOptions):

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
        use_ilu=True
        if 'use_ilu' in solverOptions.keys():
            use_ilu = solverOptions['use_ilu']

        # ILU preconditioner (fast for well-formed Laplace matrices)
        M = None
        if use_ilu:
            try:
                # spilu requires CSC/CSR format
                ilu = spilu(A.tocsc(), 
                            fill_factor=10,  # More fill
                            drop_tol=1e-4)
                M = LinearOperator(A.shape, ilu.solve)
                self.Logger.info("ILU preconditioner: Success")
            except Exception as e:
                self.Logger.error(f"ILU failed: {e}, solving without preconditioner")
                M = None

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Provide non-constant initial guess
        N = A.shape[0]
        x0 = np.random.randn(N)
        x0 = x0 - np.mean(x0)  # Remove constant component

        # Run GMRES
        self.Logger.info("Starting FGMRES solve...")
        x, info = fgmres(A, b, tol=tol, x0=x0, M=M, maxiter=maxiter, restart=restart, callback=callback)
        if verbose:
            if info != 0:
                self.Logger.info("Warning: FGMRES did not converge (info = {info})")
            else:
                self.Logger.info(f"FGMRES completed after {callback.niter} iterations. info={info}")
        else:
            if info != 0:
                self.Logger.info("Warning: FGMRES did not converge (info = {info})")

        return x
    
    def gmres_solver(self, A, b, solverOptions):

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
        use_ilu=True
        if 'use_ilu' in solverOptions.keys():
            use_ilu = solverOptions['use_ilu']

        # ILU preconditioner (fast for well-formed Laplace matrices)
        M = None
        if use_ilu:
            try:
                # spilu requires CSC/CSR format
                ilu = spilu(A.tocsc(), 
                            fill_factor=10,  # More fill
                            drop_tol=1e-4)
                M = LinearOperator(A.shape, ilu.solve)
                self.Logger.info("ILU preconditioner: Success")
            except Exception as e:
                self.Logger.error(f"ILU failed: {e}, solving without preconditioner")
                M = None

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b, self.Logger)

        # Provide non-constant initial guess
        N = A.shape[0]
        x0 = np.random.randn(N)
        x0 = x0 - np.mean(x0)  # Remove constant component

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
