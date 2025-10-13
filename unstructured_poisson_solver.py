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

    def __init__(self, options):
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
        self.exactSolution = False
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
        self.Mesh = MeshClass(options["GridName"], options["nDim"])
        self.UseApproximateLaplacianFormulation = options["UseApproximateLaplacianFormulation"]

    def read_cgns_unstructured(self, options):
        """
        Read unstructured CGNS grid with mixed element types
        Compatible with NumPy 2.0
        """
        print(f"Reading CGNS file: {self.GridFileName}")

        try:
            self.Mesh._read_cgns_h5py(options["BlockName"], options["BoundaryConditions"])
        except Exception as e:
            print(f"h5py read failed: {e}")
            raise RuntimeError("Failed to read CGNS file.")
        
    def construct_secondaryMeshStructures(self):
        
        print(f"Constructing secondary mesh structures...")

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
                print(f"Warning: Boundary '{bc_name}' not found in grid")
                print(f"Available boundaries: {list(set(list(self.Mesh.boundary_nodes.keys()) + list(self.boundary_faces.keys())))}")
                return

            self.boundary_conditions[bc_name] = {
                'BCType': BC[bc_name]['BCType'],
                'Value': BC[bc_name]['Value'],
                'typeOfExactSolution': BC[bc_name]['typeOfExactSolution']
            }
            print(f"Set BC on '{bc_name}': {BC[bc_name]['BCType']} = {BC[bc_name]['Value']}, typeOfExactSolution = {BC[bc_name]['typeOfExactSolution']}")

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


    def build_approx_fv_system(self):
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
        A = lil_matrix((N, N))
        b = np.zeros(N)

        print(f"Building system on interior nodes ({len(np.where(self.Mesh.isNodeOnBoundary == False)[0])} nodes)...")

        node_neighbors = self.Mesh.NodesConnectedToNode
        # Build Laplacian matrix
        for i in range(N):
            if i%10000 == 0:
                print("Assembling matrix at point: ", i)
            neighbors = list(node_neighbors[i])

            if len(neighbors) == 0:
                # Isolated node
                A[i, i] = 1.0
                b[i] = 0.0
                continue
            
            # if len(bcs) > 0:
            #     print("Point", i, "of coords", self.nodes[i], "is on boundary", bcs, "and has normals")
            #     print(bnormal)

            if not self.Mesh.isNodeOnBoundary[i]:
                # Interior node: standard Laplacian
                # Approximate: ∇²φ ≈ Σ(φⱼ - φᵢ) / dᵢⱼ²
                diag_val = 0.0
                for j in neighbors:
                    dist = np.linalg.norm(self.Mesh.Nodes[i] - self.Mesh.Nodes[j])
                    if dist > 1e-10:
                        coeff = 1.0 / dist**2
                        A[i, j] = coeff
                        diag_val -= coeff

                A[i, i] = diag_val
                b[i] = self.volume_condition["Value"](self.Mesh.Nodes[i, 0], self.Mesh.Nodes[i, 1], self.Mesh.Nodes[i, 2], self.volume_condition["typeOfExactSolution"])  # RHS = 0 for Laplace equation

        return A, b
    

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
        A = lil_matrix((N, N))
        b = np.zeros(N)

        print(f"Building system on interior nodes ({N} nodes)...")

        node_neighbors = self.Mesh.NodesConnectedToNode
        ControlFaceDictPerEdge = self.Mesh.ControlFaceDictPerEdge
        ControlVolumesPerNode = self.Mesh.ControlVolumesPerNode
        # Build Laplacian matrix
        for i in range(N):
            if i%10000 == 0:
                print("Assembling matrix at point: ", i)
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
                dist = np.linalg.norm(self.Mesh.Nodes[i] - self.Mesh.Nodes[j])
                controlAreaIJ = ControlFaceDictPerEdge[str(i)+"-"+str(j)]
                if dist > 1e-10:
                    coeff = controlAreaIJ / (dist*controlVolume)
                    A[i, j] = coeff
                    diag_val -= coeff

            A[i, i] = diag_val
            b[i] = self.volume_condition["Value"](self.Mesh.Nodes[i, 0], self.Mesh.Nodes[i, 1], self.Mesh.Nodes[i, 2], self.volume_condition["typeOfExactSolution"])  # RHS = 0 for Laplace equation

        return A, b





    def apply_approx_BCs(self, A, b, component_idx=1):
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

        print(f"Applying Boundary Conditions for component {component_idx} ({len(np.where(self.Mesh.isNodeOnBoundary == True)[0])} nodes)...")

        node_neighbors = self.Mesh.NodesConnectedToNode

        actualPoint = 0

        # Build Laplacian matrix
        for i in range(N):
            neighbors = list(node_neighbors[i])

            if self.Mesh.isNodeOnBoundary[i]:

                if actualPoint%1000 == 0:
                    print("Imposing boundary conditions at point: ", i)

                bc_name = self.Mesh.BCsAssociatedToNode[i][0]
                bc_data = self.boundary_conditions[bc_name]
                if bc_data['BCType'] == 'Neumann':
                    # Neumann BC: ∂φ/∂n = bc_value
                    # Approximate using one-sided difference
                    if bc_data['Value'] == 0: # Farfield
                        bc_val = 0
                    elif bc_data['Value'] == 'Normal':
                        bc_val = self.Mesh.boundaryNormal[i, component_idx]
                    else:
                        print("ERROR! Unrecognized boundary condition value!")
                        exit(1)

                    # Use neighboring nodes
                    AreNeighborsOnBoundaries = self.Mesh.isNodeOnBoundary[neighbors]
                    if not AreNeighborsOnBoundaries.all():
                        actualNeighborsIndices = np.where(AreNeighborsOnBoundaries == False)[0]
                        avg_dist = 0.0
                        for iNeigh in actualNeighborsIndices:
                            actualNeighbor = neighbors[iNeigh]
                            dist = np.linalg.norm(self.Mesh.Nodes[i] - self.Mesh.Nodes[actualNeighbor, :])
                            A[i, actualNeighbor] = 1.0/dist
                            avg_dist += dist

                        avg_dist /= len(actualNeighborsIndices)

                        A[i, i] = -len(actualNeighborsIndices)/avg_dist
                        b[i] = bc_val
                    else:
                        print("Problem! Point on the boundary with no neighbors inside the volume! Using boundary nodes")
                        # print("Point ", i, "coordinates", self.nodes[i], "on boundary", bcs)
                        A[i, i] = 1.0
                        avg_dist = 0.0
                        for iNeigh in neighbors:
                            dist = np.linalg.norm(self.Mesh.Nodes[i] - self.Mesh.Nodes[iNeigh, :])
                            A[i, iNeigh] = 1.0/dist
                            avg_dist += dist

                        avg_dist /= len(neighbors)

                        A[i, i] = -len(neighbors)/avg_dist
                        b[i] = bc_val

                elif bc_data['BCType'] == 'Dirichlet':
                    # Dirichlet BC: φ = bc_value
                    bc_val = bc_data['Value']
                    
                    if callable(bc_val):
                        bc_val = bc_val(self.Mesh.Nodes[i, 0], self.Mesh.Nodes[i, 1], self.Mesh.Nodes[i, 2], bc_data["typeOfExactSolution"])
                    A[i, :] = 0.0
                    A[i, i] = 1.0
                    b[i] = bc_val
                    
                actualPoint+= 1


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

        print(f"Applying Boundary Conditions for component {component_idx} ({len(np.where(self.Mesh.isNodeOnBoundary == True)[0])} nodes)...")

        node_neighbors = self.Mesh.NodesConnectedToNode
        ControlVolumesPerNode = self.Mesh.ControlVolumesPerNode

        actualPoint = 0

        # Build Laplacian matrix
        for iBoundNode in np.where(self.Mesh.isNodeOnBoundary)[0]:

            if actualPoint%1000 == 0:
                print("Imposing boundary conditions at point: ", iBoundNode)

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
                        print("ERROR! Unrecognized boundary condition value!")
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
        
        allNeumann = True
        for bc in self.boundary_conditions.keys():
            if not self.boundary_conditions[bc]['BCType'] == 'Neumann':
                allNeumann = False

        if allNeumann:
            # Fix one node to remove null space for pure Neumann
            A[0, :] = 0.0
            A[0, 0] = 1.0  # Pin in the reordered system
            b[0] = 0.0

            print("Pinned node row:", A[0].toarray())
            print("Pinned node RHS:", b[0])

        print("Convert matrix to CSR format")
        # Convert to CSR format
        A_csr = csr_matrix(A)
        # np.set_printoptions(threshold=sys.maxsize)
        # print(A_csr.toarray())
        # np.savetxt("Matrix.csv", A_csr.toarray(), delimiter=",")
        # np.savetxt("b.csv", b, delimiter=",")
        # A_csr.eliminate_zeros()

        if useReordering:

            print("Apply Reverse Cuthill-Mckee ordering")
            # Example usage in your solver (add before calling GMRES/SPSOLVE):
            perm = reverse_cuthill_mckee(A_csr, symmetric_mode=True)  # Or A_csr/b

            # Step 4: Permute matrix and RHS
            A_rcm = A_csr[perm, :][:, perm]
            b_rcm = b[perm]

            if allNeumann:
                # Fix one node to remove null space for pure Neumann
                A_rcm[0, :] = 0.0
                A_rcm[0, 0] = 1.0  # Pin in the reordered system
                b_rcm[0] = 0.0

                print("Pinned node row:", A[0].toarray())
                print("Pinned node RHS:", b[0])

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
            print(f"Warning: There are {len(np.where(diag==0)[0])} zeros on the diagonal!")
            print(np.where(diag==0)[0][0])

        # Solve
        print(f"Solving linear system...")

        if solver == 'spsolve':
            print("Using spsolve function (direct solver)")
            try:
                phi = spsolve(A_toSolve, b_toSolve)
            except Exception as e:
                print(f"Direct solve failed: {e}")
                print("Trying iterative solver...")
                phi = self.gmres_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'gmres':
            print("Using GMRES (iterative solver)")
            phi = self.gmres_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'fgmres':
            print("Using FGMRES (iterative solver)")
            phi = self.fgmres_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'bicgstab':
            print("Using BiCGSTAB (iterative solver)")
            phi = self.bicgstab_solver(A_toSolve, b_toSolve, solverOptions)
        elif solver == 'minres':
            print("Using MinRES (iterative solver)")
            phi = self.minres_solver(A_toSolve, b_toSolve, solverOptions)

        print(f"Solution range: [{phi.min():.6e}, {phi.max():.6e}]")

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

        # Compute geometric properties if not done
        # if self.cell_volumes is None:
        #     self.compute_geometric_properties()

        # Build system
        if self.UseApproximateLaplacianFormulation:
            A, b = self.build_approx_fv_system()
        else:
            self.Mesh.build_DualControlVolumes()
            A, b = self.build_CV_fv_system()
            # exit(1)

        for component_idx in components:
            print(f"\n{'='*60}")
            print(f"Solving component {component_idx}")
            print('='*60)
            if self.UseApproximateLaplacianFormulation:
                A, b = self.apply_approx_BCs(A, b, component_idx=component_idx)
            else:
                A, b = self.apply_CV_BCs(A, b, component_idx=component_idx)
            phi = self.solve_poisson(A, b, solver, useReordering, solverOptions, component_idx=component_idx)
            residual = self.verify_solution(A, phi, b)
            
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
        
        print(f"Residual norm: {residual_norm:.6e}")
        print(f"Relative residual: {relative_residual:.6e}")
        print(f"Max residual: {np.abs(residual).max():.6e}")
        
        return residual
    
    def check_compatibility(self, b):
        """
        Check if RHS satisfies compatibility condition
        For pure Neumann: sum(b) should be ~0
        """
        b_sum = np.sum(b)
        print(f"RHS sum (should be ~0 for Neumann): {b_sum:.6e}")
        
        # Project RHS orthogonal to constant null space
        N = len(b)
        b_mean = np.mean(b)
        b_corrected = b - b_mean
        
        print(f"RHS mean removed: {b_mean:.6e}")
        return b_corrected
    
    def verify_null_space_removed(self, A, b):
        """
        Verify that constant vector is NOT in null space
        """
        N = A.shape[0]
        ones = np.ones(N)
        
        # A*ones should NOT be ~0 if null space is removed
        A_ones = A @ ones
        
        print(f"||A*ones|| = {np.linalg.norm(A_ones):.6e}")
        
        if np.linalg.norm(A_ones) < 1e-10:
            print("ERROR: Null space NOT removed! A*ones = 0")
            print("The pinned node constraint is not working.")
            return False
        else:
            print("Good: Null space appears removed")
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
            if not self.UseApproximateLaplacianFormulation:
                phi["DualControlVolume"] = self.Mesh.ControlVolumesPerNode

            # Create mesh
            mesh = meshio.Mesh(
                points=self.Mesh.Nodes,
                cells=cells,
                point_data=phi
            )

            mesh.write(output_file)
            print(f"Solution exported to {output_file}")

        except ImportError:
            print("meshio not available. Install with: pip install meshio")
            print("Saving as NumPy array instead...")
            np.save(output_file.replace('.vtu', '.npy'), phi)
            print(f"Solution saved to {output_file.replace('.vtu', '.npy')}")
        except Exception as e:
            print(f"Error exporting to VTK: {e}")
            print("Saving as NumPy array instead...")
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
                print("Building ILU preconditioner...")
                ilu = spilu(A.tocsc(), fill_factor=10, drop_tol=1e-4)
                M = LinearOperator(A.shape, ilu.solve)
                print("ILU: SUCCESS")
            except Exception as e:
                print(f"ILU failed: {e}, solving without preconditioner")
                M = None
        

        callback = None
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b)

        # Initial guess (not zeros for Neumann problems)
        x0 = np.random.randn(len(b))
        x0 = x0 - np.mean(x0)  # Remove constant component


        print("Starting BiCGSTAB...")
        x, info = bicgstab(A, b, 
                        x0=x0,
                        tol=tol, 
                        maxiter=maxiter,
                        M=M,
                        callback=callback)

        # Check result
        print(f"\nBiCGSTAB finished")
        
        if info == 0:
            print("✓ SUCCESS: Converged to tolerance")
            residual = np.linalg.norm(A @ x - b)
            print(f"Final residual: {residual:.6e}")
        elif info > 0:
            print(f"✗ FAILURE: Did not converge in {info} iterations")
            residual = np.linalg.norm(A @ x - b)
            print(f"Final residual: {residual:.6e}")
            print("Solution is NOT reliable - use direct solver instead")
        else:
            print("✗ FAILURE: Numerical breakdown")

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
                print("Building ILU preconditioner...")
                ilu = spilu(A.tocsc(), fill_factor=10, drop_tol=1e-4)
                M = LinearOperator(A.shape, ilu.solve)
                print("ILU: SUCCESS")
            except Exception as e:
                print(f"ILU failed: {e}, solving without preconditioner")
                M = None
        

        callback = None
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b)

        # Initial guess (not zeros for Neumann problems)
        x0 = np.random.randn(len(b))
        x0 = x0 - np.mean(x0)  # Remove constant component


        print("Starting MinRES...")
        x, info = minres(A, b, 
                        x0=x0,
                        tol=tol, 
                        maxiter=maxiter,
                        M=M,
                        callback=callback)

        # Check result
        print(f"\MinRES finished")
        
        if info == 0:
            print("✓ SUCCESS: Converged to tolerance")
            residual = np.linalg.norm(A @ x - b)
            print(f"Final residual: {residual:.6e}")
        elif info > 0:
            print(f"✗ FAILURE: Did not converge in {info} iterations")
            residual = np.linalg.norm(A @ x - b)
            print(f"Final residual: {residual:.6e}")
            print("Solution is NOT reliable - use direct solver instead")
        else:
            print("✗ FAILURE: Numerical breakdown")

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
                print("ILU preconditioner: Success")
            except Exception as e:
                print("ILU preconditioner failed:", e)
                M = None

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b)

        # Provide non-constant initial guess
        N = A.shape[0]
        x0 = np.random.randn(N)
        x0 = x0 - np.mean(x0)  # Remove constant component

        # Run GMRES
        print("Starting FGMRES solve...")
        x, info = fgmres(A, b, tol=tol, x0=x0, M=M, maxiter=maxiter, restart=restart, callback=callback)
        if verbose:
            print(f"FGMRES completed after {callback.niter} iterations. info={info}")
            if info != 0:
                print("Warning: FGMRES did not converge (info =", info, ")")
        else:
            if info != 0:
                print("Warning: FGMRES did not converge (info =", info, ")")

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
                print("ILU preconditioner: Success")
            except Exception as e:
                print("ILU preconditioner failed:", e)
                M = None

        # Use callback_type='x' if available (SciPy >=1.8)
        callback = None
        callback_type = 'x'  # or 'legacy' for old behavior
        if verbose:
            callback = self.UnifiedVerboseCallback(A, b)

        # Provide non-constant initial guess
        N = A.shape[0]
        x0 = np.random.randn(N)
        x0 = x0 - np.mean(x0)  # Remove constant component

        # Run GMRES
        print("Starting GMRES solve...")
        x, info = gmres(A, b, tol=tol, x0=x0, M=M, maxiter=maxiter, restart=restart, callback=callback, callback_type=callback_type)
        if verbose:
            print(f"GMRES completed after {callback.niter} iterations. info={info}")
            if info != 0:
                print("Warning: GMRES did not converge (info =", info, ")")
        else:
            if info != 0:
                print("Warning: GMRES did not converge (info =", info, ")")
        return x

    class UnifiedVerboseCallback:
        def __init__(self, A, b):
            self.niter = 0
            self.A = A
            self.b = b
            
        def __call__(self, xk):
            self.niter += 1
            
            # Compute residual from solution vector
            res = self.A @ xk - self.b
            res_norm = np.linalg.norm(res)
            
            # Print every iteration (or every N iterations to reduce output)
            if self.niter % 10 == 0 or self.niter == 1:
                print(f"Iter {self.niter}: residual = {res_norm:.6e}")
