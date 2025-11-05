"""
Complete parallel solve workflow with VTK export.
Run with: mpiexec -n <nprocs> python solve_parallel_with_vtk.py
"""
from mpi4py import MPI
import os
# os.environ['PETSC_OPTIONS'] = '-info -log_view -mat_view ::ascii_info'
os.environ['PETSC_COMM_TSALLREDUCE'] = '0'
os.environ['OMPI_MCA_coll_tuned_use_dynamic_rules'] = '0'
os.environ['OMPI_MCA_coll_base_verbose'] = '100'  # Debug MPI collective ops
import petsc4py
# petsc4py.init(['-log_view']) # optionally petsc4py.init(['-info', '-log_view'])
petsc4py.init(['-log_view']) # optionally petsc4py.init(['-info', '-log_view'])
from petsc4py import PETSc

# # 🔧 Force initialization of PETSc option and logging subsystems
# opts = PETSc.Options()
# _ = opts.getAll()  # triggers internal PETScInitializeOptions

import h5py
import numpy as np
from scipy.sparse import csr_matrix
import time
import sys
Path2Scripts="./Scripts"
sys.path.insert(1, Path2Scripts)
from Utils import setup_logging


def load_from_hdf5(filenameMatrix, filenameMesh, comm, Logger):
    """Load matrix, RHS, and mesh from HDF5."""
    rank = comm.rank
    
    if rank == 0:
        Logger.info(f"Loading system from {filenameMatrix}...")
    
    with h5py.File(filenameMatrix, 'r') as f:
        # Matrix
        data = f['matrix/data'][:]
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        n = f['matrix'].attrs['n']
        nnz_per_row = f['matrix/nnz_per_row'][:]


        if rank == 0:
            Logger.info(f"  Raw data types:")
            Logger.info(f"    data: {data.dtype}")
            Logger.info(f"    indices: {indices.dtype}")
            Logger.info(f"    indptr: {indptr.dtype}")
        
        # CRITICAL: Ensure correct dtypes for PETSc
        data = np.ascontiguousarray(data, dtype=np.float64)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        nnz_per_row = np.ascontiguousarray(nnz_per_row, dtype=np.int32)
        
        if rank == 0:
            Logger.info(f"  Converted dtypes:")
            Logger.info(f"    data: {data.dtype}")
            Logger.info(f"    indices: {indices.dtype}")
            Logger.info(f"    indptr: {indptr.dtype}")
        
        # RHS
        b = np.ascontiguousarray(f['b'][:], dtype=np.float64)
        # Perm
        perm = np.ascontiguousarray(f['perm'][:], dtype=np.int32) if 'perm' in f else None

        
    if rank == 0:
        Logger.info(f"Loading mesh from {filenameMesh}...")

    with h5py.File(filenameMesh, 'r') as f:
        
        # Mesh
        nodes = np.ascontiguousarray(f['mesh/nodes'][:], dtype=np.float64)
        
        # Elements
        elements = {}
        for section_name in f['mesh/elements'].keys():
            section_grp = f['mesh/elements'][section_name]
            elements[section_name] = {
                'type': int(section_grp.attrs['type']),
                'connectivity': np.ascontiguousarray(
                    section_grp['connectivity'][:], 
                    dtype=np.int32
                )
            }
        
        # Optional data
        control_volumes = (
            np.ascontiguousarray(f['mesh/DualControlVolume'][:], dtype=np.float64) 
            if 'mesh/DualControlVolume' in f else None
        )
        boundary_normals = (
            np.ascontiguousarray(f['mesh/boundary_normals'][:], dtype=np.float64) 
            if 'mesh/boundary_normals' in f else None
        )
        ExactSolution = (
            np.ascontiguousarray(f['mesh/ExactSolution'][:], dtype=np.float64) 
            if 'mesh/ExactSolution' in f else None
        )
        

    # Validate CSR structure
    if rank == 0:
        Logger.info(f"\n  Validating CSR structure:")
        Logger.info(f"    Matrix shape: {n}")
        Logger.info(f"    indptr shape: {indptr.shape}")
        Logger.info(f"    indices shape: {indices.shape}")
        Logger.info(f"    data shape: {data.shape}")
        
        assert len(indptr) == n + 1, f"indptr length {len(indptr)} != n+1 {n+1}"
        assert len(indices) == len(data), f"indices {len(indices)} != data {len(data)}"
        assert indptr[0] == 0, f"indptr[0] = {indptr[0]} != 0"
        assert indptr[-1] == len(data), f"indptr[-1] = {indptr[-1]} != len(data) {len(data)}"
        assert np.all(indices >= 0), "Negative indices found!"
        assert np.all(indices < n), f"Indices >= {n} found!"
        Logger.info(f"    ✓ CSR structure valid")


    # Reconstruct matrix
    A_scipy = csr_matrix((data, indices, indptr), shape=(n, n))
    
    mesh_data = {
        'nodes': nodes,
        'elements': elements,
        'control_volumes': control_volumes,
        'boundary_normals': boundary_normals,
        'ExactSolution': ExactSolution
    }
    
    if rank == 0:
        Logger.info(f"  Matrix: {n:,} unknowns")
        Logger.info(f"  Mesh: {nodes.shape[0]:,} nodes")
    
    return A_scipy, b, perm, nnz_per_row, mesh_data

def create_petsc_matrix_optimal(A_scipy, nnz_per_row, comm, Logger):
    """Create distributed PETSc matrix - same as before."""

    rank = comm.rank
    n_global = A_scipy.shape[0]
    A_csr = A_scipy.tocsr()

    if rank == 0:
        Logger.info(f"  Input matrix validation:")
        Logger.info(f"    Format: {A_csr.format}")
        Logger.info(f"    Shape: {A_csr.shape}")
        Logger.info(f"    NNZ: {A_csr.nnz}")
        Logger.info(f"    data dtype: {A_csr.data.dtype}")
        Logger.info(f"    indices dtype: {A_csr.indices.dtype}")
        Logger.info(f"    indptr dtype: {A_csr.indptr.dtype}")
    
    A = PETSc.Mat().create(comm=comm)
    A.setSizes([n_global, n_global])
    A.setType(PETSc.Mat.Type.MPIAIJ)
    A.setUp()
    
    rstart, rend = A.getOwnershipRange()
    n_local = rend - rstart
    
    # Preallocation
    d_nnz = np.zeros(n_local, dtype=np.int32)
    o_nnz = np.zeros(n_local, dtype=np.int32)
    
    for i in range(rstart, rend):
        cols = A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]]
        d_nnz[i - rstart] = np.sum((cols >= rstart) & (cols < rend))
        o_nnz[i - rstart] = np.sum((cols < rstart) | (cols >= rend))
    
    if rank == 0:
        Logger.info(f"    ✓ Preallocation computed")
        Logger.info(f"    d_nnz: min={d_nnz.min()}, mean={d_nnz.mean():.1f}, max={d_nnz.max()}")
        Logger.info(f"    o_nnz: min={o_nnz.min()}, mean={o_nnz.mean():.1f}, max={o_nnz.max()}")
    

    A.setPreallocationNNZ([d_nnz, o_nnz])
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    
    if rank == 0:
        Logger.info(f"  Preallocation set")
    
    # Fill matrix values with error checking
    if rank == 0:
        Logger.info(f"  Filling matrix values...")

    try:
        for i in range(rstart, rend):
            row_start = A_csr.indptr[i]
            row_end = A_csr.indptr[i + 1]
            
            cols = A_csr.indices[row_start:row_end]
            vals = A_csr.data[row_start:row_end]
            
            # Ensure correct dtypes
            cols = np.ascontiguousarray(cols, dtype=np.int32)
            vals = np.ascontiguousarray(vals, dtype=np.float64)
            
            if len(cols) > 0:
                A.setValues(i, cols, vals)
    
    except Exception as e:
        if rank == 0:
            Logger.error(f"ERROR during matrix fill: {e}")
            import traceback
            traceback.print_exc()
        comm.Abort()
    
    if rank == 0:
        Logger.info(f"  Values filled, assembling...")
        
    A.assemblyBegin()

    if rank == 0:
        Logger.info(f"  assembly Begin done! Now Assembly end...")

    A.assemblyEnd()
    
    if rank == 0:
        info = A.getInfo()
        Logger.info(f"  Matrix assembled: {info['mallocs']} mallocs")

    
    
    return A

def setup_solver(ksp, comm, Logger):
    """Configure HYPRE BoomerAMG - same as before."""
    rank = comm.rank
    
    if rank == 0:
        Logger.info("\nConfiguring solver...")
    
    ksp.setType(PETSc.KSP.Type.CG)
    
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    pc.setHYPREType('boomeramg')
    
    opts = PETSc.Options()
    opts.setValue('-pc_hypre_boomeramg_coarsen_type', 'HMIS')
    opts.setValue('-pc_hypre_boomeramg_strong_threshold', '0.5')
    opts.setValue('-pc_hypre_boomeramg_interp_type', 'ext+i')
    opts.setValue('-pc_hypre_boomeramg_P_max', '4')
    opts.setValue('-pc_hypre_boomeramg_agg_nl', '1')
    opts.setValue('-pc_hypre_boomeramg_relax_type_all', 'symmetric-SOR/Jacobi')
    opts.setValue('-pc_hypre_boomeramg_max_levels', '25')
    
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
    ksp.setFromOptions()

def solve_system(A, b_numpy, comm, Logger):
    """Solve linear system with extra debugging."""
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        Logger.info("\n" + "="*70)
        Logger.info("ENTERING SOLVE_SYSTEM")
        Logger.info("="*70)
    
    try:
        if rank == 0:
            Logger.info("Step 1: Creating vectors...")
        
        # Step 1: Create vectors properly
        x = A.createVecRight()
        b = A.createVecLeft()
        
        if rank == 0:
            Logger.info("  ✓ Vectors created")
            Logger.info("Step 2: Getting local ownership...")
        
        # Step 2: Get local ownership
        rstart, rend = A.getOwnershipRange()
        n_local = rend - rstart
        
        if rank == 0:
            Logger.info(f"  ✓ Matrix size: {A.getSize()[0]:,}")
            Logger.info(f"  ✓ Local rows: {n_local}")
            Logger.info("Step 3: Verifying sizes...")
        
        # Step 3: Verify sizes match
        assert A.getSize()[0] == len(b_numpy), \
            f"Matrix size {A.getSize()[0]} != b_numpy size {len(b_numpy)}"
        
        if rank == 0:
            Logger.info(f"  ✓ Sizes match")
            Logger.info("Step 4: Setting RHS values...")
        
        # Step 4: Set RHS values
        b_local = np.ascontiguousarray(
            b_numpy[rstart:rend], 
            dtype=np.float64
        )
        
        local_indices = np.arange(rstart, rend, dtype=np.int32)
        b.setValues(local_indices, b_local)
        
        if rank == 0:
            Logger.info(f"  ✓ RHS values set ({len(b_local)} values)")
            Logger.info("Step 5: Assembly end (without explicit begin)...")
        
        # Skip assemblyBegin() - just do assemblyEnd()
        # This is sufficient for simple setValues operations
        b.assemblyEnd()
        if rank == 0:
            bnorm = b.norm()
            Logger.info(f"  ✓ assemblyEnd() succeeded, norm = {bnorm:.6e}")
        
        if rank == 0:
            Logger.info("Step 6: Creating KSP solver...")
        
        # Step 6: Create KSP
        ksp = PETSc.KSP().create(comm=comm)
        if rank == 0:
            Logger.info("  ✓ KSP created")
        
        if rank == 0:
            Logger.info("Step 7: Setting operators...")
        
        ksp.setOperators(A)
        if rank == 0:
            Logger.info("  ✓ Operators set")
        
        if rank == 0:
            Logger.info("Step 8: Setting KSP type...")
        
        ksp.setType(PETSc.KSP.Type.CG)
        if rank == 0:
            Logger.info("  ✓ KSP type set to CG")
        
        if rank == 0:
            Logger.info("Step 9: Setting preconditioner...")
        
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.HYPRE)
        pc.setHYPREType('boomeramg')
        if rank == 0:
            Logger.info("  ✓ Preconditioner set to HYPRE BoomerAMG")
        
        if rank == 0:
            Logger.info("Step 10: Setting HYPRE options...")
        
        opts = PETSc.Options()
        opts.setValue('-pc_hypre_boomeramg_coarsen_type', 'HMIS')
        opts.setValue('-pc_hypre_boomeramg_strong_threshold', '0.5')
        opts.setValue('-pc_hypre_boomeramg_interp_type', 'ext+i')
        opts.setValue('-pc_hypre_boomeramg_P_max', '4')
        opts.setValue('-pc_hypre_boomeramg_agg_nl', '1')
        opts.setValue('-pc_hypre_boomeramg_max_levels', '25')
        if rank == 0:
            Logger.info("  ✓ HYPRE options set")
        
        if rank == 0:
            Logger.info("Step 11: Setting convergence criteria...")
        
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
        if rank == 0:
            Logger.info("  ✓ Tolerances set")
        
        if rank == 0:
            Logger.info("Step 12: setFromOptions()...")
        
        ksp.setFromOptions()
        if rank == 0:
            Logger.info("  ✓ setFromOptions() succeeded")
        
        # Monitor
        iteration = [0]
        def monitor(ksp, its, rnorm):
            iteration[0] = its
            if rank == 0 and its % 10 == 0:
                Logger.info(f"    Iteration {its:4d}: residual = {rnorm:.6e}")
        
        ksp.setMonitor(monitor)
        
        if rank == 0:
            Logger.info("\n" + "="*70)
            Logger.info("STARTING SOLVE")
            Logger.info("="*70 + "\n")
        
        comm.Barrier()
        t_start = time.time()
        
        if rank == 0:
            Logger.info("About to call ksp.solve()...")
        
        ksp.solve(b, x)
        
        if rank == 0:
            Logger.info("ksp.solve() returned successfully!")
        
        comm.Barrier()
        t_solve = time.time() - t_start
        
        # Get info
        its = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        rnorm = ksp.getResidualNorm()
        
        if rank == 0:
            Logger.info(f"\n" + "="*70)
            Logger.info("SOLVE COMPLETED")
            Logger.info("="*70)
            conv_msg = "YES ✓" if reason > 0 else "NO ✗"
            Logger.info(f"Converged: {conv_msg}")
            Logger.info(f"Iterations: {its}")
            Logger.info(f"Final residual: {rnorm:.6e}")
            Logger.info(f"Solve time: {t_solve:.2f} seconds")
            Logger.info("="*70)
        
        if rank == 0:
            Logger.info(f"\nReturning solution vector x...")
        
        return x
    
    except Exception as e:
        if rank == 0:
            Logger.error(f"\n!!! EXCEPTION IN SOLVE_SYSTEM !!!")
            Logger.error(f"Exception type: {type(e).__name__}")
            Logger.error(f"Exception message: {e}")
            import traceback
            Logger.error("\nFull traceback:")
            traceback.print_exc()
        
        if rank == 0:
            Logger.info("\nCalling comm.Abort()...")
        
        comm.Barrier()
        comm.Abort()
        return None
    
def export_solution_vtk(nodes, elements, solution_dict, output_file):
    """
    Export solution to VTK format using meshio.
    This runs on rank 0 only.
    
    Parameters:
    -----------
    nodes : array (N x 3)
        Node coordinates
    elements : dict
        Element connectivity by section
    solution_dict : dict
        Dictionary of field_name: field_values
    output_file : str
        Output VTK file (.vtu)
    """
    try:
        import meshio
        
        print(f"\nExporting to VTK: {output_file}")
        
        # Prepare cells
        cells = []
        for section_name, elem_data in elements.items():
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
                print(f"  Warning: Unsupported element type {elem_type}")
                continue
            
            cell_type, nodes_per_elem = cell_type_map[elem_type]
            cells.append((cell_type, connectivity))
        
        # Create mesh
        mesh = meshio.Mesh(
            points=nodes,
            cells=cells,
            point_data=solution_dict
        )
        
        mesh.write(output_file)
        print(f"  Solution exported successfully")
        print(f"  Fields: {list(solution_dict.keys())}")
        
    except ImportError:
        print("ERROR: meshio not available. Install with: pip install meshio")
        print("Saving as HDF5 instead...")
        
        with h5py.File(output_file.replace('.vtu', '.h5'), 'w') as f:
            f.create_dataset('nodes', data=nodes)
            for field_name, field_values in solution_dict.items():
                f.create_dataset(field_name, data=field_values)
        
        print(f"  Saved to {output_file.replace('.vtu', '.h5')}")
    
    except Exception as e:
        print(f"ERROR exporting to VTK: {e}")
        import traceback
        traceback.print_exc()

def gather_solution(x_petsc, comm):
    """Gather distributed solution to rank 0."""
    rank = comm.rank
    
    x_local = x_petsc.getArray().copy()
    x_gathered = comm.gather(x_local, root=0)
    
    if rank == 0:
        return np.concatenate(x_gathered)
    else:
        return None

def solve_system_serial_workaround(A_scipy, b_numpy, comm, Logger):
    """
    Solve on rank 0 serially, then distribute solution.
    Bypasses PETSc matrix assembly issues.
    """
    rank = comm.rank
    size = comm.size
    
    if rank == 0:
        print("\nUsing serial solve workaround (rank 0 only)...")
    
    # Gather full matrix and RHS to rank 0
    A_full = comm.gather(A_scipy if rank == 0 else None, root=0)
    b_full = comm.gather(b_numpy if rank == 0 else None, root=0)
    
    if rank == 0:
        # Combine on rank 0
        from scipy.sparse import vstack, hstack
        
        print("  Combining matrix pieces on rank 0...")

        
        # Stack all pieces
        A_combined = vstack([A_full[i] for i in range(size)], format='csr')
        b_combined = np.array(b_full[0])
        n = A_combined.shape[0]

        print(f"  Combined matrix: {A_combined.shape}")
        print(f"  Combined b: {b_combined.shape}")
        print(f"  Solving with PETSc (serial)...")
        
        # Create PETSc matrix WITHOUT distributed operations
        A_petsc = PETSc.Mat().create(comm=MPI.COMM_SELF)  # Serial MPI comm!
        A_petsc.setSizes([A_combined.shape[0], A_combined.shape[1]])
        A_petsc.setType(PETSc.Mat.Type.SEQAIJ)
        A_petsc.setUp()
        
        # Fill matrix
        A_csr = A_combined.tocsr()
        for i in range(A_combined.shape[0]):
            cols = A_csr.indices[A_csr.indptr[i]:A_csr.indptr[i+1]]
            vals = A_csr.data[A_csr.indptr[i]:A_csr.indptr[i+1]]
            if len(cols) > 0:
                A_petsc.setValues(i, np.ascontiguousarray(cols, dtype=np.int32),
                                 np.ascontiguousarray(vals, dtype=np.float64))
        
        A_petsc.assemblyBegin()
        A_petsc.assemblyEnd()
        
        print("  ✓ Matrix assembled (serial)")
        
        # Create vectors
        x = A_petsc.createVecRight()
        b_petsc = A_petsc.createVecLeft()

        # ===== CREATE INITIAL GUESS =====
        print("\n  Creating random zero-mean initial guess...")
        
        np.random.seed(42)  # For reproducibility
        x_random = np.random.randn(n)
        
        # Remove mean
        x_mean = np.mean(x_random)
        x_zeromean = x_random - x_mean
        
        print(f"    Mean (before): {np.mean(x_random):.6e}")
        print(f"    Mean (after): {np.mean(x_zeromean):.6e}")
        print(f"    Std dev: {np.std(x_zeromean):.6e}")
        print(f"    Min/Max: {x_zeromean.min():.6e} / {x_zeromean.max():.6e}")
        
        # Set initial guess
        x.setValues(range(n), x_zeromean)
        x.assemblyBegin()
        x.assemblyEnd()
        
        x_norm = x.norm()
        print(f"    Initial guess norm: {x_norm:.6e}")
        
        # Set RHS
        b_petsc.setValues(range(len(b_combined)), b_combined)
        b_petsc.assemblyBegin()
        b_petsc.assemblyEnd()

        print("  ✓ b vector assembled (serial)")
        
        check_matrix_quality(A_combined, b_combined, rank_0_only=True)

        print("\nAnalyzing problem type...")
        is_pure_neumann, max_row_sum, b_sum, is_singular, is_compatible = detect_pure_neumann(A_scipy, b_numpy, tolerance=1e-9)

        print("Matrix Analysis:")
        print(f"  Max row sum: {max_row_sum:.6e} (should be ~0)")
        print(f"  Singular: {is_singular}")
        
        print("\nRHS Analysis:")
        print(f"  RHS sum: {b_sum:.6e} (should be ~0)")
        print(f"  Compatible: {is_compatible}")

        ksp = PETSc.KSP().create(comm=MPI.COMM_SELF)
        ksp.setOperators(A_petsc)
        opts = PETSc.Options()

        if is_pure_neumann:
            print("  ✓ Pure Neumann problem detected!")
            print(f"    Max row sum: {max_row_sum:.6e}")
            print(f"    RHS sum: {b_sum:.6e}")

            nullvec = A_petsc.createVecRight()
            nullvec.setValues(range(len(b_combined)), [1.0]*len(b_combined))
            nullvec.assemblyBegin()
            nullvec.assemblyEnd()
            nullvec.normalize()
            nullsp = PETSc.NullSpace().create(vectors=[nullvec], comm=MPI.COMM_SELF)

            A_petsc.setNullSpace(nullsp)

            # Make b compatible
            nullsp.remove(b_petsc)
        
        else:
            print("  ✓ Mixed/Dirichlet problem detected")
            print(f"    Max row sum: {max_row_sum:.6e}")
            print(f"    RHS sum: {b_sum:.6e}")


        solver_type = "CG"
        # Mixed BC: use CG
        ksp.setType(PETSc.KSP.Type.CG)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.GAMG)
    
        # Set options BEFORE setFromOptions()
        opts.setValue('-pc_gamg_type', 'agg')
        opts.setValue('-pc_gamg_agg_nsmooths', '1')
        opts.setValue('-pc_gamg_threshold', '0.02')
        opts.setValue('-pc_gamg_coarse_eq_limit', '1000')
        print("  ✓ Using CG")

        # Setup solver (serial)
        
        # Solver options
        opts.setValue('-ksp_type', 'cg')
        opts.setValue('-ksp_rtol', '1e-10')
        opts.setValue('-ksp_atol', '1e-12')
        opts.setValue('-ksp_max_it', '500')
        opts.setValue('-ksp_monitor_true_residual', '')
        
        # NOW call setFromOptions to apply them
        ksp.setFromOptions()
        
        ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)
        ksp.setInitialGuessNonzero(True)

        def monitor(ksp, its, rnorm):
            if rank == 0 and (its == 0 or its % 5 == 0 or its > 490):
                print(f"    Iteration {its:4d}: residual = {rnorm:.6e}")
                if np.isnan(rnorm) or np.isinf(rnorm):
                    print(f"    WARNING: Residual is NaN/Inf at iteration {its}!")
        
        ksp.setMonitor(monitor)
        
        print("  Solving...")
        ksp.solve(b_petsc, x)
        
        its = ksp.getIterationNumber()
        reason = ksp.getConvergedReason()
        rnorm = ksp.getResidualNorm()
        
        print(f"\nResults:")
        print(f"  Iterations: {its}")
        print(f"  Final residual: {rnorm:.6e}")
        
        # Decode convergence reason
        if reason == 2:
            print("✓ Converged to relative tolerance")
        elif reason == 3:
            print("✓ Converged to absolute tolerance")
        elif reason == -3:
            print("✗ Stopped: max iterations reached")
        elif reason == -8:
            print("✗ Diverged: NaN or Inf in residual")
        else:
            print(f"? Unknown convergence reason: {reason}")
                
        # Get solution
        x_solution = x.getArray().copy()
    else:
        x_solution = None
    
    # Broadcast solution back to all processes
    x_solution = comm.bcast(x_solution, root=0)
    
    return x_solution

def detect_pure_neumann(A_scipy, b_numpy, tolerance=1e-10):
    """
    Detect if this is a pure Neumann problem.
    
    Two conditions for pure Neumann:
    1. Matrix rows sum to ~0 (matrix is singular)
    2. RHS sum to ~0 (compatibility condition)
    """
    A_csr = A_scipy.tocsr()
    
    # Condition 1: Check row sums
    row_sums = np.array(A_csr.sum(axis=1)).flatten()
    max_row_sum = np.max(np.abs(row_sums))
    
    # Condition 2: Check RHS sum (compatibility)
    b_sum = np.sum(b_numpy)
    
    # Both conditions must be satisfied
    is_singular = max_row_sum < tolerance
    is_compatible = np.abs(b_sum) < tolerance
    
    is_pure_neumann = is_singular and is_compatible
    
    return is_pure_neumann, max_row_sum, b_sum, is_singular, is_compatible

def check_matrix_quality(A_combined, b_combined, rank_0_only=True):
    """Diagnose matrix and RHS issues."""
    rank = 0  # Serial solve is on rank 0
    n = A_combined.shape[0]
    
    # ===== DIAGNOSTICS ON SCIPY MATRIX =====
    print("\n" + "="*70)
    print("MATRIX/RHS QUALITY CHECK")
    print("="*70)
    
    A_csr = A_combined.tocsr()
    
    print("\nMatrix Analysis:")
    print(f"  Shape: {A_csr.shape}")
    print(f"  Non-zeros: {A_csr.nnz}")
    print(f"  Sparsity: {100*(1 - A_csr.nnz/(n*n)):.2f}%")
    print(f"  Contains NaN: {np.isnan(A_csr.data).any()}")
    print(f"  Contains Inf: {np.isinf(A_csr.data).any()}")
    print(f"  Min value: {np.min(A_csr.data):.6e}")
    print(f"  Max value: {np.max(A_csr.data):.6e}")
    print(f"  Mean value: {np.mean(A_csr.data):.6e}")
    
    # Check symmetry (only for small matrices)
    if n <= 5000:
        print(f"\n  Checking symmetry (n={n})...")
        A_T = A_csr.T
        diff = (A_csr - A_T).data
        max_diff = np.max(np.abs(diff)) if len(diff) > 0 else 0
        print(f"    Max symmetry error: {max_diff:.6e}")
        print(f"    Is symmetric: {max_diff < 1e-10}")
    
    print("\nRHS Analysis:")
    print(f"  Size: {len(b_combined)}")
    print(f"  Contains NaN: {np.isnan(b_combined).any()}")
    print(f"  Contains Inf: {np.isinf(b_combined).any()}")
    print(f"  Min value: {np.min(b_combined):.6e}")
    print(f"  Max value: {np.max(b_combined):.6e}")
    print(f"  Mean value: {np.mean(b_combined):.6e}")
    print(f"  Norm: {np.linalg.norm(b_combined):.6e}")
    
    # Check diagonal dominance
    print("\nDiagonal Analysis:")
    diag = A_csr.diagonal()
    print(f"  Diagonal contains NaN: {np.isnan(diag).any()}")
    print(f"  Diagonal contains Inf: {np.isinf(diag).any()}")
    print(f"  Diagonal min: {np.min(diag):.6e}")
    print(f"  Diagonal max: {np.max(diag):.6e}")
    print(f"  Diagonal mean: {np.mean(diag):.6e}")
    
    print("\n" + "="*70)
    
    # If matrix looks bad, stop here
    if np.isnan(A_csr.data).any() or np.isnan(b_combined).any():
        print("\nERROR: Matrix or RHS contains NaN values!")
        print("Cannot proceed with solve. Check data assembly.")
        return None
    
    if np.isinf(A_csr.data).any() or np.isinf(b_combined).any():
        print("\nERROR: Matrix or RHS contains Inf values!")
        print("Cannot proceed with solve. Check data assembly.")
        return None


def main():
    """Main workflow."""
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size


    Logger = setup_logging(filenameOut='outputSolver.log', filenameErr='errorSolver.log', verbose=True)
    
    if rank == 0:
        print("="*70)
        print("PARALLEL POISSON SOLVER WITH VTK EXPORT")
        print("="*70)
        print(f"MPI processes: {size}")

    # if rank == 0:
    #     print("\nAvailable preconditioners:")
    #     # Try to list available PC types
    #     for pc_type in ['hypre', 'gamg', 'ilu', 'bjacobi', 'jacobi', 'sor', 'mg']:
    #         try:
    #             pc_test = PETSc.PC().create()
    #             pc_test.setType(getattr(PETSc.PC.Type, pc_type.upper()))
    #             print(f"  ✓ {pc_type}")
    #         except:
    #             print(f"  ✗ {pc_type}")

    # exit(1)
    
    # Load data
    A_scipy, b, perm, nnz_per_row, mesh_data = load_from_hdf5('matrix_data.h5', 'Mesh_data.h5', comm, Logger)
    
    # Convert to PETSc
    # if rank == 0:
    #     print("\nConverting to PETSc format...")
    # A_petsc = create_petsc_matrix_optimal(A_scipy, nnz_per_row, comm, Logger)
    # del A_scipy  # Free memory
    
    # Solve
    # x_petsc = solve_system(A_petsc, b, comm, Logger)
    x_full = solve_system_serial_workaround(A_scipy, b, comm, Logger)


    if rank == 0:
        print("\nExporting to VTK...")
        
        if perm is not None:
            phi_reordered = np.empty_like(x_full)
            phi_reordered[perm] = x_full
            x_full = phi_reordered
        
        solution_dict = {'phi': x_full}
        if mesh_data['control_volumes'] is not None:
            solution_dict['DualControlVolume'] = mesh_data['control_volumes']
        if mesh_data['ExactSolution'] is not None:
            solution_dict['ExactSolution'] = mesh_data['ExactSolution']
        
        export_solution_vtk(
            nodes=mesh_data['nodes'],
            elements=mesh_data['elements'],
            solution_dict=solution_dict,
            output_file='solution.vtu'
        )
        
        print("Done!")

    exit(1)
    
    # Gather solution to rank 0
    if rank == 0:
        print("\nGathering solution...")
    x_full = gather_solution(x_petsc, comm)
    
    # Export VTK (rank 0 only)
    if rank == 0:
        # Prepare solution dictionary
        solution_dict = {
            'phi': x_full,  # Your main solution field
        }
        
        # Add control volumes if available
        if mesh_data['control_volumes'] is not None:
            solution_dict['DualControlVolume'] = mesh_data['control_volumes']
        
        # Add boundary normals if available
        if mesh_data['boundary_normals'] is not None:
            solution_dict['Normal_X'] = mesh_data['boundary_normals'][:, 0]
            solution_dict['Normal_Y'] = mesh_data['boundary_normals'][:, 1]
            solution_dict['Normal_Z'] = mesh_data['boundary_normals'][:, 2]
        if not mesh_data['ExactSolution'] == None:
            solution_dict['ExactSolution'] = mesh_data['ExactSolution']
        
        # Add exact solution if you have it
        # solution_dict['ExactSolution'] = exact_solution_function(...)
        
        # Export
        export_solution_vtk(
            nodes=mesh_data['nodes'],
            elements=mesh_data['elements'],
            solution_dict=solution_dict,
            output_file='solution.vtu'
        )
        
        print("\n" + "="*70)
        print("COMPLETE!")
        print("="*70)

if __name__ == '__main__':
    main()

    petsc4py.finalize()
