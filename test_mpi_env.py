import sys
import os

print(f"Rank: {os.environ.get('OMPI_COMM_WORLD_RANK', 'N/A')}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

try:
    import petsc4py
    print(f"petsc4py location: {petsc4py.__file__}")
    print("✓ petsc4py imported successfully")
except ImportError as e:
    print(f"✗ petsc4py import failed: {e}")

try:
    import mpi4py
    print(f"mpi4py location: {mpi4py.__file__}")
    print("✓ mpi4py imported successfully")
except ImportError as e:
    print(f"✗ mpi4py import failed: {e}")