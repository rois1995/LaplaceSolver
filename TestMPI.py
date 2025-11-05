from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

print(f"Rank {rank}: Hello from process {rank}/{size}", flush=True)

# Try a simple operation
try:
    val = rank + 1
    result = comm.allreduce(val, op=MPI.SUM)
    if rank == 0:
        print(f"AllReduce result: {result}")
except Exception as e:
    print(f"Rank {rank}: AllReduce failed: {e}", flush=True)