NSLOTS=2


export OMP_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export VECLIB_MAXIMUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

PYTHON_ENV="/home/rausa/PythonVirtualEnvironments/Python3.8.10"
PETSCDIR="/home/rausa/Software/petsc"
MPIEXECDIR=${PETSCDIR}"/build/bin"

export PYTHONPATH=${PYTHON_ENV}/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=${PETSCDIR}/build/lib:$PYTHONPATH

PARALLELSOLVE=1

sed -i 's/^\s*solveParallel=.*$/solveParallel= False/' Parameters.py
if (( PARALLELSOLVE == 1 ))
then
    sed -i 's/^\s*solveParallel=.*$/solveParallel= True/' Parameters.py
fi

source ${PYTHON_ENV}/bin/activate

python3 Main.py

if (( PARALLELSOLVE == 1 ))
then
    # ${MPIEXECDIR}/mpiexec -n $NSLOTS python3 TestMPI.py
    ${MPIEXECDIR}/mpiexec -n $NSLOTS python3 parallel_Solve.py
fi

deactivate
