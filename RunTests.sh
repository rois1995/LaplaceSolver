#!/bin/bash

function createLink() {
  if [[ -L $2 ]]; then
    unlink $2
  fi
  if [[ -f $2 ]]; then
    rm $2
  fi
  ln -s $1 $2
}

HOME="$(pwd)"

declare -a BigCaseNames=("Neumann" "Dirichlet")
# declare -a BigCaseNames=("Dirichlet")
# declare -a TestCaseNames=("UnitSquare_Struct" "UnitCube_Struct")
declare -a TestCaseNames=("UnitSquare_Struct" "UnitSquare_UnstructAligned" "UnitSquare_Unstruct" "UnitSquare_UnstructMixedAligned" "UnitSquare_UnstructMixed" "UnitCube_Struct" "UnitCube_OnlyTets" "UnitCube_OnlyPrisms")
# declare -a TestCaseNames=("UnitCube_Struct" "UnitCube_OnlyTets" "UnitCube_OnlyPrisms")
CleanFolder=0
Verbose=Silent


for (( iBigCase = 0; iBigCase < ${#BigCaseNames[@]}; iBigCase++ ))
do

  BigCaseName=${BigCaseNames[$iBigCase]}

  for (( iTestCase = 0; iTestCase < ${#TestCaseNames[@]}; iTestCase++ ))
  do

    TestCase=${TestCaseNames[$iTestCase]}

    iMesh=0

    for mesh in "Coarse" "Medium" "Fine" "VeryFine"
    do 

      cd ${HOME}/TestCases/${BigCaseName}/${TestCase}
      echo "Running test case ${BigCaseName}/${TestCase}/${mesh}"

      mkdir -p Solutions

      mkdir -p $mesh
      cd $mesh
      cp ../Parameters_ToUse.py .

      createLink $HOME/Meshes/${TestCase}_${mesh}.cgns Mesh.cgns


      cp -r $HOME/*.py . && rm Parameters.py && cp Parameters_ToUse.py Parameters.py

      if [ $Verbose == "Silent" ]
      then
        sed -i 's/^\s*verbose=.*$/verbose= False/' Parameters.py
      else
        sed -i 's/^\s*verbose=.*$/verbose= True/' Parameters.py
      fi

      source /home/rausa/PythonVirtualEnvironments/Python3.8.10/bin/activate

      python3 Main.py

      deactivate

      if grep -q "Solution exported to" "output.log"
      then
        echo "PASSED!"

        createLink ../${mesh}/$(ls *.vtu) ../Solutions/flow_$(printf '%05d' $iMesh).vtu

        mv Parameters_ToUse.py ../dummy.py

        rm *.py

        if [ $CleanFolder == 1 ]
        then
          rm *
        fi

        mv ../dummy.py Parameters_ToUse.py

      else
        echo "Failed test ${BigCaseName}/${TestCase}/${mesh}"
        break
      fi

    done

    iMesh=$((iMesh+1))

    if [ $Verbose != "Silent" ]
    then
      echo " "
    fi

    echo " "

  done

done
