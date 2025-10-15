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

# declare -a BigCaseNames=("Neumann" "Dirichlet")
declare -a BigCaseNames=("Dirichlet")
declare -a TestCaseNames=("UnitSquare_Struct")
declare -a SubTestCaseNames=("Coarse" "Medium" "Fine" "VeryFine")
declare -a HowManySubCases=("4")
CleanFolder=0
Verbose=Silent


for (( iBigCase = 0; iBigCase < ${#BigCaseNames[@]}; iBigCase++ ))
do

  BigCaseName=${BigCaseNames[$iBigCase]}

  globalSubCase=0
  for (( iTestCase = 0; iTestCase < ${#TestCaseNames[@]}; iTestCase++ ))
  do

    TestCase=${TestCaseNames[$iTestCase]}

    for (( iSubTestCase = 0; iSubTestCase < ${HowManySubCases[$iTestCase]}; iSubTestCase++ ))
    do 

      SubCase=""
      cd ${HOME}/TestCases/${BigCaseName}/${TestCase}
      echo "Running test case ${BigCaseName}/${TestCase}/${SubCase}"

      if [ ${HowManySubCases[$iTestCase]} -eq "1" ]
      then
        createLink $HOME/Meshes/${TestCase}.cgns Mesh.cgns
      else
        SubCase=${SubTestCaseNames[$globalSubCase]}
        mkdir -p $SubCase
        cd $SubCase
        cp ../Parameters_ToUse.py .

        createLink $HOME/Meshes/${TestCase}_${SubCase}.cgns Mesh.cgns

      fi

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

        mv Parameters_ToUse.py ../dummy.py

        rm *.py

        if [ $CleanFolder == 1 ]
        then
          rm *
        fi

        mv ../dummy.py Parameters_ToUse.py

      else
        echo "Failed test ${BigCaseName}/${TestCase}/${SubCase}"
        break
      fi

      globalSubCase=$((globalSubCase+1))

    done

    if [ $Verbose != "Silent" ]
    then
      echo " "
    fi

    echo " "

  done

done
