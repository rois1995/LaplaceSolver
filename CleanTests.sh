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
# declare -a TestCaseNames=("UnitSquare_Struct" "UnitSquare_UnstructAligned" "UnitSquare_Unstruct" "UnitSquare_UnstructMixedAligned" "UnitSquare_UnstructMixed" "UnitCube_OnlyTets" "UnitCube_OnlyPrisms")
declare -a TestCaseNames=("UnitSquare_Struct" "UnitSquare_Unstruct" "UnitCube_Struct" "UnitCube_OnlyTets" "UnitCube_OnlyPrisms" "UnitCube_WithTetsAndPyras")



for (( iBigCase = 0; iBigCase < ${#BigCaseNames[@]}; iBigCase++ ))
do

  BigCaseName=${BigCaseNames[$iBigCase]}

  for (( iTestCase = 0; iTestCase < ${#TestCaseNames[@]}; iTestCase++ ))
  do

    TestCase=${TestCaseNames[$iTestCase]}

    for mesh in "Coarse" "Medium" "Fine" "VeryFine"
    do 

      cd ${HOME}/TestCases/${BigCaseName}/${TestCase}


      echo "Cleaning test case ${BigCaseName}/${TestCase}/${mesh}"
      if [[ -d $mesh ]]; then
        rm -rf $mesh
      else
        echo "Test Case ${BigCaseName}/${TestCase}/${Meshes} already cleaned"
      fi
    

    done

  done

done
