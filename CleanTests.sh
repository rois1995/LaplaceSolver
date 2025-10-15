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
declare -a TestCaseNames=("UnitSquare_Struct" "UnitSquare_UnstructAligned" "UnitSquare_Unstruct" "UnitSquare_UnstructMixedAligned")
declare -a SubTestCaseNames=("Coarse" "Medium" "Fine" "VeryFine" "Coarse" "Medium" "Fine" "VeryFine" "Coarse" "Medium" "Fine" "VeryFine" "Coarse" "Medium" "Fine" "VeryFine")
declare -a HowManySubCases=("4" "4" "4" "4")



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

      if [ ${HowManySubCases[$iTestCase]} -eq "1" ]
      then
        echo "Cleaning test case ${BigCaseName}/${TestCase}"
        mv Parameters_ToUse.py ../dummy.py && rm -r * && mv ../dummy.py Parameters_ToUse.py
      else
        SubCase=${SubTestCaseNames[$globalSubCase]}
        echo "Cleaning test case ${BigCaseName}/${TestCase}/${SubCase}"
        if [[ -d $SubCase ]]; then
          rm -rf $SubCase
        else
          echo "Test Case ${BigCaseName}/${TestCase}/${SubCase} already cleaned"
        fi
      fi
      


      globalSubCase=$((globalSubCase+1))

    done

  done

done
