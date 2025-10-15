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
      echo "Cleaning test case ${BigCaseName}/${TestCase}/${SubCase}"

      if [ ${HowManySubCases[$iTestCase]} -eq "1" ]
      then
         mv Parameters_ToUse.py ../dummy.py && rm -r * && mv ../dummy.py Parameters_ToUse.py
      else
        SubCase=${SubTestCaseNames[$globalSubCase]}
        rm -rf $SubCase

      fi


      globalSubCase=$((globalSubCase+1))

    done

  done

done
