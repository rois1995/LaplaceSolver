import os
import matplotlib.pyplot as plt
import numpy as np

BigCases = ["Neumann", "Dirichlet"]
# TestCases = ["UnitSquare_Struct", "UnitCube_Struct", "UniteSquare_UnstructAligned"]
TestCases = ["UnitSquare_Struct", "UnitSquare_UnstructAligned", "UnitSquare_Unstruct",  "UnitSquare_UnstructMixedAligned"]
Meshes = ["Coarse", "Medium", "Fine", "VeryFine"]
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(TestCases)))
CaseDim = [2, 2, 2, 2]

for BigCase in BigCases:

    plt.figure(figsize=(8, 6))

    for iTestCase, TestCase in enumerate(TestCases):
        
        ErrorVec = []
        NOfPointsVec = []

        for Mesh in Meshes:

            filename = "TestCases/"+BigCase+"/"+TestCase+"/"+Mesh+"/Conv.log"

            data = np.loadtxt(filename)

            NOfPoints = int(data[0])
            Res = float(data[1])
            
            ErrorVec += [Res]
            NOfPointsVec += [NOfPoints]
        
        ds = 1.0/(np.array(NOfPointsVec)**(1.0/CaseDim[iTestCase]))
        plt.plot(ds, ErrorVec, color=colors[iTestCase], label=TestCase,linewidth=2.0)
        print(f"{TestCase} has errors {ErrorVec}")

    plt.plot(ds, ds, color='r', label="First Order",linewidth=2.0)
    plt.plot(ds, ds**2, color='k', label="Second Order",linewidth=2.0)
    plt.xscale('log')      # Logarithmic x-axis
    plt.yscale('log')      # Logarithmic y-axis
    plt.title(BigCase+" BCs")
    plt.legend()
    plt.savefig(BigCase+".eps")
    plt.savefig(BigCase+".png")
    # plt.show()

        
