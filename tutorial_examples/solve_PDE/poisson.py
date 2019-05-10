"""
Gauss' Law of Electrostatics
============================

Here we use the discretize package to solve for the electric potential and
electric fields in 2D that result from a static charge distribution. Starting with
Gauss' law and Faraday's law:
    
.. math::
    \\nabla \cdot \mathbf{e} = \frac{\rho}{\epsilon_0} \n
    \\nabla \\times \mathbf{e} = \mathbf{0} \;\;\; \Rightarrow \;\;\; \mathbf{e} = -\\nabla \phi 
    
By defining the scalar potential to be zero at the boundary, we must solve
the following differential equation:
    
.. math::
    \\nabla^2 \phi = -frac{\rho}{\epsilon_0} \n
    \phi \Big |_{\partial \Omega} = 0


"""

###############################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#


from discretize import TensorMesh
from pymatsolver import SolverLU, Pardiso
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
from discretize.utils import sdiag, speye, kron3, spzeros, ddx, av, av_extrap


###############################################
#
# Solving the Problem
# -------------------
#

# Create a tensor mesh
hx = np.ones(75)
hy = np.ones(75)
mesh = TensorMesh([hx, hy], 'CC')

# Define charge distribution at cell centers and average to nodes
xycc = mesh.gridCC
kneg = (xycc[:, 0]==-10) & (xycc[:, 1]==0)  # negative charge at (-10, 0)
kpos = (xycc[:, 0]==10) & (xycc[:, 1]==0)   # positive charge at (10, 0)

rho = np.zeros(mesh.nC)
rho[kneg] = -1.
rho[kpos] = 1.

An2cc = mesh.aveN2CC  # Nodes to centers averaging operator
RHS = - An2cc.T*rho   # Transpose goes centers to nodes

# Define nodal Laplacian and envoke boundary conditions
L = - mesh.nodalLaplacian
L = L.tolil()

bInd = mesh.nodalBoundaryInd  # Indecies of nodes on boundaries
bInd = (bInd[0]) | (bInd[1]) | (bInd[2]) | (bInd[3]) 

L[bInd, :] = np.zeros((np.sum(bInd), mesh.nN))
L[bInd, bInd] = 1

# LU factorization and solve
#AinvM = SolverLU(L)
AinvM = Pardiso(L)
phi = AinvM*RHS
E = - mesh.nodalGrad*phi

fig = plt.figure(figsize=(14, 4))

Ax1 = fig.add_subplot(131)
mesh.plotImage(rho, vType='CC', ax=Ax1)

Ax2 = fig.add_subplot(132)
mesh.plotImage(phi, vType='N', ax=Ax2)

Ax3 = fig.add_subplot(133)
mesh.plotImage(E, ax=Ax3, vType='E', view='vec',
               streamOpts={'color': 'w', 'density': 1.0})


